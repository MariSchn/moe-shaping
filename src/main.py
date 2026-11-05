import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from methods.bilevel import BilevelOptimizer, build_bilevel_optimizer
from methods.lola import LOLA
from models import Model
from router import BanditRouter, LinearRouter
from targets import ModelTarget, PiecewiseLinearTarget
from utils import (
    calculate_last_layer_regret,
    calculate_load_balancing_loss,
    calculate_max_vio,
    calculate_per_expert_loss,
    expert_load_metrics,
    gating_gradient_norm,
    get_device,
    per_expert_gradient_norm,
    sample_uniformly,
    set_seed,
)
from visualization import (
    expert_visualization,
    export_training_animation_visualization,
    model_visualization,
    per_expert_sample_count_visualization,
    router_visualization,
    routing_bias_visualization,
    target_router_visualization,
    target_visualization,
    top_expert_visualization,
)


def load_config() -> DictConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args, overrides = parser.parse_known_args()

    file_cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_dotlist(overrides)
    return OmegaConf.merge(file_cfg, cli_cfg)


def main() -> None:
    cfg = load_config()

    use_wandb = cfg.wandb.enabled
    if use_wandb:
        run_name = None
        if not os.environ.get("WANDB_SWEEP_ID"):
            run_name = cfg.wandb.run_name or None

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            tags=cfg.wandb.run_tags,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        # When running as part of a sweep, merge sweep parameters into config
        if wandb.run.sweep_id:
            sweep_cfg = OmegaConf.create(dict(wandb.config))
            cfg = OmegaConf.merge(cfg, sweep_cfg)

            # Name run after the swept parameters
            api = wandb.Api()
            sweep_def = api.sweep(
                f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.sweep_id}"
            )
            sweep_params = sweep_def.config.get("parameters", {})
            # Extract leaf parameter names and full paths from nested sweep config
            leaf_paths = []
            queue = [("", sweep_params)]
            while queue:
                prefix, params = queue.pop()
                for name, spec in params.items():
                    full_path = f"{prefix}.{name}" if prefix else name
                    if "parameters" in spec:
                        queue.append((full_path, spec["parameters"]))
                    else:
                        leaf_paths.append((name, full_path))

            parts = [
                f"{name}={OmegaConf.select(cfg, path)}"
                for name, path in sorted(leaf_paths)
                if OmegaConf.select(cfg, path) is not None
            ]
            if parts:
                wandb.run.name = "_".join(parts)
                cfg.output_dir = os.path.join(
                    os.path.dirname(cfg.output_dir),
                    "sweeps",
                    sweep_def.name,
                    wandb.run.name,
                )

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = get_device()

    target_cfg = cfg.target
    domain = target_cfg.domain
    target_kwargs = (
        OmegaConf.to_container(target_cfg.kwargs, resolve=True)
        if target_cfg.kwargs
        else {}
    )

    if target_cfg.type == "piecewise_linear":
        target_function = PiecewiseLinearTarget(
            domain=domain, device=device, **target_kwargs
        )
    elif target_cfg.type == "model":
        target_model = Model(**target_kwargs).to(device)
        target_function = ModelTarget(target_model)
    else:
        raise ValueError(f"Unknown target type: {target_cfg.type}")

    model_cfg = cfg.model
    freeze_router = model_cfg.get("freeze_router", False)
    freeze_experts = model_cfg.get("freeze_experts", False)
    model_init_cfg = OmegaConf.to_container(model_cfg, resolve=True)
    model_init_cfg.pop("freeze_router", None)
    model_init_cfg.pop("freeze_experts", None)
    model = Model(**model_init_cfg).to(device)

    if freeze_router:
        for p in model.router.parameters():
            p.requires_grad_(False)
    if freeze_experts:
        for p in model.experts.parameters():
            p.requires_grad_(False)
    if use_wandb:
        wandb.summary["num_parameters"] = sum(p.numel() for p in model.parameters())
        wandb.summary["num_trainable_parameters"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

    training_cfg = cfg.training
    optimizer_type = training_cfg.get("optimizer", "SGD")
    optimizer_cls = getattr(torch.optim, optimizer_type)

    optimizer = build_bilevel_optimizer(
        model, optimizer_cls, training_cfg.learning_rate, training_cfg.bilevel
    )
    if optimizer is None:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optimizer_cls(trainable_params, lr=training_cfg.learning_rate)
    loss_fn = nn.MSELoss()

    lola = LOLA(**training_cfg.lola)

    viz_frames = []

    viz_num_points = 200
    viz_x = torch.linspace(
        domain[0], domain[1], viz_num_points, device=device
    ).unsqueeze(-1)

    output_dir = cfg.output_dir or "./outputs"
    init_dir = os.path.join(output_dir, "init")
    final_dir = os.path.join(output_dir, "final")
    anim_dir = os.path.join(output_dir, "anims")
    for d in (init_dir, final_dir, anim_dir):
        os.makedirs(d, exist_ok=True)

    def log_static_visualizations(subdir, panel_prefix):
        figs = {}
        if isinstance(model.router, LinearRouter):
            figs["routing_bias"] = routing_bias_visualization(model)["figure"]

        if model.input_dim == 1:
            viz_kwargs = dict(
                model=model,
                domain=tuple(domain),
                num_points=viz_num_points,
                target_function=target_function,
            )
            figs["model"] = model_visualization(**viz_kwargs)["figure"]
            figs["top_expert"] = top_expert_visualization(**viz_kwargs)["figure"]
            if isinstance(model.router, LinearRouter):
                figs["router"] = router_visualization(**viz_kwargs)["figure"]
            figs["expert"] = expert_visualization(**viz_kwargs)["figure"]
            figs["per_expert_sample_count"] = per_expert_sample_count_visualization(
                model=model,
                domain=tuple(domain),
                num_points=viz_num_points,
            )["figure"]
            figs["target"] = target_visualization(
                target_function, tuple(domain), viz_num_points
            )["figure"]
            target_router = target_router_visualization(
                target_function, tuple(domain), viz_num_points
            )
            if target_router is not None:
                figs["target_router"] = target_router["figure"]
            if isinstance(target_function, ModelTarget):
                figs["target_expert"] = expert_visualization(
                    model=target_function.model,
                    domain=tuple(domain),
                    num_points=viz_num_points,
                    target_function=target_function,
                )["figure"]
                figs["target_per_expert_sample_count"] = (
                    per_expert_sample_count_visualization(
                        model=target_function.model,
                        domain=tuple(domain),
                        num_points=viz_num_points,
                    )["figure"]
                )

        for name, fig in figs.items():
            fig.savefig(
                os.path.join(subdir, f"{name}.png"), dpi=150, bbox_inches="tight"
            )
        if use_wandb:
            wandb.log(
                {
                    f"{panel_prefix}/{name}": wandb.Image(fig)
                    for name, fig in figs.items()
                }
            )
        for fig in figs.values():
            plt.close(fig)

    log_static_visualizations(init_dir, "initialization")

    # Running sums of per-step routing regret (cumulative bandit regret).
    cumulative_regret_gating = 0.0
    cumulative_regret_optimal = 0.0

    pbar = tqdm(range(training_cfg.num_steps), desc="Training")
    for step in pbar:
        # ===== TRAINING =====
        x = sample_uniformly(domain, training_cfg.batch_size, model_cfg.input_dim).to(
            device
        )
        y = target_function(x)

        output = model(x)
        loss = loss_fn(output["predictions"], y)

        # Always calculate load balancing loss for logging, but only add it to the loss if the weight is positive.
        load_balancing_loss = calculate_load_balancing_loss(
            output["gating_scores"], output["selected_experts"]
        )
        if training_cfg.load_balancing_loss_weight > 0:
            loss += training_cfg.load_balancing_loss_weight * load_balancing_loss

        optimizer.zero_grad()
        loss.backward()
        lola.shape_gradients(model, output, x, y)

        model.router.update(
            x,
            output["selected_experts"],
            (output["predictions"].detach() - y).pow(2).mean(dim=-1),
        )

        # ===== LOGGING =====
        per_expert_grad_norms = per_expert_gradient_norm(model)
        per_expert_losses = calculate_per_expert_loss(
            output["expert_outputs"].detach(),
            output["selected_experts"],
            y,
            torch.nn.functional.mse_loss,
        ).cpu()

        if use_wandb:
            regret_gating, regret_optimal = calculate_last_layer_regret(
                output["expert_outputs"].detach(),
                output["gating_scores"].detach(),
                output["predictions"].detach(),
                y,
                model_cfg.router_activation,
                combo_size=model_cfg.router_top_k,
            )
            cumulative_regret_gating += regret_gating
            cumulative_regret_optimal += regret_optimal

            wandb.log(
                {
                    "loss": loss.item(),
                    "bandit/last_layer_regret_gating": cumulative_regret_gating,
                    "bandit/last_layer_regret_optimal": cumulative_regret_optimal,
                    "expert_imbalance": load_balancing_loss.item(),
                    "max_vio": calculate_max_vio(
                        output["selected_experts"], model.num_experts
                    ),
                    **expert_load_metrics(
                        output["selected_experts"], model.num_experts
                    ),
                    "gating_grad_norm": gating_gradient_norm(model),
                    "expert_grad_norm": sum(n * n for n in per_expert_grad_norms)
                    ** 0.5,
                    **lola.metrics(model),
                    **model.router.metrics(),
                    **(
                        optimizer.metrics()
                        if isinstance(optimizer, BilevelOptimizer)
                        else {}
                    ),
                }
            )

        # ===== VISUALIZATION SNAPSHOT =====
        if step % cfg.logging.anim_sampling_rate == 0:
            frame = {
                "step": step,
                "per_expert_loss": per_expert_losses,
                "per_expert_grad_norm": per_expert_grad_norms,
                "train_selected_experts": output["selected_experts"].detach().cpu(),
            }
            if isinstance(model.router, LinearRouter):
                frame["routing_biases"] = model.router.routing_biases.detach().cpu()
            if model.input_dim == 1:
                with torch.inference_mode():
                    viz_output = model(viz_x)
                    if isinstance(model.router, BanditRouter):
                        frame["index_terms"] = torch.stack(
                            model.router.index_terms(viz_x)
                        ).cpu()
                frame["predictions"] = viz_output["predictions"].cpu()
                frame["gating_scores"] = viz_output["gating_scores"].cpu()
                frame["expert_outputs"] = viz_output["expert_outputs"].cpu()
                frame["selected_experts"] = viz_output["selected_experts"].cpu()
            viz_frames.append(frame)

        optimizer.step()
        if isinstance(model.router, LinearRouter):
            model.router.update_routing_biases(
                output["selected_experts"],
                training_cfg.auxiliarly_loss_free_load_balancing_gamma,
            )
        pbar.set_postfix({"loss": loss.item()})

    log_static_visualizations(final_dir, "final")

    export_training_animation_visualization(
        viz_frames=viz_frames,
        viz_x=viz_x.cpu(),
        output_dir=anim_dir,
        domain=tuple(domain),
        target_function=target_function,
        fps=len(viz_frames) // 10,
    )

    if use_wandb:
        anim_names = [
            "model",
            "top_expert",
            "router",
            "expert",
            "per_expert_loss",
            "per_expert_grad_norm",
            "per_expert_sample_count",
            "routing_biases",
            "index",
        ]
        log_dict = {}
        for name in anim_names:
            path = os.path.join(anim_dir, f"{name}.mp4")
            if os.path.exists(path):
                log_dict[f"animations/{name}"] = wandb.Video(path, format="mp4")
        routing_map_path = os.path.join(anim_dir, "routing_map.png")
        if os.path.exists(routing_map_path):
            log_dict["animations/routing_map"] = wandb.Image(routing_map_path)
        wandb.log(log_dict)


if __name__ == "__main__":
    main()
