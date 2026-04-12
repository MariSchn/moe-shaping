import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from models import Model
from targets import ModelTarget, PiecewiseLinearTarget
from utils import (
    calculate_load_balancing_loss,
    calculate_per_expert_loss,
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
        for p in model.gating_function.parameters():
            p.requires_grad_(False)
    if freeze_experts:
        for p in model.experts.parameters():
            p.requires_grad_(False)

    training_cfg = cfg.training
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(trainable_params, lr=training_cfg.learning_rate)
    loss_fn = nn.MSELoss()

    can_visualize = model_cfg.input_dim == 1
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
        viz_kwargs = dict(
            model=model,
            domain=tuple(domain),
            num_points=viz_num_points,
            target_function=target_function,
        )
        figs = {
            "model": model_visualization(**viz_kwargs)["figure"],
            "top_expert": top_expert_visualization(**viz_kwargs)["figure"],
            "router": router_visualization(**viz_kwargs)["figure"],
            "expert": expert_visualization(**viz_kwargs)["figure"],
            "routing_bias": routing_bias_visualization(model)["figure"],
            "target": target_visualization(
                target_function, tuple(domain), viz_num_points
            )["figure"],
        }
        target_router = target_router_visualization(
            target_function, tuple(domain), viz_num_points
        )
        if target_router is not None:
            figs["target_router"] = target_router["figure"]

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

    if can_visualize:
        log_static_visualizations(init_dir, "initialization")

    pbar = tqdm(range(training_cfg.num_steps), desc="Training")
    for step in pbar:
        # ===== TRAINING =====
        x = sample_uniformly(domain, training_cfg.batch_size, model_cfg.input_dim).to(
            device
        )
        y = target_function(x)

        output = model(x)
        loss = loss_fn(output["predictions"], y)

        if training_cfg.load_balancing_loss_weight > 0:
            load_balancing_loss = calculate_load_balancing_loss(
                output["gating_scores"], output["selected_experts"]
            )
            loss += training_cfg.load_balancing_loss_weight * load_balancing_loss
        else:
            load_balancing_loss = None

        optimizer.zero_grad()
        loss.backward()

        # ===== LOGGING =====
        per_expert_grad_norms = per_expert_gradient_norm(model)
        per_expert_losses = calculate_per_expert_loss(
            output["expert_outputs"].detach(),
            output["selected_experts"],
            y,
            torch.nn.functional.mse_loss,
        ).cpu()

        if use_wandb:
            wandb.log(
                {
                    "loss": loss.item(),
                    "load_balancing_loss": load_balancing_loss.item()
                    if load_balancing_loss is not None
                    else None,
                    "gating_grad_norm": gating_gradient_norm(model),
                }
            )

        # ===== VISUALIZATION SNAPSHOT =====
        if can_visualize and step % cfg.logging.anim_sampling_rate == 0:
            with torch.inference_mode():
                viz_output = model(viz_x)
            viz_frames.append(
                {
                    "step": step,
                    "predictions": viz_output["predictions"].cpu(),
                    "gating_scores": viz_output["gating_scores"].cpu(),
                    "expert_outputs": viz_output["expert_outputs"].cpu(),
                    "selected_experts": viz_output["selected_experts"].cpu(),
                    "per_expert_loss": per_expert_losses,
                    "per_expert_grad_norm": per_expert_grad_norms,
                    "train_selected_experts": output["selected_experts"].detach().cpu(),
                    "routing_biases": model.routing_biases.detach().cpu(),
                }
            )

        optimizer.step()
        model.update_routing_biases(
            output["selected_experts"],
            training_cfg.auxiliarly_loss_free_load_balancing_gamma,
        )
        pbar.set_postfix({"loss": loss.item()})

    if can_visualize:
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
            gif_names = [
                "model",
                "top_expert",
                "router",
                "expert",
                "per_expert_loss",
                "per_expert_grad_norm",
                "per_expert_sample_count",
                "routing_biases",
            ]
            log_dict = {}
            for name in gif_names:
                path = os.path.join(anim_dir, f"{name}.mp4")
                if os.path.exists(path):
                    log_dict[f"animations/{name}"] = wandb.Video(path, format="mp4")
            wandb.log(log_dict)


if __name__ == "__main__":
    main()
