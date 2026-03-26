import argparse
import os

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from models import Model
from targets import PiecewiseLinearTarget
from utils import (
    calculate_per_expert_loss,
    gating_gradient_norm,
    get_device,
    per_expert_gradient_norm,
    sample_uniformly,
)
from visualization import (
    export_training_animation_visualization,
)


def load_config(argv: list[str] | None = None) -> DictConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args, overrides = parser.parse_known_args(argv)

    file_cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_dotlist(overrides)
    return OmegaConf.merge(file_cfg, cli_cfg)


def main() -> None:
    cfg = load_config()
    device = get_device()

    target_cfg = cfg.target
    domain = target_cfg.domain
    target_function = PiecewiseLinearTarget(**target_cfg, device=device)

    model_cfg = cfg.model
    model = Model(**model_cfg).to(device)

    training_cfg = cfg.training
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg.learning_rate)
    loss_fn = nn.MSELoss()

    viz_num_points = 200
    viz_x = torch.linspace(
        domain[0], domain[1], viz_num_points, device=device
    ).unsqueeze(-1)
    viz_frames = []

    output_dir = cfg.output_dir or "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    use_wandb = cfg.wandb.enabled
    if use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            tags=cfg.wandb.run_tags,
            config=cfg,
        )

    pbar = tqdm(range(training_cfg.num_steps), desc="Training")
    for step in pbar:
        # ===== TRAINING =====
        x = sample_uniformly(domain, training_cfg.batch_size).to(device)
        y = target_function(x)

        output = model(x)
        loss = loss_fn(output["predictions"], y)

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
                    "gating_grad_norm": gating_gradient_norm(model),
                }
            )

        # ===== VISUALIZATION SNAPSHOT =====
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
            }
        )

        optimizer.step()
        pbar.set_postfix({"loss": loss.item()})

    export_training_animation_visualization(
        viz_frames=viz_frames,
        viz_x=viz_x.cpu(),
        output_dir=output_dir,
        domain=tuple(domain),
        target_function=target_function,
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
        ]
        for name in gif_names:
            path = os.path.join(output_dir, f"{name}.gif")
            if os.path.exists(path):
                wandb.log({f"animations/{name}": wandb.Video(path, format="gif")})


if __name__ == "__main__":
    main()
