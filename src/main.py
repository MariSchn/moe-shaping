import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


from targets import PiecewiseLinearTarget
from models import Model
from utils import get_device, calculate_per_expert_loss, sample_uniformly
from visualization import (
    model_visualization,
    top_expert_visualization,
    router_visualization,
    expert_visualization,
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

    logging_cfg = cfg.logging
    wandb_cfg = cfg.wandb
    if wandb_cfg.enabled:
        wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            name=wandb_cfg.run_name,
            tags=wandb_cfg.run_tags,
            config=cfg,
        )
        expert_loss_steps: list[int] = []
        expert_loss_ys: list[list[float]] = [[] for _ in range(model_cfg.num_experts)]
        expert_loss_keys = [f"expert_{e}" for e in range(model_cfg.num_experts)]

    pbar = tqdm(range(training_cfg.num_steps), desc="Training")
    for step in pbar:
        # ===== TRAINING =====
        x = sample_uniformly(domain, training_cfg.batch_size).to(device)
        y = target_function(x)

        output = model(x)
        loss = loss_fn(output["predictions"], y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({"loss": loss.item()})

        # ===== LOGGING =====
        if wandb_cfg.enabled:
            log_dict = {
                "train/loss": loss.item(),
            }

            per_expert_loss = (
                calculate_per_expert_loss(
                    output["expert_outputs"],
                    output["selected_experts"],
                    y,
                    torch.nn.functional.mse_loss,
                )
                .detach()
                .cpu()
                .tolist()
            )

            if step % logging_cfg.log_plots_every_n_steps == 0:
                fig = model_visualization(model, domain, 100, target_function)["figure"]
                log_dict["plots/model"] = wandb.Image(fig)
                plt.close(fig)

                fig = top_expert_visualization(model, domain, 100, target_function)[
                    "figure"
                ]
                log_dict["plots/top_expert"] = wandb.Image(fig)
                plt.close(fig)

                fig = router_visualization(model, domain, 100, target_function)[
                    "figure"
                ]
                log_dict["plots/router"] = wandb.Image(fig)
                plt.close(fig)

                fig = expert_visualization(model, domain, 100)["figure"]
                log_dict["plots/expert"] = wandb.Image(fig)
                plt.close(fig)

            expert_loss_steps.append(step)
            for e, v in enumerate(per_expert_loss):
                expert_loss_ys[e].append(v)

            wandb.log(log_dict, step=step)

    if wandb_cfg.enabled:
        if expert_loss_steps:
            wandb.log(
                {
                    "per_expert_loss": wandb.plot.line_series(
                        expert_loss_steps,
                        expert_loss_ys,
                        keys=expert_loss_keys,
                        title="Per-Expert Loss",
                        xname="step",
                    ),
                },
                step=expert_loss_steps[-1],
            )
        wandb.finish()


if __name__ == "__main__":
    main()
