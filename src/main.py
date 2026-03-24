import argparse

import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from typing import Tuple

from utils import get_device, calculate_per_expert_loss, sample_uniformly


class PiecewiseLinearTarget:
    def __init__(
        self,
        domain: Tuple[float, float],
        num_pieces: int,
        device: torch.device,
        slopes: torch.Tensor = None,
        intercepts: torch.Tensor = None,
        breakpoints: torch.Tensor = None,
    ) -> None:
        self.domain = domain
        self.num_pieces = num_pieces
        self.device = device

        if slopes is None:
            slopes = torch.randn(num_pieces, device=device)
        if intercepts is None:
            intercepts = torch.randn(num_pieces, device=device)
        if breakpoints is None:
            breakpoints = torch.linspace(
                domain[0], domain[1], num_pieces + 1, device=device
            )
        else:
            breakpoints = breakpoints.to(device)
        slopes = slopes.to(device)
        intercepts = intercepts.to(device)

        assert slopes.shape == (num_pieces,), (
            f"Slopes must be a tensor of shape ({num_pieces},)"
        )
        assert intercepts.shape == (num_pieces,), (
            f"Intercepts must be a tensor of shape ({num_pieces},)"
        )
        assert breakpoints.shape == (num_pieces + 1,), (
            f"Breakpoints must be a tensor of shape ({num_pieces + 1},)"
        )

        self.slopes = slopes
        self.intercepts = intercepts
        self.breakpoints = breakpoints

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        indices = torch.bucketize(x, self.breakpoints) - 1
        indices = indices.clamp(0, self.num_pieces - 1)

        slope = self.slopes[indices]
        intercept = self.intercepts[indices]
        return slope * x + intercept


class Model(nn.Module):
    def __init__(
        self, num_experts: int, input_dim: int, output_dim: int, router_top_k: int
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.router_top_k = router_top_k

        self.experts = nn.ModuleList(
            [nn.Linear(input_dim, output_dim, bias=True) for _ in range(num_experts)]
        )
        self.gating_function = nn.Linear(input_dim, num_experts, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        assert x.ndim == 2, f"Input must be a 2D tensor (B, D), got {x.ndim}D"
        assert x.shape[1] == self.input_dim, (
            f"Input must have {self.input_dim} features, got {x.shape[1]} features"
        )

        # Routing
        gating_scores = self.gating_function(x)  # (B, num_experts)
        top_k_scores, top_k_indices = torch.topk(
            gating_scores, self.router_top_k, dim=1
        )  # (B, top_k) each

        # Run all experts on the input (inefficient, but simple)
        # Shape: (B, num_experts, output_dim)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # Get the outputs from the top k experts.
        top_k_expanded = top_k_indices.unsqueeze(-1).expand(
            -1, -1, expert_outputs.size(-1)
        )
        selected_outputs = torch.gather(expert_outputs, dim=1, index=top_k_expanded)

        # Softmax only over the top-k experts, not all experts.
        selected_gate_probs = torch.softmax(top_k_scores, dim=1)  # (B, top_k)

        # Combine expert outputs: (B, output_dim)
        weighted = selected_outputs * selected_gate_probs.unsqueeze(-1)

        out = {
            "predictions": weighted.sum(dim=1),  # (B, output_dim)
            "gating_scores": gating_scores,  # (B, num_experts)
            "expert_outputs": expert_outputs,  # (B, num_experts, output_dim)
            "selected_experts": top_k_indices,  # (B, top_k)
        }

        return out


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

    wandb_cfg = OmegaConf.select(cfg, "wandb")
    if wandb_cfg.enabled:
        wandb.init(
            project=wandb_cfg.project,
            entity=OmegaConf.select(wandb_cfg, "entity", default=None),
            name=OmegaConf.select(wandb_cfg, "run_name", default=None),
            tags=OmegaConf.select(wandb_cfg, "run_tags", default=[]),
            config=OmegaConf.to_container(cfg, resolve=True),
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

            expert_loss_steps.append(step)
            for e, v in enumerate(per_expert_loss):
                expert_loss_ys[e].append(v)

            wandb.log({"loss": loss.item()}, step=step)

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
