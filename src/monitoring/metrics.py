import itertools
from typing import Callable, Iterable

import torch

from model.moe import MoELayer, MoEOutput


def expert_load(selected_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Per-expert load = fraction of the batch's tokens routed to each expert.

    With top-k routing the loads sum to top_k, so perfect balance is
    top_k/num_experts per expert; load -> 1 means an expert funnels (almost)
    every token, load -> 0 means it is (nearly) dead.

    Args:
        selected_experts: (B, top_k) indices of the selected experts.
        num_experts: total number of experts E.

    Returns:
        (num_experts,) load per expert.
    """
    counts = torch.bincount(selected_experts.flatten(), minlength=num_experts)
    return counts.float() / selected_experts.shape[0]


def load_metrics(selected_experts: torch.Tensor, num_experts: int) -> dict[str, float]:
    """The minimum load, for dead experts, and MaxVio (Wang et al., 2024),
    (max_i load_i - mean load) / mean load. The mean load is always
    top_k / num_experts, so MaxVio stands in for the max load."""
    load = expert_load(selected_experts, num_experts)
    return {
        "load/min": load.min().item(),
        "load/max_vio": ((load.max() - load.mean()) / load.mean()).item(),
    }


def per_expert_loss(
    expert_outputs: torch.Tensor,
    selected_experts: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    MSE of each expert's outputs, over the samples it was selected for.

    Args:
        expert_outputs: (B, num_experts, output_dim)
        selected_experts: (B, top_k)
        targets: (B, output_dim)

    Returns:
        The loss for each expert. (num_experts,)
    """
    losses = (expert_outputs - targets.unsqueeze(1)).pow(2).mean(dim=-1)  # (B, E)
    mask = torch.zeros_like(losses).scatter_(1, selected_experts, 1.0)
    # Avoid division by zero for experts that weren't selected.
    return (losses * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1.0)


def routing_regret(
    expert_outputs: torch.Tensor,
    gating_scores: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    gate_weights: Callable[[torch.Tensor], torch.Tensor],
    combo_size: int,
) -> float:
    """
    Bandit-style routing regret for the last MoE layer.

    Treats each sample as a context, the router's choice of which ``combo_size``
    experts to combine as the action, and negative MSE as the reward. Regret is
    the router's actual prediction loss minus the loss of the best candidate
    expert combination, mixed with the router's own ``gate_weights`` of their
    gating scores, averaged over the batch. This isolates the routing *choice*
    with the gating mechanism fixed.

    Args:
        expert_outputs: (B, num_experts, output_dim)
        gating_scores: (B, num_experts)
        predictions: (B, output_dim) router's actual combined output
        targets: (B, output_dim)
        gate_weights: the router's `Router.gate_weights`.
        combo_size: Number of experts per candidate combination (top-k).
    """
    num_experts = expert_outputs.shape[1]

    # Router's actual per-sample loss (the action it took).
    chosen_loss = ((predictions - targets) ** 2).mean(dim=-1)  # (B,)

    # All candidate combinations of experts -> (C, combo_size).
    combos = list(itertools.combinations(range(num_experts), combo_size))
    combo_idx = torch.tensor(combos, dtype=torch.long, device=expert_outputs.device)

    sc = gating_scores[:, combo_idx]  # (B, C, combo_size)
    outs = expert_outputs[:, combo_idx, :]  # (B, C, combo_size, output_dim)
    pred = (outs * gate_weights(sc).unsqueeze(-1)).sum(dim=2)  # (B, C, output_dim)
    losses = ((pred - targets[:, None, :]) ** 2).mean(dim=-1)  # (B, C)
    best = losses.min(dim=1).values  # (B,)
    return torch.relu(chosen_loss - best).mean().item()


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """L2 norm of the parameters' gradients, 0 without any."""
    sum_sq = sum(p.grad.pow(2).sum() for p in parameters if p.grad is not None)
    return float(sum_sq) ** 0.5


def layer_metrics(layer: MoELayer, output: MoEOutput) -> dict[str, float]:
    """Metrics of one MoE layer, keyed ``<topic>/<name>``."""
    return {
        **load_metrics(output.selected_experts, layer.router.num_experts),
        "grad_norm/router": grad_norm(layer.router.parameters()),
        "grad_norm/experts": grad_norm(layer.experts.parameters()),
        **{f"router/{k}": v for k, v in layer.router.metrics().items()},
    }
