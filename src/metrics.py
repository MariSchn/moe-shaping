import itertools
from typing import Callable, Iterable, Tuple

import torch

from models import ExpertBank, MoELayer, MoEOutput


def load_balancing_loss(
    gating_scores: torch.Tensor,
    selected_experts: torch.Tensor,
) -> torch.Tensor:
    """
    Auxiliary load balancing loss to encourage uniform expert utilization.

    Follows Switch Transformer: L = num_experts * sum_i(f_i * P_i)
    where f_i is the fraction of tokens dispatched to expert i (non-differentiable)
    and P_i is the mean softmax router probability for expert i (differentiable).

    Args:
        gating_scores: (B, num_experts) raw router logits
        selected_experts: (B, top_k) indices of selected experts

    Returns:
        Scalar load balancing loss.
    """
    num_experts = gating_scores.shape[1]
    P = torch.softmax(gating_scores, dim=-1).mean(dim=0)
    f = expert_load(selected_experts, num_experts)
    return num_experts * (f * P).sum()


# Quantiles (in [0, 1]) of the per-expert load distribution, logged as
# "load/p10".."load/p90" alongside "load/min" (the p0 extreme) and "load/max"
# (p100). Mirrors _LOAD_QUANTILES in the Megatron-LM patch (lm/patches/0004).
LOAD_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


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
    """Per-expert load distribution summary, mirroring the Megatron-LM expert-load
    logging (lm/patches/0004): "load/min", "load/max", "load/pXX", and MaxVio
    (Wang et al., 2024), (max_i load_i - mean load) / mean load."""
    load = expert_load(selected_experts, num_experts)
    quantiles = torch.quantile(load, torch.tensor(LOAD_QUANTILES, device=load.device))
    return {
        "load/max": load.max().item(),
        "load/min": load.min().item(),
        **{
            f"load/p{round(q * 100)}": v
            for q, v in zip(LOAD_QUANTILES, quantiles.tolist())
        },
        "max_vio": ((load.max() - load.mean()) / load.mean()).item(),
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
) -> Tuple[float, float]:
    """
    Bandit-style routing regret for the last MoE layer.

    Treats each sample as a context, the router's choice of which ``combo_size``
    experts to combine as the action, and negative MSE as the reward. Regret is
    the router's actual prediction loss minus the loss of the best candidate
    expert combination, averaged over the batch (per-sample regret).

    Two oracles are returned, differing in how a candidate combination is mixed:

    - ``regret_gating``: candidate experts mixed with the router's own
      ``gate_weights`` of their gating scores. Isolates the routing *choice*
      regret with the gating mechanism fixed.
    - ``regret_optimal``: candidate experts mixed with the loss-minimizing convex
      weight (closed form, ``combo_size == 2`` only). Stronger oracle that also
      captures weighting suboptimality. Returns ``nan`` if ``combo_size != 2``.

    By construction ``regret_optimal >= regret_gating >= 0``.

    Args:
        expert_outputs: (B, num_experts, output_dim)
        gating_scores: (B, num_experts)
        predictions: (B, output_dim) router's actual combined output
        targets: (B, output_dim)
        gate_weights: the router's `Router.gate_weights`.
        combo_size: Number of experts per candidate combination (top-k).

    Returns:
        (regret_gating, regret_optimal) as Python floats.
    """
    num_experts = expert_outputs.shape[1]

    # Router's actual per-sample loss (the action it took).
    chosen_loss = ((predictions - targets) ** 2).mean(dim=-1)  # (B,)

    # All candidate combinations of experts -> (C, combo_size).
    combos = list(itertools.combinations(range(num_experts), combo_size))
    combo_idx = torch.tensor(combos, dtype=torch.long, device=expert_outputs.device)

    sc = gating_scores[:, combo_idx]  # (B, C, combo_size)
    outs = expert_outputs[:, combo_idx, :]  # (B, C, combo_size, output_dim)
    y = targets[:, None, :]  # (B, 1, output_dim)

    # ----- Gating oracle: the router's gate weights over the combination.
    gating_pred = (outs * gate_weights(sc).unsqueeze(-1)).sum(dim=2)
    gating_losses = ((gating_pred - y) ** 2).mean(dim=-1)  # (B, C)
    gating_best = gating_losses.min(dim=1).values  # (B,)
    regret_gating = torch.relu(chosen_loss - gating_best).mean().item()

    # ----- Optimal oracle: best convex weight per pair (combo_size == 2 only).
    if combo_size == 2:
        a = outs[:, :, 0, :]  # (B, C, output_dim)
        b = outs[:, :, 1, :]
        d = a - b
        num = (d * (y - b)).sum(dim=-1)  # (B, C)
        den = (d * d).sum(dim=-1).clamp(min=1e-12)  # (B, C)
        w = (num / den).clamp(0.0, 1.0).unsqueeze(-1)  # (B, C, 1)
        optimal_pred = w * a + (1.0 - w) * b  # (B, C, output_dim)
        optimal_losses = ((optimal_pred - y) ** 2).mean(dim=-1)  # (B, C)
        optimal_best = optimal_losses.min(dim=1).values  # (B,)
        regret_optimal = torch.relu(chosen_loss - optimal_best).mean().item()
    else:
        regret_optimal = float("nan")

    return regret_gating, regret_optimal


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """L2 norm of the parameters' gradients, 0 without any."""
    sum_sq = sum(p.grad.pow(2).sum() for p in parameters if p.grad is not None)
    return float(sum_sq) ** 0.5


def per_expert_grad_norm(experts: ExpertBank) -> torch.Tensor:
    """L2 norm of each expert's parameter gradients, ``(num_experts,)``."""
    sum_sq = sum(p.grad.pow(2).flatten(1).sum(dim=1) for p in experts.parameters())
    return sum_sq.sqrt()


def layer_metrics(
    layer: MoELayer, output: MoEOutput, load_balancing_loss: torch.Tensor
) -> dict[str, float]:
    return {
        "load_balancing_loss": load_balancing_loss.item(),
        **load_metrics(output.selected_experts, layer.router.num_experts),
        "router_grad_norm": grad_norm(layer.router.parameters()),
        "expert_grad_norm": grad_norm(layer.experts.parameters()),
        **layer.router.metrics(),
    }
