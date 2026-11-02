import random

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Tuple


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_device(model: nn.Module) -> torch.device:
    p = next(model.parameters(), None)
    if p is not None:
        return p.device
    return torch.device("cpu")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_uniformly(
    domain: Tuple[float, float], batch_size: int, input_dim: int = 1
) -> torch.Tensor:
    return (domain[1] - domain[0]) * torch.rand(batch_size, input_dim) + domain[0]


def calculate_load_balancing_loss(
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
    B, num_experts = gating_scores.shape

    # P_i: mean softmax probability for each expert over the batch (differentiable)
    router_probs = torch.softmax(gating_scores, dim=-1)  # (B, num_experts)
    P = router_probs.mean(dim=0)  # (num_experts,)

    # f_i: fraction of tokens routed to each expert (non-differentiable)
    expert_counts = torch.zeros(num_experts, device=gating_scores.device)
    expert_counts.scatter_add_(
        0,
        selected_experts.flatten(),
        torch.ones(selected_experts.numel(), device=gating_scores.device),
    )
    f = expert_counts / B

    return num_experts * (f * P).sum()


# Quantiles (in [0, 1]) of the per-expert load distribution, logged as
# "load/p10".."load/p90" alongside "load/min" (the p0 extreme) and "load/max"
# (p100). Mirrors _LOAD_QUANTILES in the Megatron-LM patch (lm/patches/0004).
LOAD_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def calculate_expert_load(
    selected_experts: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
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
    B = selected_experts.shape[0]
    expert_counts = torch.zeros(num_experts, device=selected_experts.device)
    expert_counts.scatter_add_(
        0,
        selected_experts.flatten(),
        torch.ones(selected_experts.numel(), device=selected_experts.device),
    )
    return expert_counts / B


def calculate_max_vio(
    selected_experts: torch.Tensor,
    num_experts: int,
) -> float:
    """MaxVio (Wang et al., 2024): (max_i load_i - mean load) / mean load."""
    load = calculate_expert_load(selected_experts, num_experts)
    return ((load.max() - load.mean()) / load.mean()).item()


def expert_load_metrics(
    selected_experts: torch.Tensor,
    num_experts: int,
) -> dict[str, float]:
    """Per-expert load distribution summary: "load/min", "load/max", "load/pXX".

    Mirrors the Megatron-LM expert-load logging (lm/patches/0004). The toy model
    has a single MoE layer, so the per-expert load vector is itself the pool over
    which the quantiles are taken (load/min == p0, load/max == p100).

    Args:
        selected_experts: (B, top_k) indices of the selected experts.
        num_experts: total number of experts E.

    Returns:
        A dict of scalar load metrics keyed by name (own "load/" wandb panel).
    """
    load = calculate_expert_load(selected_experts, num_experts)
    metrics = {
        "load/max": load.max().item(),
        "load/min": load.min().item(),
    }
    quantiles = torch.quantile(load, torch.tensor(LOAD_QUANTILES, device=load.device))
    for q, qv in zip(LOAD_QUANTILES, quantiles.tolist()):
        metrics[f"load/p{int(round(q * 100))}"] = qv
    return metrics


def calculate_per_expert_loss(
    expert_outputs: torch.Tensor,
    selected_experts: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Callable = torch.nn.functional.mse_loss,
) -> torch.Tensor:
    """
    Calculate the loss for each expert individually.
    Loss is only calculated over the samples where the expert was selected among the top-k experts.

    Args:
        expert_outputs: (B, num_experts, output_dim)
        selected_experts: (B, top_k)
        targets: (B, 1)
        loss_fn: The loss function to use.

    Returns:
        The loss for each expert. (num_experts,)
    """

    assert expert_outputs.ndim == 3, (
        f"Expert outputs must be a 3D tensor (B, num_experts, output_dim), got {expert_outputs.ndim}D of shape {expert_outputs.shape}"
    )
    assert selected_experts.ndim == 2, (
        f"Selected experts must be a 2D tensor (B, top_k), got {selected_experts.ndim}D of shape {selected_experts.shape}"
    )
    assert targets.ndim == 2, (
        f"Targets must be a 2D tensor (B, 1), got {targets.ndim}D of shape {targets.shape}"
    )

    targets_expanded = targets.unsqueeze(1).expand_as(expert_outputs)

    # Calculate the loss for each expert-sample pair and reduce over the output dimension.
    full_loss_matrix = loss_fn(expert_outputs, targets_expanded, reduction="none")
    full_loss_matrix = full_loss_matrix.mean(dim=-1)  # Shape: (B, num_experts)

    # Create a binary mask for selected experts.
    mask = torch.zeros_like(full_loss_matrix)
    mask.scatter_(1, selected_experts, 1.0)
    masked_loss = full_loss_matrix * mask

    # Aggregate the loss over all samples and count the number of times each expert was selected.
    expert_counts = mask.sum(dim=0)
    per_expert_loss_sum = masked_loss.sum(dim=0)

    # Avoid division by zero for experts that weren't selected.
    per_expert_loss = per_expert_loss_sum / expert_counts.clamp(min=1.0)

    return per_expert_loss


def calculate_last_layer_regret(
    expert_outputs: torch.Tensor,
    gating_scores: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    router_activation: str = "softmax",
    combo_size: int = 2,
) -> Tuple[float, float]:
    """
    Bandit-style routing regret for the (last) MoE layer.

    Treats each sample as a context, the router's choice of which ``combo_size``
    experts to combine as the action, and negative MSE as the reward. Regret is
    the router's actual prediction loss minus the loss of the best candidate
    expert combination, averaged over the batch (per-sample regret).

    Two oracles are returned, differing in how a candidate combination is mixed:

    - ``regret_gating``: candidate experts mixed with the router's own gating
      logits, renormalized over the combination (matches ``Model.forward``).
      Isolates the routing *choice* regret with the gating mechanism fixed.
    - ``regret_optimal``: candidate experts mixed with the loss-minimizing convex
      weight (closed form, ``combo_size == 2`` only). Stronger oracle that also
      captures weighting suboptimality. Returns ``nan`` if ``combo_size != 2``.

    By construction ``regret_optimal >= regret_gating >= 0``.

    Args:
        expert_outputs: (B, num_experts, output_dim)
        gating_scores: (B, num_experts) raw router logits
        predictions: (B, output_dim) router's actual combined output
        targets: (B, output_dim)
        router_activation: "softmax" or "sigmoid" (mirrors the model).
        combo_size: Number of experts per candidate combination (top-k).

    Returns:
        (regret_gating, regret_optimal) as Python floats.
    """
    import itertools

    assert expert_outputs.ndim == 3, (
        f"Expert outputs must be a 3D tensor (B, num_experts, output_dim), got {expert_outputs.shape}"
    )

    _, num_experts, _ = expert_outputs.shape
    device = expert_outputs.device

    # Router's actual per-sample loss (the action it took).
    chosen_loss = ((predictions - targets) ** 2).mean(dim=-1)  # (B,)

    # All candidate combinations of experts -> (C, combo_size).
    combos = list(itertools.combinations(range(num_experts), combo_size))
    combo_idx = torch.tensor(combos, dtype=torch.long, device=device)  # (C, combo_size)

    # Gather scores/outputs for every combination.
    sc = gating_scores[:, combo_idx]  # (B, C, combo_size)
    outs = expert_outputs[:, combo_idx, :]  # (B, C, combo_size, output_dim)
    y = targets[:, None, :]  # (B, 1, output_dim)

    # ----- Gating oracle: renormalize the router's logits over the combination.
    if router_activation == "softmax":
        probs = torch.softmax(sc, dim=-1)
    else:  # "sigmoid": sigmoid + sum-normalization (mirrors Model.forward)
        sig = torch.sigmoid(sc)
        probs = sig / sig.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    gating_pred = (outs * probs.unsqueeze(-1)).sum(dim=2)  # (B, C, output_dim)
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


def per_expert_gradient_norm(model: nn.Module) -> list[float]:
    """L2 norm of each expert submodule's parameter gradients, one scalar per expert."""
    norms: list[float] = []
    for expert in model.experts:
        sum_sq = 0.0
        for p in expert.parameters():
            if p.grad is not None:
                sum_sq += float(p.grad.detach().pow(2).sum().item())
        norms.append(sum_sq**0.5)
    return norms


def gating_gradient_norm(model: nn.Module) -> float:
    """L2 norm of the router / gating Linear's parameter gradients."""
    sum_sq = 0.0
    for p in model.gating_function.parameters():
        if p.grad is not None:
            sum_sq += float(p.grad.detach().pow(2).sum().item())
    return sum_sq**0.5
