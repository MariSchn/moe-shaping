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


def apply_lola_router_shaping(
    model: nn.Module,
    output: dict,
    targets: torch.Tensor,
    alpha: float,
) -> None:
    """
    Add a LOLA-style shaping correction to the router (gating) gradients.

    For each sample b, let `top_b` be the most-chosen selected expert (highest
    gating score among the top-k selected) and `others_b` the remaining selected
    experts. For each j in `others_b`, the correction added to j's gating-weight
    row is

        delta W_j = -alpha * ( d^2 L_b / (dW_j dW_{top_b}) ) * (dL_b / dW_{top_b})

    and analogously for the bias. The top expert's row is left untouched; the
    "others" shape themselves with respect to how they couple to `top_b`. This
    is computed as the gradient (w.r.t. W_j) of the LOLA-DiCE shaping scalar

        S_b = (dL_b / dW_{top_b}) . stop_grad(dL_b / dW_{top_b}),

    which yields the same Hessian-vector product. The correction is averaged
    over the batch to match the scale of `MSELoss` (mean reduction), and is
    *added* to whatever is already in `gating_function.{weight,bias}.grad`, so
    it can be applied either before or after the standard `loss.backward()`
    call (caller is responsible for `retain_graph=True` if needed).

    No-op when `alpha == 0` or `top_k <= 1` (no "others" to shape from).

    Args:
        model: A `Model` instance with a `.gating_function` Linear.
        output: The dict returned by `Model.forward` for the current batch.
            Must contain `predictions`, `gating_scores`, `selected_experts`.
        targets: (B, output_dim) regression targets for the current batch.
        alpha: LOLA inner-step coefficient. Set to 0 to disable.
    """
    if alpha == 0:
        return

    selected = output["selected_experts"]  # (B, top_k)
    B, top_k = selected.shape
    if top_k <= 1:
        return

    gating_scores = output["gating_scores"]  # (B, num_experts)
    predictions = output["predictions"]  # (B, output_dim)

    # Identify the top-1 expert per sample within the selected set.
    top_k_scores = torch.gather(gating_scores, 1, selected)  # (B, top_k)
    top_idx_within_selected = top_k_scores.argmax(dim=1)  # (B,)
    top_per_sample = torch.gather(
        selected, 1, top_idx_within_selected.unsqueeze(1)
    ).squeeze(1)  # (B,)

    # Per-sample MSE loss (mean over output_dim, matching MSELoss(reduction='mean')).
    per_sample_loss = (predictions - targets).pow(2).mean(dim=-1)  # (B,)

    gating_w = model.gating_function.weight  # (num_experts, input_dim)
    gating_bias = model.gating_function.bias  # (num_experts,)

    weight_correction = torch.zeros_like(gating_w)
    bias_correction = torch.zeros_like(gating_bias)

    selected_list = selected.tolist()
    top_list = top_per_sample.tolist()

    # Per-sample loop. The "top" expert varies per sample, so the correction is
    # naturally per-sample; vectorizing across the batch would require torch.func
    # / vmap which is not used here for clarity.
    for b in range(B):
        top_b = top_list[b]
        others_b = [e for e in selected_list[b] if e != top_b]
        if not others_b:
            continue

        # First-order grads of L_b wrt the full gating weight + bias.
        gW, gb = torch.autograd.grad(
            per_sample_loss[b],
            (gating_w, gating_bias),
            create_graph=True,
            retain_graph=True,
        )

        # LOLA shaping scalar built from the *top* expert's first-order grad.
        # dS/dW_j = d^2 L_b / (dW_j dW_top) * stop_grad(dL_b / dW_top), which is
        # the LOLA correction for the non-top selected experts.
        S = (gW[top_b] * gW[top_b].detach()).sum() + (gb[top_b] * gb[top_b].detach())

        # Hessian-vector product: dS/dW gives the correction; we keep only the
        # rows corresponding to the *other* selected experts for this sample.
        cW, cb = torch.autograd.grad(
            S,
            (gating_w, gating_bias),
            retain_graph=True,
        )
        for j in others_b:
            weight_correction[j] = weight_correction[j] - alpha * cW[j]
            bias_correction[j] = bias_correction[j] - alpha * cb[j]

    # Match the per-batch scaling of mean-reduced MSELoss.
    weight_correction = weight_correction / B
    bias_correction = bias_correction / B

    weight_correction = weight_correction.detach()
    bias_correction = bias_correction.detach()

    if gating_w.grad is None:
        gating_w.grad = weight_correction
    else:
        gating_w.grad = gating_w.grad + weight_correction
    if gating_bias.grad is None:
        gating_bias.grad = bias_correction
    else:
        gating_bias.grad = gating_bias.grad + bias_correction


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
