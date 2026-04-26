import random

import numpy as np
import torch
import torch.nn as nn
from torch.func import grad as fgrad, vmap
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
    x: torch.Tensor,
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
            Must contain `gating_scores`, `expert_outputs`, `selected_experts`.
        targets: (B, output_dim) regression targets for the current batch.
        alpha: LOLA inner-step coefficient. Set to 0 to disable.
        x: (B, input_dim) model inputs for the current batch.
    """
    if alpha == 0:
        return

    selected = output["selected_experts"]  # (B, top_k)
    B, top_k = selected.shape
    if top_k <= 1:
        return

    gating_scores = output["gating_scores"]  # (B, num_experts)
    # expert_outputs are fixed w.r.t. gating params — detach to avoid leaking
    # gradients through the expert networks.
    expert_outputs = output["expert_outputs"].detach()  # (B, E, output_dim)

    # Identify the top-1 expert per sample within the selected set.
    top_k_scores = torch.gather(gating_scores, 1, selected)  # (B, top_k)
    top_idx_within_selected = top_k_scores.argmax(dim=1)  # (B,)
    top_per_sample = torch.gather(
        selected, 1, top_idx_within_selected.unsqueeze(1)
    ).squeeze(1)  # (B,)

    gating_w = model.gating_function.weight  # (E, input_dim)
    gating_bias = model.gating_function.bias  # (E,)
    router_activation = model.router_activation

    # Pure per-sample loss as a function of gating params only.  Routing is
    # treated as fixed (sel_b comes from the forward pass and is not
    # differentiated), which matches what torch.autograd.grad does through the
    # original graph (topk indices carry no gradient).
    def _single_sample_loss(gW, gb, x_b, eo_b, t_b, sel_b):
        gs = x_b @ gW.T + gb  # (E,)
        tk_scores = gs[sel_b]  # (top_k,)
        sel_out = eo_b[sel_b]  # (top_k, output_dim)
        if router_activation == "softmax":
            probs = torch.softmax(tk_scores, dim=0)
        else:
            sig = torch.sigmoid(tk_scores)
            probs = sig / sig.sum().clamp(min=1e-9)
        pred = (sel_out * probs.unsqueeze(-1)).sum(dim=0)
        return (pred - t_b).pow(2).mean()

    _grad_loss = fgrad(_single_sample_loss, argnums=(0, 1))

    def _single_sample_correction(gW, gb, x_b, eo_b, t_b, sel_b, top_b):
        # First-order grads; stop-grad the top expert's rows to form v.
        gW_b, gb_b = _grad_loss(gW, gb, x_b, eo_b, t_b, sel_b)
        # Keep top_b as a 1-D index to avoid .item() calls inside vmap.
        top_idx = top_b.unsqueeze(0)  # (1,)
        v_w = gW_b[top_idx].squeeze(0).detach()  # (input_dim,)
        v_bias = gb_b[top_idx].squeeze(0).detach()  # ()

        # S_b = v . gW_b[top] + v_bias * gb_b[top]; grad of S_b w.r.t. params
        # gives the Hessian-vector product used as the LOLA correction.
        def _S(gW_, gb_):
            gW_b_, gb_b_ = _grad_loss(gW_, gb_, x_b, eo_b, t_b, sel_b)
            return (v_w * gW_b_[top_idx].squeeze(0)).sum() + v_bias * gb_b_[
                top_idx
            ].squeeze(0)

        return fgrad(_S, argnums=(0, 1))(gW, gb)  # (cW_b, cb_b)

    # Vectorise over the batch dimension; gW and gb are shared across samples.
    all_cW, all_cb = vmap(
        _single_sample_correction, in_dims=(None, None, 0, 0, 0, 0, 0)
    )(gating_w, gating_bias, x, expert_outputs, targets, selected, top_per_sample)
    # all_cW: (B, E, input_dim),  all_cb: (B, E)

    # The top expert's row is left untouched; zero its contribution.
    E = gating_w.shape[0]
    top_mask = torch.zeros(B, E, dtype=gating_w.dtype, device=gating_w.device)
    top_mask[torch.arange(B), top_per_sample] = 1.0  # (B, E)

    weight_correction = (
        -(alpha / B) * (all_cW * (1.0 - top_mask).unsqueeze(-1)).sum(0).detach()
    )
    bias_correction = -(alpha / B) * (all_cb * (1.0 - top_mask)).sum(0).detach()

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
