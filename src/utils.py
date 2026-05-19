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


def _lola_naive_indices(
    gating_scores: torch.Tensor,
    selected: torch.Tensor,
    lola_learner: str,
) -> torch.Tensor:
    """
    Naive-role indices (in expert space) implied by the `lola_learner` spec.

    The for-loops in the shaping functions iterate over naive roles per sample
    and apply the correction to the LOLA-learner rows (i.e. the selected
    experts that aren't playing the naive role in that iteration). This helper
    returns those naive indices given which selected expert(s) the user
    designates as LOLA learners.

    Returns (B, K) where K=1 for "top1"/"bottom1" and K=top_k for "all" and
    "fixed:N" (every selected expert plays the naive role in turn while the
    others act as LOLA learners shaping each other; for "fixed:N" the
    iteration where naive == N self-cancels via the recipient mask).
    """
    fixed_n = _parse_fixed_lola_learner(lola_learner)
    if lola_learner == "all" or fixed_n is not None:
        if fixed_n is not None:
            num_experts = gating_scores.shape[1]
            if not 0 <= fixed_n < num_experts:
                raise ValueError(
                    f"fixed lola_learner index {fixed_n} out of range [0, {num_experts})"
                )
        return selected
    top_k_scores = torch.gather(gating_scores, 1, selected)  # (B, top_k)
    # `lola_learner == "top1"` means the top-scoring selected expert is the
    # LOLA learner, so the naive role goes to the bottom-scoring one (and vice
    # versa). For top_k=2 this is symmetric; for top_k>2 the helper still
    # picks a single naive index per sample.
    if lola_learner == "top1":
        idx = top_k_scores.argmin(dim=1)
    elif lola_learner == "bottom1":
        idx = top_k_scores.argmax(dim=1)
    else:
        raise ValueError(
            f"lola_learner must be one of ('top1', 'bottom1', 'all', 'fixed:N'), got {lola_learner!r}"
        )
    return torch.gather(selected, 1, idx.unsqueeze(1))  # (B, 1)


def _parse_fixed_lola_learner(lola_learner: str) -> int | None:
    """Return the integer N if `lola_learner` is `fixed:N`, else None.

    Raises ValueError on a malformed `fixed:` string (e.g. non-integer suffix).
    Range checking against `num_experts` is done by the caller.
    """
    if not isinstance(lola_learner, str) or not lola_learner.startswith("fixed:"):
        return None
    suffix = lola_learner[len("fixed:") :]
    try:
        return int(suffix)
    except ValueError:
        raise ValueError(
            f"fixed lola_learner must be of form 'fixed:N' with N an integer, got {lola_learner!r}"
        )


def _lola_recipient_mask(
    lola_learner: str,
    selected: torch.Tensor,
    naive_per_sample: torch.Tensor,
    num_experts: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """(B, E) mask of experts that should receive the LOLA correction.

    For non-fixed modes this is `1 - naive_mask` — every non-naive expert
    (non-selected experts have zero gradient anyway, so the correction lands
    only on the other selected experts).

    For `fixed:N`, only expert N receives the correction, and only for samples
    where N is in `selected`. The iteration where naive == N self-cancels
    because expert N is then both the recipient and the naive expert.
    """
    B = selected.shape[0]
    device = selected.device
    arange_B = torch.arange(B, device=device)
    naive_mask = torch.zeros(B, num_experts, dtype=dtype, device=device)
    naive_mask[arange_B, naive_per_sample] = 1.0

    n = _parse_fixed_lola_learner(lola_learner)
    if n is None:
        return 1.0 - naive_mask

    fixed_one_hot = torch.zeros(num_experts, dtype=dtype, device=device)
    fixed_one_hot[n] = 1.0
    is_selected = (selected == n).any(dim=1, keepdim=True).to(dtype)  # (B, 1)
    recipient = fixed_one_hot.unsqueeze(0).expand(B, -1) * is_selected  # (B, E)
    return recipient * (1.0 - naive_mask)


def apply_router_router_lola_shaping(
    model: nn.Module,
    output: dict,
    targets: torch.Tensor,
    alpha: float,
    x: torch.Tensor,
    lola_learner: str = "all",
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Router-router LOLA shaping: routers shape routers via gating gradients.

    Some selected experts play the LOLA-learner role (their gating row is
    corrected to anticipate the naive learners' update) and the remaining
    selected experts are naive learners (their gating row gets only the
    regular gradient). Which selected expert(s) act as LOLA learners is
    controlled by `lola_learner` — see below.

    Args:
        model: A `Model` instance with a `.gating_function` Linear.
        output: The dict returned by `Model.forward` for the current batch.
            Must contain `gating_scores`, `expert_outputs`, `selected_experts`.
        targets: (B, output_dim) regression targets for the current batch.
        alpha: LOLA inner-step coefficient. Set to 0 to disable.
        x: (B, input_dim) model inputs for the current batch.
        lola_learner: Which selected expert(s) are LOLA learners per sample.
            "top1" picks the highest-scoring; "bottom1" picks the
            lowest-scoring; "all" makes every selected expert a LOLA
            learner that shapes the others — the correction is summed over
            each selected expert's shaping role, so its magnitude scales
            with top_k. "fixed:N" pins expert N as the sole LOLA learner
            regardless of gating scores; samples where N is not in the
            selected experts contribute zero correction.

    Returns:
        (weight_correction, bias_correction) tensors that were added to the
        gating gradients, or None when the correction is a no-op (alpha == 0
        or top_k <= 1). The caller can use these to separate LOLA from the
        regular gradient after loss.backward().
    """
    if alpha == 0:
        return None

    selected = output["selected_experts"]  # (B, top_k)
    B, top_k = selected.shape
    if top_k <= 1:
        return None

    # expert_outputs are fixed w.r.t. gating params -> detach to avoid leaking
    expert_outputs = output["expert_outputs"].detach()  # (B, E, output_dim)
    gating_scores = output["gating_scores"]  # (B, num_experts)

    naive_indices = _lola_naive_indices(
        gating_scores, selected, lola_learner
    )  # (B, K) — K=1 for top1/bottom1, K=top_k for "all"

    # Extract the gating function
    gating_w = model.gating_function.weight  # (E, input_dim)
    gating_bias = model.gating_function.bias  # (E,)
    router_activation = model.router_activation

    # Pure per-sample loss as a function of gating params only.
    # Routing is treated as fixed based on the input selected experts.
    # The suffix "_b" indicates that a variable stores a value for a single sample.
    def _single_sample_loss(
        gating_weight, gating_bias, input_b, expert_out_b, target_b, selected_b
    ):
        all_gating_scores = input_b @ gating_weight.T + gating_bias  # (E,)
        selected_scores = all_gating_scores[selected_b]  # (top_k,)
        selected_expert_out = expert_out_b[selected_b]  # (top_k, output_dim)
        if router_activation == "softmax":
            gate_probs = torch.softmax(selected_scores, dim=0)
        else:
            sig = torch.sigmoid(selected_scores)
            gate_probs = sig / sig.sum().clamp(min=1e-9)
        prediction = (selected_expert_out * gate_probs.unsqueeze(-1)).sum(dim=0)
        return (prediction - target_b).pow(2).mean()

    # Function to compute the gradient of the loss with respect to the gating function parameters.
    _grad_loss = fgrad(_single_sample_loss, argnums=(0, 1))

    # Function to compute the LOLA correction for a single sample.
    # The suffix "_b" indicates that a variable stores a value for a single sample.
    def _single_sample_correction(
        gating_weight,
        gating_bias,
        input_b,
        expert_out_b,
        target_b,
        selected_b,
        naive_expert_b,
    ):
        # First-order grads; stop-grad the naive learner's row to form the HVP direction v.
        grad_weight_b, grad_bias_b = _grad_loss(
            gating_weight, gating_bias, input_b, expert_out_b, target_b, selected_b
        )

        # unsqueeze to keep as 1-D index — avoids .item() calls inside vmap.
        naive_idx = naive_expert_b.unsqueeze(0)  # (1,)
        v_weight = grad_weight_b[naive_idx].squeeze(0).detach()  # (input_dim,)
        v_bias = grad_bias_b[naive_idx].squeeze(0).detach()  # ()

        # S_b = v . grad_weight_b[naive] + v_bias * grad_bias_b[naive]
        # grad(S_b) gives the Hessian-vector product used as the LOLA correction.
        def _shaping_scalar(gating_weight_, gating_bias_):
            grad_weight_b_, grad_bias_b_ = _grad_loss(
                gating_weight_,
                gating_bias_,
                input_b,
                expert_out_b,
                target_b,
                selected_b,
            )
            return (
                v_weight * grad_weight_b_[naive_idx].squeeze(0)
            ).sum() + v_bias * grad_bias_b_[naive_idx].squeeze(0)

        return fgrad(_shaping_scalar, argnums=(0, 1))(gating_weight, gating_bias)

    # For each naive role (one for top1/bottom1, top_k for "none"), vmap over
    # the batch and accumulate the correction with the naive's own row masked
    # out — i.e. each expert is shaped only by the other selected experts.
    num_experts = gating_w.shape[0]
    weight_correction = torch.zeros_like(gating_w)
    bias_correction = torch.zeros_like(gating_bias)
    for naive_per_sample in naive_indices.unbind(dim=1):  # each (B,)
        all_w, all_b = vmap(
            _single_sample_correction, in_dims=(None, None, 0, 0, 0, 0, 0)
        )(gating_w, gating_bias, x, expert_outputs, targets, selected, naive_per_sample)
        others_mask = _lola_recipient_mask(
            lola_learner, selected, naive_per_sample, num_experts, gating_w.dtype
        )
        weight_correction = (
            weight_correction
            - (alpha / B) * (all_w * others_mask.unsqueeze(-1)).sum(0).detach()
        )
        bias_correction = (
            bias_correction - (alpha / B) * (all_b * others_mask).sum(0).detach()
        )

    if gating_w.grad is None:
        gating_w.grad = weight_correction
    else:
        gating_w.grad = gating_w.grad + weight_correction
    if gating_bias.grad is None:
        gating_bias.grad = bias_correction
    else:
        gating_bias.grad = gating_bias.grad + bias_correction

    # Clone before returning: gating_w.grad and weight_correction are the same
    # tensor object when grad was None, so backward's in-place add_ would
    # otherwise mutate the returned tensors and make the regular norm always 0.
    return weight_correction.clone(), bias_correction.clone()


def apply_expert_expert_lola_shaping(
    model: nn.Module,
    output: dict,
    targets: torch.Tensor,
    alpha: float,
    x: torch.Tensor,
    lola_learner: str = "top1",
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Expert-expert LOLA shaping: experts shape experts via FFN gradients.

    Per sample, one or more selected experts act as LOLA learners whose
    parameter gradients are corrected to anticipate the naive learners' step;
    the remaining selected experts are naive learners (their parameters
    descend along their own loss gradient).

    Args:
        model: A `Model` instance with `model.experts` an `nn.ModuleList` of
            `nn.Linear` and `model.router_activation` set.
        output: The dict returned by `Model.forward` for the current batch.
            Must contain `gating_scores` and `selected_experts`.
        targets: (B, output_dim) regression targets for the current batch.
        alpha: LOLA inner-step coefficient. Set to 0 to disable.
        x: (B, input_dim) model inputs for the current batch.
        lola_learner: Which selected expert(s) are LOLA learners per sample.
            Same semantics as `apply_router_router_lola_shaping`.

    Returns:
        (weight_correction, bias_correction) tensors that were added to the
        per-expert parameter gradients, with shapes
        (num_experts, output_dim, input_dim) and (num_experts, output_dim);
        or None when the correction is a no-op (alpha == 0 or top_k <= 1).
    """
    if alpha == 0:
        return None

    selected = output["selected_experts"]  # (B, top_k)
    B, top_k = selected.shape
    if top_k <= 1:
        return None

    gating_scores = output["gating_scores"]  # (B, num_experts)
    router_activation = model.router_activation

    # Gating probabilities over the selected experts.
    # Detached, since for the expert-shaping loss the gates are treated as fixed.
    selected_scores = torch.gather(gating_scores, 1, selected)  # (B, top_k)
    if router_activation == "softmax":
        gate_probs = torch.softmax(selected_scores, dim=1)
    else:
        sig = torch.sigmoid(selected_scores)
        gate_probs = sig / sig.sum(dim=1, keepdim=True).clamp(min=1e-9)
    gate_probs = gate_probs.detach()  # (B, top_k)

    naive_indices = _lola_naive_indices(
        gating_scores, selected, lola_learner
    )  # (B, K) — K=1 for top1/bottom1, K=top_k for "all"

    # Stack expert params into a single (E, ...) tensor so we can vmap over samples and take grads w.r.t. all experts at once.
    experts_w = torch.stack([e.weight for e in model.experts], dim=0)
    experts_b = torch.stack([e.bias for e in model.experts], dim=0)
    num_experts = experts_w.shape[0]

    # Per-sample loss as a function of expert params only. Routing (which
    # experts and with what gate weight) is fixed via `selected` and
    # `gate_probs_b` — the loss varies only `experts_w` and `experts_b`.
    def _single_sample_loss(
        experts_w, experts_b, input_b, gate_probs_b, target_b, selected_b
    ):
        sel_w = experts_w[selected_b]  # (top_k, output_dim, input_dim)
        sel_b = experts_b[selected_b]  # (top_k, output_dim)
        sel_outs = sel_w @ input_b + sel_b  # (top_k, output_dim)
        prediction = (sel_outs * gate_probs_b.unsqueeze(-1)).sum(dim=0)
        return (prediction - target_b).pow(2).mean()

    _grad_loss = fgrad(_single_sample_loss, argnums=(0, 1))

    def _single_sample_correction(
        experts_w,
        experts_b,
        input_b,
        gate_probs_b,
        target_b,
        selected_b,
        naive_expert_b,
    ):
        grad_w_b, grad_b_b = _grad_loss(
            experts_w, experts_b, input_b, gate_probs_b, target_b, selected_b
        )

        naive_idx = naive_expert_b.unsqueeze(0)  # (1,)
        v_w = grad_w_b[naive_idx].squeeze(0).detach()  # (output_dim, input_dim)
        v_b = grad_b_b[naive_idx].squeeze(0).detach()  # (output_dim,)

        def _shaping_scalar(experts_w_, experts_b_):
            gw, gb = _grad_loss(
                experts_w_,
                experts_b_,
                input_b,
                gate_probs_b,
                target_b,
                selected_b,
            )
            return (v_w * gw[naive_idx].squeeze(0)).sum() + (
                v_b * gb[naive_idx].squeeze(0)
            ).sum()

        return fgrad(_shaping_scalar, argnums=(0, 1))(experts_w, experts_b)

    weight_correction = torch.zeros_like(experts_w)
    bias_correction = torch.zeros_like(experts_b)
    for naive_per_sample in naive_indices.unbind(dim=1):  # each (B,)
        all_w, all_b = vmap(
            _single_sample_correction, in_dims=(None, None, 0, 0, 0, 0, 0)
        )(experts_w, experts_b, x, gate_probs, targets, selected, naive_per_sample)
        # all_w: (B, E, output_dim, input_dim), all_b: (B, E, output_dim)
        others_mask = _lola_recipient_mask(
            lola_learner, selected, naive_per_sample, num_experts, experts_w.dtype
        )
        weight_correction = (
            weight_correction
            - (alpha / B)
            * (all_w * others_mask.view(B, num_experts, 1, 1)).sum(0).detach()
        )
        bias_correction = (
            bias_correction
            - (alpha / B)
            * (all_b * others_mask.view(B, num_experts, 1)).sum(0).detach()
        )

    # Distribute the per-expert correction back to each Linear's .grad.
    for i, expert in enumerate(model.experts):
        if expert.weight.grad is None:
            expert.weight.grad = weight_correction[i].clone()
        else:
            expert.weight.grad = expert.weight.grad + weight_correction[i]
        if expert.bias.grad is None:
            expert.bias.grad = bias_correction[i].clone()
        else:
            expert.bias.grad = expert.bias.grad + bias_correction[i]

    return weight_correction.clone(), bias_correction.clone()


def _cross_recipient_mask(
    pairing: str,
    lola_learner: str,
    selected: torch.Tensor,
    naive_per_sample: torch.Tensor,
    num_experts: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """(B, E) recipient mask for cross-shaping (router-expert / expert-router).

    Unlike inter-router and inter-expert, the diagonal pairing i = j is not
    self-shaping here: w_Gi and W_i are disjoint parameter sets. We therefore
    expose three pairing modes via `pairing`:

    - "off_diagonal" (i != j): same convention as `_lola_recipient_mask` for
      the existing inter-router/expert shapings — every non-naive selected
      expert receives the LOLA correction.
    - "diagonal" (i == j): only the naive expert's own pair receives the
      correction (the qualitatively new case from the derivation; see
      report §A.5, "Note on the i=j case").
    - "all": every selected expert receives the correction (union of the
      above two; recipients are the entire selected set).

    For `fixed:N` lola_learner, the mask is further restricted to expert N,
    matching `_lola_recipient_mask`'s fixed-mode behaviour.
    """
    B = selected.shape[0]
    device = selected.device
    arange_B = torch.arange(B, device=device)
    naive_mask = torch.zeros(B, num_experts, dtype=dtype, device=device)
    naive_mask[arange_B, naive_per_sample] = 1.0

    if pairing == "off_diagonal":
        base = 1.0 - naive_mask
    elif pairing == "diagonal":
        base = naive_mask
    elif pairing == "all":
        base = torch.ones(B, num_experts, dtype=dtype, device=device)
    else:
        raise ValueError(
            f"pairing must be one of ('off_diagonal', 'diagonal', 'all'), got {pairing!r}"
        )

    n = _parse_fixed_lola_learner(lola_learner)
    if n is None:
        return base

    fixed_one_hot = torch.zeros(num_experts, dtype=dtype, device=device)
    fixed_one_hot[n] = 1.0
    is_selected = (selected == n).any(dim=1, keepdim=True).to(dtype)  # (B, 1)
    return base * fixed_one_hot.unsqueeze(0).expand(B, -1) * is_selected


def apply_router_expert_lola_shaping(
    model: nn.Module,
    output: dict,
    targets: torch.Tensor,
    alpha: float,
    x: torch.Tensor,
    lola_learner: str = "all",
    pairing: str = "diagonal",
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Router-Expert LOLA shaping (Direction A): a router row w_Gi is the LOLA
    learner that anticipates an expert's naive step on W_j (and b_j).

    The closed-form correction derived in report §A.5 is

        Delta w_Gi = -4 alpha G_j^2 (||x||^2 + 1) (r^T v_ij) x
        Delta b_Gi = -4 alpha G_j^2 (||x||^2 + 1) (r^T v_ij)

    with v_ij = (delta_{ji} - G_i) r + G_i delta_i. This function realises the
    same correction without that closed form, by computing the HVP

        -alpha * grad_{router} [ v^T grad_{W_j, b_j} L ]

    via torch.func.grad / vmap — same pattern as the existing inter-router and
    inter-expert shaping functions in this module. The closed-form formula is
    verified against this autograd implementation in
    `debug_cross_shaping.py` at the repo root.

    Args:
        model, output, targets, alpha, x, lola_learner: same semantics as
            `apply_router_router_lola_shaping`.
        pairing: how to pair the LOLA router index i with the naive expert
            index j per sample. See `_cross_recipient_mask`. Defaults to
            "diagonal" (i = j) — the qualitatively new cross-shaping case
            from the derivation, where each selected expert's router row is
            shaped to anticipate that same expert's weight update.

    Returns:
        (weight_correction, bias_correction) tensors added to the router's
        gradients, with shapes (num_experts, input_dim) and (num_experts,);
        or None when the correction is a no-op (alpha == 0, or pairing is
        "off_diagonal" with top_k <= 1).
    """
    if alpha == 0:
        return None

    selected = output["selected_experts"]  # (B, top_k)
    B, top_k = selected.shape
    if pairing == "off_diagonal" and top_k <= 1:
        return None  # off-diagonal pairing needs at least two selected experts

    gating_scores = output["gating_scores"]
    router_activation = model.router_activation

    naive_indices = _lola_naive_indices(
        gating_scores, selected, lola_learner
    )  # (B, K) — which expert plays the naive role per sample

    gating_w = model.gating_function.weight  # (E, input_dim)
    gating_bias = model.gating_function.bias  # (E,)
    experts_w = torch.stack(
        [e.weight for e in model.experts], dim=0
    )  # (E, output_dim, input_dim)
    experts_b = torch.stack([e.bias for e in model.experts], dim=0)  # (E, output_dim)
    num_experts = gating_w.shape[0]

    # Per-sample loss as a function of BOTH router and expert params.
    # The cross-Hessian is non-zero precisely because of this joint dependence.
    def _single_sample_loss(
        gating_weight,
        gating_bias_,
        experts_w_,
        experts_b_,
        input_b,
        target_b,
        selected_b,
    ):
        all_gating_scores = input_b @ gating_weight.T + gating_bias_  # (E,)
        selected_scores = all_gating_scores[selected_b]  # (top_k,)
        if router_activation == "softmax":
            gate_probs = torch.softmax(selected_scores, dim=0)
        else:
            sig = torch.sigmoid(selected_scores)
            gate_probs = sig / sig.sum().clamp(min=1e-9)
        sel_w = experts_w_[selected_b]  # (top_k, output_dim, input_dim)
        sel_b = experts_b_[selected_b]  # (top_k, output_dim)
        sel_outs = sel_w @ input_b + sel_b  # (top_k, output_dim)
        prediction = (sel_outs * gate_probs.unsqueeze(-1)).sum(dim=0)
        return (prediction - target_b).pow(2).mean()

    # Naive-side gradient: w.r.t. expert params (argnums 2, 3).
    _grad_naive = fgrad(_single_sample_loss, argnums=(2, 3))

    def _single_sample_correction(
        gating_weight,
        gating_bias_,
        experts_w_,
        experts_b_,
        input_b,
        target_b,
        selected_b,
        naive_expert_b,
    ):
        # Detached naive gradient direction v = (grad_{W_j, b_j} L).
        grad_W_full, grad_b_full = _grad_naive(
            gating_weight,
            gating_bias_,
            experts_w_,
            experts_b_,
            input_b,
            target_b,
            selected_b,
        )
        naive_idx = naive_expert_b.unsqueeze(0)  # (1,)
        v_W = grad_W_full[naive_idx].squeeze(0).detach()  # (output_dim, input_dim)
        v_b = grad_b_full[naive_idx].squeeze(0).detach()  # (output_dim,)

        # Shaping scalar S(gating_w, gating_b) = v . grad_{W_j, b_j} L(gating_w, gating_b).
        # grad(S) gives the cross-HVP used as the LOLA correction.
        def _shaping_scalar(gating_weight_, gating_bias_2):
            gW, gB = _grad_naive(
                gating_weight_,
                gating_bias_2,
                experts_w_,
                experts_b_,
                input_b,
                target_b,
                selected_b,
            )
            return (v_W * gW[naive_idx].squeeze(0)).sum() + (
                v_b * gB[naive_idx].squeeze(0)
            ).sum()

        return fgrad(_shaping_scalar, argnums=(0, 1))(gating_weight, gating_bias_)

    weight_correction = torch.zeros_like(gating_w)
    bias_correction = torch.zeros_like(gating_bias)
    for naive_per_sample in naive_indices.unbind(dim=1):  # each (B,)
        all_w, all_b = vmap(
            _single_sample_correction, in_dims=(None, None, None, None, 0, 0, 0, 0)
        )(
            gating_w,
            gating_bias,
            experts_w,
            experts_b,
            x,
            targets,
            selected,
            naive_per_sample,
        )
        # all_w: (B, E, input_dim), all_b: (B, E)
        mask = _cross_recipient_mask(
            pairing,
            lola_learner,
            selected,
            naive_per_sample,
            num_experts,
            gating_w.dtype,
        )
        weight_correction = (
            weight_correction
            - (alpha / B) * (all_w * mask.unsqueeze(-1)).sum(0).detach()
        )
        bias_correction = bias_correction - (alpha / B) * (all_b * mask).sum(0).detach()

    if gating_w.grad is None:
        gating_w.grad = weight_correction
    else:
        gating_w.grad = gating_w.grad + weight_correction
    if gating_bias.grad is None:
        gating_bias.grad = bias_correction
    else:
        gating_bias.grad = gating_bias.grad + bias_correction

    return weight_correction.clone(), bias_correction.clone()


def apply_expert_router_lola_shaping(
    model: nn.Module,
    output: dict,
    targets: torch.Tensor,
    alpha: float,
    x: torch.Tensor,
    lola_learner: str = "all",
    pairing: str = "diagonal",
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Expert-Router LOLA shaping (Direction B): an expert W_j is the LOLA
    learner that anticipates a router row's naive step on w_Gi (and b_Gi).

    The closed-form correction derived in report §A.5 is

        Delta W_j = -4 alpha G_i G_j (||x||^2 + 1) (r^T delta_i) v_ij x^T
        Delta b_j = -4 alpha G_i G_j (||x||^2 + 1) (r^T delta_i) v_ij

    with v_ij = (delta_{ji} - G_i) r + G_i delta_i, where Schwarz's symmetry
    gives the same cross-Hessian as Direction A — we just contract it over
    the w_Gi axis instead of the W_j axes. Verified against this autograd
    implementation in `debug_cross_shaping.py`.

    Args:
        model, output, targets, alpha, x, lola_learner: same semantics as
            `apply_expert_expert_lola_shaping`.
        pairing: how to pair the LOLA expert index j with the naive router
            index i. See `_cross_recipient_mask`. Defaults to "diagonal"
            (j = i) — each selected expert's weights are shaped to
            anticipate that same expert's router row update.

    Returns:
        (weight_correction, bias_correction) tensors added to the per-expert
        gradients, with shapes (num_experts, output_dim, input_dim) and
        (num_experts, output_dim); or None when the correction is a no-op
        (alpha == 0, or pairing is "off_diagonal" with top_k <= 1).
    """
    if alpha == 0:
        return None

    selected = output["selected_experts"]  # (B, top_k)
    B, top_k = selected.shape
    if pairing == "off_diagonal" and top_k <= 1:
        return None

    gating_scores = output["gating_scores"]
    router_activation = model.router_activation

    naive_indices = _lola_naive_indices(
        gating_scores, selected, lola_learner
    )  # (B, K) — which expert plays the naive role per sample

    gating_w = model.gating_function.weight
    gating_bias = model.gating_function.bias
    experts_w = torch.stack([e.weight for e in model.experts], dim=0)
    experts_b = torch.stack([e.bias for e in model.experts], dim=0)
    num_experts = gating_w.shape[0]

    # Same joint loss as Direction A.
    def _single_sample_loss(
        gating_weight,
        gating_bias_,
        experts_w_,
        experts_b_,
        input_b,
        target_b,
        selected_b,
    ):
        all_gating_scores = input_b @ gating_weight.T + gating_bias_
        selected_scores = all_gating_scores[selected_b]
        if router_activation == "softmax":
            gate_probs = torch.softmax(selected_scores, dim=0)
        else:
            sig = torch.sigmoid(selected_scores)
            gate_probs = sig / sig.sum().clamp(min=1e-9)
        sel_w = experts_w_[selected_b]
        sel_b = experts_b_[selected_b]
        sel_outs = sel_w @ input_b + sel_b
        prediction = (sel_outs * gate_probs.unsqueeze(-1)).sum(dim=0)
        return (prediction - target_b).pow(2).mean()

    # Naive-side gradient: w.r.t. router params (argnums 0, 1).
    _grad_naive = fgrad(_single_sample_loss, argnums=(0, 1))

    def _single_sample_correction(
        gating_weight,
        gating_bias_,
        experts_w_,
        experts_b_,
        input_b,
        target_b,
        selected_b,
        naive_expert_b,
    ):
        # Detached naive gradient v = (grad_{w_Gi, b_Gi} L) at the current params.
        grad_w_full, grad_b_full = _grad_naive(
            gating_weight,
            gating_bias_,
            experts_w_,
            experts_b_,
            input_b,
            target_b,
            selected_b,
        )
        naive_idx = naive_expert_b.unsqueeze(0)
        v_w = grad_w_full[naive_idx].squeeze(0).detach()  # (input_dim,)
        v_b = grad_b_full[naive_idx].squeeze(0).detach()  # ()

        # Shaping scalar S(experts_w, experts_b); its grad gives the cross-HVP.
        def _shaping_scalar(experts_w_2, experts_b_2):
            gw, gB = _grad_naive(
                gating_weight,
                gating_bias_,
                experts_w_2,
                experts_b_2,
                input_b,
                target_b,
                selected_b,
            )
            return (v_w * gw[naive_idx].squeeze(0)).sum() + v_b * gB[naive_idx].squeeze(
                0
            )

        return fgrad(_shaping_scalar, argnums=(0, 1))(experts_w_, experts_b_)

    weight_correction = torch.zeros_like(experts_w)
    bias_correction = torch.zeros_like(experts_b)
    for naive_per_sample in naive_indices.unbind(dim=1):
        all_w, all_b = vmap(
            _single_sample_correction, in_dims=(None, None, None, None, 0, 0, 0, 0)
        )(
            gating_w,
            gating_bias,
            experts_w,
            experts_b,
            x,
            targets,
            selected,
            naive_per_sample,
        )
        # all_w: (B, E, output_dim, input_dim), all_b: (B, E, output_dim)
        mask = _cross_recipient_mask(
            pairing,
            lola_learner,
            selected,
            naive_per_sample,
            num_experts,
            experts_w.dtype,
        )
        weight_correction = (
            weight_correction
            - (alpha / B) * (all_w * mask.view(B, num_experts, 1, 1)).sum(0).detach()
        )
        bias_correction = (
            bias_correction
            - (alpha / B) * (all_b * mask.view(B, num_experts, 1)).sum(0).detach()
        )

    # Distribute the per-expert correction back to each Linear's .grad.
    for i, expert in enumerate(model.experts):
        if expert.weight.grad is None:
            expert.weight.grad = weight_correction[i].clone()
        else:
            expert.weight.grad = expert.weight.grad + weight_correction[i]
        if expert.bias.grad is None:
            expert.bias.grad = bias_correction[i].clone()
        else:
            expert.bias.grad = expert.bias.grad + bias_correction[i]

    return weight_correction.clone(), bias_correction.clone()


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


def per_expert_lola_regular_cosine_similarity(
    lola_weight_grad: torch.Tensor,
    lola_bias_grad: torch.Tensor,
    regular_weight_grad: torch.Tensor,
    regular_bias_grad: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """
    Per-expert cosine similarity between the LOLA gradient and the regular
    gradient, averaged over experts that received a non-zero LOLA update.
    Returns NaN for the average if no expert has a LOLA update in this batch.

    Inputs may be any shape with the leading dim equal to num_experts; trailing
    dims are flattened into a single per-expert vector. For 1-D bias of shape
    (num_experts,), this collapses to sign agreement (±1).
    """

    def _per_expert_cos(lola: torch.Tensor, regular: torch.Tensor) -> float:
        lola_flat = lola.reshape(lola.shape[0], -1)
        reg_flat = regular.reshape(regular.shape[0], -1)
        lola_norm = lola_flat.norm(dim=-1)
        reg_norm = reg_flat.norm(dim=-1)
        cos = (lola_flat * reg_flat).sum(dim=-1) / (lola_norm * reg_norm).clamp(min=eps)
        mask = lola_norm > eps
        return cos[mask].mean().item() if mask.any() else float("nan")

    return (
        _per_expert_cos(lola_weight_grad, regular_weight_grad),
        _per_expert_cos(lola_bias_grad, regular_bias_grad),
    )
