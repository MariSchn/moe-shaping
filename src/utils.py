import torch
from typing import Callable, Tuple


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_uniformly(domain: Tuple[float, float], batch_size: int) -> torch.Tensor:
    return (domain[1] - domain[0]) * torch.rand(batch_size, 1) + domain[0]


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
