from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Router(nn.Module, ABC):
    """Picks ``top_k`` of ``num_experts`` experts per input and weights their outputs."""

    def __init__(self, input_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k

    @abstractmethod
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route a batch ``x`` of shape ``(B, input_dim)``.

        Returns:
            gating_scores: (B, num_experts) score of every expert.
            selected_experts: (B, top_k) indices of the selected experts.
            gate_probs: (B, top_k) weights of the selected experts' outputs.
        """


class LinearRouter(Router):
    """Linear gating scores ``x @ W.T + b``, one row of ``W`` per expert.

    Selects the top-k experts by score, shifted by the auxiliary loss-free
    routing biases, and normalizes their scores into gate weights.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int,
        activation: str = "softmax",
        initial_weights: list[list[float]] = None,
        initial_biases: list[float] = None,
    ):
        super().__init__(input_dim, num_experts, top_k)

        if activation not in ("softmax", "sigmoid"):
            raise ValueError(
                f"activation must be 'softmax' or 'sigmoid', got {activation!r}"
            )
        self.activation = activation

        self.linear = nn.Linear(input_dim, num_experts, bias=True)

        # Auxiliary loss-free load balancing (ignored if cfg.training.load_balancing_loss_weight < 0)
        self.register_buffer("routing_biases", torch.zeros(num_experts))

        # Insert hard-coded weights and biases if provided
        if initial_weights is not None:
            assert len(initial_weights) == num_experts, (
                f"Initial gating weights must have {num_experts} elements, got {len(initial_weights)}"
            )
            self.linear.weight.data = torch.tensor(initial_weights, dtype=torch.float32)
        if initial_biases is not None:
            assert len(initial_biases) == num_experts, (
                f"Initial gating biases must have {num_experts} elements, got {len(initial_biases)}"
            )
            self.linear.bias.data = torch.tensor(initial_biases, dtype=torch.float32)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gating_scores = self.linear(x)
        _, selected_experts = torch.topk(
            gating_scores + self.routing_biases, self.top_k, dim=1
        )
        selected_scores = torch.gather(gating_scores, dim=1, index=selected_experts)

        # Normalize only over the top-k experts, not all experts.
        if self.activation == "softmax":
            gate_probs = torch.softmax(selected_scores, dim=1)
        else:  # "sigmoid": sigmoid + sum-normalization
            sig = torch.sigmoid(selected_scores)
            gate_probs = sig / sig.sum(dim=1, keepdim=True).clamp(min=1e-9)

        return gating_scores, selected_experts, gate_probs

    def update_routing_biases(
        self,
        selected_experts: torch.Tensor,
        gamma: float = 1e-3,
        target_load: float | None = None,
    ):
        if gamma <= 0:
            return

        B, _ = selected_experts.shape

        # Use uniform load as default target load
        target_load = target_load or self.top_k / self.num_experts

        # Calculate the mean load of each expert
        expert_counts = torch.zeros(self.num_experts, device=selected_experts.device)
        expert_counts.scatter_add_(
            0,
            selected_experts.flatten(),
            torch.ones(selected_experts.numel(), device=selected_experts.device),
        )
        expert_loads = expert_counts / B

        # Calculate the difference between the expert load and the target load
        load_differences = expert_loads - target_load

        bias_updates = torch.zeros(self.num_experts, device=selected_experts.device)
        bias_updates[load_differences < 0] = gamma
        bias_updates[load_differences > 0] = -gamma

        # Update the routing biases
        self.routing_biases += bias_updates
