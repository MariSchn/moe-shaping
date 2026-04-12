import torch
import torch.nn as nn
from typing import Tuple


class Model(nn.Module):
    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        output_dim: int,
        router_top_k: int,
        initial_expert_weights: list[list[float]] = None,
        initial_expert_biases: list[list[float]] = None,
        initial_gating_weights: list[list[float]] = None,
        initial_gating_biases: list[float] = None,
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

        # Auxiliary loss-free load balancing (ignored if cfg.training.load_balancing_loss_weight < 0)
        self.register_buffer("routing_biases", torch.zeros(num_experts))

        # Insert hard-coded weights and biases if provided
        self.set_initial_weights(
            initial_expert_weights,
            initial_expert_biases,
            initial_gating_weights,
            initial_gating_biases,
        )

    def set_initial_weights(
        self,
        initial_expert_weights: list[list[float]],
        initial_expert_biases: list[list[float]],
        initial_gating_weights: list[list[float]],
        initial_gating_biases: list[float],
    ):
        if initial_expert_weights is not None:
            assert len(initial_expert_weights) == self.num_experts, (
                f"Initial weights must have {self.num_experts} elements, got {len(initial_expert_weights)}"
            )
            for i in range(self.num_experts):
                self.experts[i].weight.data = torch.tensor(
                    initial_expert_weights[i], dtype=torch.float32
                )
        if initial_expert_biases is not None:
            assert len(initial_expert_biases) == self.num_experts, (
                f"Initial biases must have {self.num_experts} elements, got {len(initial_expert_biases)}"
            )
            for i in range(self.num_experts):
                self.experts[i].bias.data = torch.tensor(
                    initial_expert_biases[i], dtype=torch.float32
                )
        if initial_gating_weights is not None:
            assert len(initial_gating_weights) == self.num_experts, (
                f"Initial gating weights must have {self.num_experts} elements, got {len(initial_gating_weights)}"
            )
            self.gating_function.weight.data = torch.tensor(
                initial_gating_weights, dtype=torch.float32
            )
        if initial_gating_biases is not None:
            assert len(initial_gating_biases) == self.num_experts, (
                f"Initial gating biases must have {self.num_experts} elements, got {len(initial_gating_biases)}"
            )
            self.gating_function.bias.data = torch.tensor(
                initial_gating_biases, dtype=torch.float32
            )

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
        target_load = target_load or self.router_top_k / self.num_experts

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        assert x.ndim == 2, f"Input must be a 2D tensor (B, D), got {x.ndim}D"
        assert x.shape[1] == self.input_dim, (
            f"Input must have {self.input_dim} features, got {x.shape[1]} features"
        )

        # Routing
        gating_scores = self.gating_function(x)  # (B, num_experts)
        _, top_k_indices = torch.topk(
            gating_scores + self.routing_biases, self.router_top_k, dim=1
        )  # (B, top_k) each
        top_k_scores = torch.gather(gating_scores, dim=1, index=top_k_indices)

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
