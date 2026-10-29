import torch
import torch.nn as nn
from typing import Tuple


class Expert(nn.Module):
    """A single expert mapping ``input_dim -> output_dim``.

    For ``num_layers == 1`` this is exactly a single ``nn.Linear(input_dim,
    output_dim)``, so its behavior — and its ``.weight`` / ``.bias`` parameters —
    are identical to the plain linear experts used previously. For
    ``num_layers > 1`` it is a standard MLP with ``hidden_dim`` hidden units and
    ReLU non-linearities between consecutive linear layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 1,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if num_layers > 1 and hidden_dim is None:
            raise ValueError("hidden_dim must be set when num_layers > 1")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if num_layers == 1:
            layers = [nn.Linear(input_dim, output_dim, bias=True)]
        else:
            layers = [nn.Linear(input_dim, hidden_dim, bias=True), nn.ReLU()]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim, bias=True), nn.ReLU()]
            layers += [nn.Linear(hidden_dim, output_dim, bias=True)]

        self.net = nn.Sequential(*layers)

    @property
    def is_linear(self) -> bool:
        return self.num_layers == 1

    @property
    def linears(self) -> list[nn.Linear]:
        """The expert's linear layers in order (length == num_layers)."""
        return [m for m in self.net if isinstance(m, nn.Linear)]

    @property
    def weight(self) -> torch.Tensor:
        # Exposed for backward compatibility (e.g. LoLA shaping), which assumes a
        # single linear expert. Only well-defined for single-layer experts.
        if not self.is_linear:
            raise AttributeError(
                "Expert.weight is only defined for single-layer (num_layers=1) experts"
            )
        return self.net[0].weight

    @property
    def bias(self) -> torch.Tensor:
        if not self.is_linear:
            raise AttributeError(
                "Expert.bias is only defined for single-layer (num_layers=1) experts"
            )
        return self.net[0].bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Model(nn.Module):
    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        output_dim: int,
        router_top_k: int,
        router_activation: str = "softmax",
        expert_hidden_dim: int | None = None,
        expert_num_layers: int = 1,
        initial_expert_weights: list[list[float]] = None,
        initial_expert_biases: list[list[float]] = None,
        initial_gating_weights: list[list[float]] = None,
        initial_gating_biases: list[float] = None,
    ):
        super().__init__()

        if router_activation not in ("softmax", "sigmoid"):
            raise ValueError(
                f"router_activation must be 'softmax' or 'sigmoid', got {router_activation!r}"
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.router_top_k = router_top_k
        self.router_activation = router_activation

        self.expert_hidden_dim = expert_hidden_dim
        self.expert_num_layers = expert_num_layers

        self.experts = nn.ModuleList(
            [
                Expert(
                    input_dim,
                    output_dim,
                    hidden_dim=expert_hidden_dim,
                    num_layers=expert_num_layers,
                )
                for _ in range(num_experts)
            ]
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

    def _compute_expert_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Run all experts on ``x`` in a single batched pass.

        ``(B, input_dim) -> (B, num_experts, output_dim)``.

        All experts share the same architecture, so instead of looping over the
        expert modules (one set of small matmuls per expert) we stack each
        linear layer's weights across experts and evaluate the whole bank with
        batched matmuls. This is numerically equivalent to running the experts
        individually but launches far fewer kernels. Gradients still flow back
        to each expert's own parameters via the stack.
        """
        h = x  # (B, input_dim)
        for li in range(self.expert_num_layers):
            # (E, out, in) and (E, out) stacked across experts for this layer.
            weight = torch.stack([e.linears[li].weight for e in self.experts], dim=0)
            bias = torch.stack([e.linears[li].bias for e in self.experts], dim=0)
            if li == 0:
                # First layer: x is shared across experts -> (B, E, out).
                h = torch.einsum("bi,eoi->beo", h, weight) + bias
            else:
                h = torch.einsum("bei,eoi->beo", h, weight) + bias
            if li < self.expert_num_layers - 1:
                h = torch.relu(h)
        return h  # (B, num_experts, output_dim)

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

        # Run all experts on the input as a single batched pass.
        # Shape: (B, num_experts, output_dim)
        expert_outputs = self._compute_expert_outputs(x)

        # Get the outputs from the top k experts.
        top_k_expanded = top_k_indices.unsqueeze(-1).expand(
            -1, -1, expert_outputs.size(-1)
        )
        selected_outputs = torch.gather(expert_outputs, dim=1, index=top_k_expanded)

        # Normalize only over the top-k experts, not all experts.
        if self.router_activation == "softmax":
            selected_gate_probs = torch.softmax(top_k_scores, dim=1)  # (B, top_k)
        else:  # "sigmoid": sigmoid + sum-normalization
            sig = torch.sigmoid(top_k_scores)
            selected_gate_probs = sig / sig.sum(dim=1, keepdim=True).clamp(min=1e-9)

        # Combine expert outputs: (B, output_dim)
        weighted = selected_outputs * selected_gate_probs.unsqueeze(-1)

        out = {
            "predictions": weighted.sum(dim=1),  # (B, output_dim)
            "gating_scores": gating_scores,  # (B, num_experts)
            "expert_outputs": expert_outputs,  # (B, num_experts, output_dim)
            "selected_experts": top_k_indices,  # (B, top_k)
        }

        return out
