from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from router import Router


@dataclass
class MoEOutput:
    x: torch.Tensor  # (B, input_dim) input of the layer
    output: torch.Tensor  # (B, output_dim)
    gating_scores: torch.Tensor  # (B, num_experts)
    expert_outputs: torch.Tensor  # (B, num_experts, output_dim)
    selected_experts: torch.Tensor  # (B, top_k)


class ExpertBank(nn.Module):
    """``num_experts`` MLPs of the same shape, each layer's parameters stacked over
    the experts: ``weights[l]`` is ``(num_experts, out, in)``, ``biases[l]``
    ``(num_experts, out)``. Linear for ``num_layers == 1``."""

    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 1,
    ):
        super().__init__()
        if num_layers > 1 and hidden_dim is None:
            raise ValueError("hidden_dim must be set when num_layers > 1")

        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            # nn.Linear's default initialization.
            bound = d_in**-0.5
            self.weights.append(
                nn.Parameter(
                    torch.empty(num_experts, d_out, d_in).uniform_(-bound, bound)
                )
            )
            self.biases.append(
                nn.Parameter(torch.empty(num_experts, d_out).uniform_(-bound, bound))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Every expert on every input, ``(B, input_dim) -> (B, num_experts, output_dim)``."""
        h = torch.einsum("bi,eoi->beo", x, self.weights[0]) + self.biases[0]
        for weight, bias in zip(self.weights[1:], self.biases[1:]):
            h = torch.einsum("bei,eoi->beo", torch.relu(h), weight) + bias
        return h


class MoELayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        top_k: int,
        router: Callable[[int, int, int], Router],
        expert_hidden_dim: int | None = None,
        expert_num_layers: int = 1,
    ):
        super().__init__()
        self.router = router(input_dim, num_experts, top_k)
        self.experts = ExpertBank(
            num_experts, input_dim, output_dim, expert_hidden_dim, expert_num_layers
        )

    def forward(self, x: torch.Tensor) -> MoEOutput:
        gating_scores, selected_experts, gate_weights = self.router(x)
        expert_outputs = self.experts(x)
        selected_outputs = expert_outputs.gather(
            1, selected_experts.unsqueeze(-1).expand(-1, -1, expert_outputs.shape[-1])
        )
        output = (selected_outputs * gate_weights.unsqueeze(-1)).sum(dim=1)
        return MoEOutput(x, output, gating_scores, expert_outputs, selected_experts)


class Model(nn.Module):
    """A stack of MoE layers, layer ``i`` mapping ``dims[i] -> dims[i + 1]``.

    ``router`` builds each layer's router from ``(input_dim, num_experts, top_k)``.
    ``initial_parameters`` overwrites entries of the state dict, e.g.
    ``layers.0.experts.weights.0``.
    """

    def __init__(
        self,
        dims: list[int],
        num_experts: int,
        top_k: int,
        router: Callable[[int, int, int], Router],
        expert_hidden_dim: int | None = None,
        expert_num_layers: int = 1,
        initial_parameters: dict[str, list] | None = None,
    ):
        super().__init__()
        self.input_dim = dims[0]
        self.output_dim = dims[-1]
        self.num_experts = num_experts
        self.layers = nn.ModuleList(
            MoELayer(
                d_in,
                d_out,
                num_experts,
                top_k,
                router,
                expert_hidden_dim,
                expert_num_layers,
            )
            for d_in, d_out in zip(dims[:-1], dims[1:])
        )

        if initial_parameters is not None:
            state = {k: torch.tensor(v) for k, v in initial_parameters.items()}
            unexpected = self.load_state_dict(state, strict=False).unexpected_keys
            if unexpected:
                raise ValueError(f"Unknown initial parameters: {unexpected}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[MoEOutput]]:
        """Predictions ``(B, output_dim)`` and the output of every layer."""
        outputs = []
        for layer in self.layers:
            outputs.append(layer(x))
            x = outputs[-1].output
        return x, outputs

    def update_routers(self, outputs: list[MoEOutput], losses: torch.Tensor) -> None:
        """Pass every router the per-token losses ``(B,)`` of the final predictions."""
        for layer, output in zip(self.layers, outputs):
            layer.router.update(output.x, output.selected_experts, losses)
