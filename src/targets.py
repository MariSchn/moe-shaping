import torch
from abc import ABC, abstractmethod
from typing import Tuple

from utils import sample_uniformly


class TargetFunction(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

    @property
    def breakpoints(self) -> torch.Tensor | None:
        return None


class PiecewiseLinearTarget(TargetFunction):
    def __init__(
        self,
        domain: Tuple[float, float],
        num_pieces: int,
        device: torch.device,
        breakpoint_mode: str = "linear",
        slopes: list[list[float]] = None,
        intercepts: list[list[float]] = None,
        breakpoints: torch.Tensor = None,
    ) -> None:
        self.domain = domain
        self.num_pieces = num_pieces
        self.device = device

        self.slopes = torch.randn(num_pieces, device=device)
        self.intercepts = torch.randn(num_pieces, device=device)
        if breakpoint_mode == "linear":
            self._breakpoints = torch.linspace(
                domain[0], domain[1], num_pieces + 1, device=device
            )
        elif breakpoint_mode == "random":
            self._breakpoints = sample_uniformly(domain, num_pieces + 1)
        else:
            raise ValueError(f"Invalid breakpoint mode: {breakpoint_mode}")

        # Override with hard-coded values if provided
        self.set_initial_weights(slopes, intercepts, breakpoints)

    @property
    def breakpoints(self) -> torch.Tensor:
        return self._breakpoints

    def set_initial_weights(
        self,
        slopes: list[list[float]],
        intercepts: list[list[float]],
        breakpoints: torch.Tensor,
    ):
        if slopes is not None:
            assert len(slopes) == self.num_pieces, (
                f"Slopes must have {self.num_pieces} elements, got {len(slopes)}"
            )
            self.slopes = torch.tensor(slopes, device=self.device)
        if intercepts is not None:
            assert len(intercepts) == self.num_pieces, (
                f"Intercepts must have {self.num_pieces} elements, got {len(intercepts)}"
            )
            self.intercepts = torch.tensor(intercepts, device=self.device)
        if breakpoints is not None:
            assert len(breakpoints) == self.num_pieces + 1, (
                f"Breakpoints must have {self.num_pieces + 1} elements, got {len(breakpoints)}"
            )
            self._breakpoints = torch.tensor(breakpoints, device=self.device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        indices = torch.bucketize(x, self._breakpoints) - 1
        indices = indices.clamp(0, self.num_pieces - 1)

        slope = self.slopes[indices]
        intercept = self.intercepts[indices]
        return slope * x + intercept


class ModelTarget(TargetFunction):
    """Wraps a Model instance as a target function.

    Useful for constructing scenarios with a known closed-form solution
    to verify that training converges correctly.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.model.eval()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        model_device = next(self.model.parameters()).device
        x_2d = x.reshape(-1, self.model.input_dim).to(model_device)
        with torch.no_grad():
            output = self.model(x_2d)
        return output["predictions"].to(x.device)

    @property
    def breakpoints(self) -> torch.Tensor | None:
        """Analytically computes where the top-1 expert in the router changes.

        Finds all pairwise intersections of the linear gating functions, then
        filters to only those where the argmax expert actually switches.
        Returns None for models with input_dim != 1.
        """
        if self.model.input_dim != 1:
            return None

        w = self.model.gating_function.weight.detach().squeeze(-1)  # (num_experts,)
        b = (
            self.model.gating_function.bias.detach()
            + self.model.routing_biases.detach()
        )  # (num_experts,)

        # Collect all pairwise intersections where two gating lines cross
        intersections = []
        for i in range(self.model.num_experts):
            for j in range(i + 1, self.model.num_experts):
                dw = w[i] - w[j]
                if abs(dw.item()) > 1e-8:
                    intersections.append(((b[j] - b[i]) / dw).item())

        if not intersections:
            return None

        intersections = sorted(intersections)

        # Filter to intersections where the argmax top expert actually changes.
        # Evaluate at midpoints between consecutive intersections (plus sentinels).
        eval_points = (
            [intersections[0] - 1.0]
            + [
                (intersections[k] + intersections[k + 1]) / 2
                for k in range(len(intersections) - 1)
            ]
            + [intersections[-1] + 1.0]
        )
        x_eval = torch.tensor(eval_points, device=w.device).unsqueeze(-1)  # (N, 1)
        scores = x_eval * w.unsqueeze(0) + b.unsqueeze(0)  # (N, num_experts)
        top_experts = scores.argmax(dim=-1)  # (N,)

        true_breakpoints = [
            bp
            for k, bp in enumerate(intersections)
            if top_experts[k] != top_experts[k + 1]
        ]

        if not true_breakpoints:
            return None

        return torch.tensor(
            true_breakpoints, device=self.model.gating_function.weight.device
        )
