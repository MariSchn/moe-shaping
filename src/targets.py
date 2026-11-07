import torch
from abc import ABC, abstractmethod
from typing import Tuple

from models import Model
from utils import sample_uniformly


class TargetFunction(ABC):
    """Inputs are sampled uniformly from ``domain``. ``model`` is the MoE model the
    target is computed by, if any."""

    domain: Tuple[float, float]
    device: torch.device
    model: Model | None = None

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

    def __init__(
        self, model: Model, domain: Tuple[float, float], device: torch.device
    ) -> None:
        self.model = model.to(device).eval()
        self.domain = domain
        self.device = device

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[0]

    @property
    @torch.no_grad()
    def breakpoints(self) -> torch.Tensor | None:
        """Where the set of selected experts changes in any layer, on a fine grid
        over the domain. None for models with input_dim != 1."""
        if self.model.input_dim != 1:
            return None

        x = torch.linspace(*self.domain, 10_001, device=self.device).unsqueeze(-1)
        _, outputs = self.model(x)
        selected = torch.cat(
            [out.selected_experts.sort(dim=1).values for out in outputs], dim=1
        )
        changes = (selected[1:] != selected[:-1]).any(dim=1)
        if not changes.any():
            return None
        return ((x[1:] + x[:-1]) / 2)[changes, 0]
