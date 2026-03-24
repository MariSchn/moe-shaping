import torch
from typing import Tuple

from utils import sample_uniformly


class PiecewiseLinearTarget:
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
            self.breakpoints = torch.linspace(
                domain[0], domain[1], num_pieces + 1, device=device
            )
        elif breakpoint_mode == "random":
            self.breakpoints = sample_uniformly(domain, num_pieces + 1)
        else:
            raise ValueError(f"Invalid breakpoint mode: {breakpoint_mode}")

        # Override with hard-coded values if provided
        if slopes is not None:
            assert len(slopes) == num_pieces, (
                f"Slopes must have {num_pieces} elements, got {len(slopes)}"
            )
            self.slopes = torch.tensor(slopes, device=device)
        if intercepts is not None:
            assert len(intercepts) == num_pieces, (
                f"Intercepts must have {num_pieces} elements, got {len(intercepts)}"
            )
            self.intercepts = torch.tensor(intercepts, device=device)
        if breakpoints is not None:
            assert len(breakpoints) == num_pieces + 1, (
                f"Breakpoints must have {num_pieces + 1} elements, got {len(breakpoints)}"
            )
            self.breakpoints = torch.tensor(breakpoints, device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        indices = torch.bucketize(x, self.breakpoints) - 1
        indices = indices.clamp(0, self.num_pieces - 1)

        slope = self.slopes[indices]
        intercept = self.intercepts[indices]
        return slope * x + intercept
