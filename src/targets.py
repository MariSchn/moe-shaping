import torch
from typing import Tuple


class PiecewiseLinearTarget:
    def __init__(
        self,
        domain: Tuple[float, float],
        num_pieces: int,
        device: torch.device,
        slopes: torch.Tensor = None,
        intercepts: torch.Tensor = None,
        breakpoints: torch.Tensor = None,
    ) -> None:
        self.domain = domain
        self.num_pieces = num_pieces
        self.device = device

        if slopes is None:
            slopes = torch.randn(num_pieces, device=device)
        if intercepts is None:
            intercepts = torch.randn(num_pieces, device=device)
        if breakpoints is None:
            breakpoints = torch.linspace(
                domain[0], domain[1], num_pieces + 1, device=device
            )
        else:
            breakpoints = breakpoints.to(device)
        slopes = slopes.to(device)
        intercepts = intercepts.to(device)

        assert slopes.shape == (num_pieces,), (
            f"Slopes must be a tensor of shape ({num_pieces},)"
        )
        assert intercepts.shape == (num_pieces,), (
            f"Intercepts must be a tensor of shape ({num_pieces},)"
        )
        assert breakpoints.shape == (num_pieces + 1,), (
            f"Breakpoints must be a tensor of shape ({num_pieces + 1},)"
        )

        self.slopes = slopes
        self.intercepts = intercepts
        self.breakpoints = breakpoints

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        indices = torch.bucketize(x, self.breakpoints) - 1
        indices = indices.clamp(0, self.num_pieces - 1)

        slope = self.slopes[indices]
        intercept = self.intercepts[indices]
        return slope * x + intercept
