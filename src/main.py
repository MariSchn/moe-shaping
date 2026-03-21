import torch
import torch.nn as nn

from typing import Tuple

from utils import get_device


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


class Model(nn.Module):
    def __init__(
        self, num_experts: int, input_dim: int, output_dim: int, router_top_k: int
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        assert x.ndim == 2, f"Input must be a 2D tensor (B, D), got {x.ndim}D"
        assert x.shape[1] == self.input_dim, (
            f"Input must have {self.input_dim} features, got {x.shape[1]} features"
        )

        # Routing
        gating_scores = self.gating_function(x)  # (B, num_experts)
        top_k_scores, top_k_indices = torch.topk(
            gating_scores, self.router_top_k, dim=1
        )  # (B, top_k) each

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

        return weighted.sum(dim=1), gating_scores


def main() -> None:
    num_pieces = 2
    domain = (-10, 10)
    num_steps = 100
    batch_size = 100
    learning_rate = 0.01
    device = get_device()
    target_function = PiecewiseLinearTarget(domain, num_pieces, device=device)

    model = Model(num_experts=num_pieces, input_dim=1, output_dim=1, router_top_k=1).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for i in range(num_steps):
        x = torch.randn(batch_size, 1).to(device)
        y = target_function(x)

        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {i}, Loss: {loss.item()}")

    print(f"Final loss: {loss.item()}")


if __name__ == "__main__":
    main()
