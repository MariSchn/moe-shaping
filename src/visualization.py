import torch
import torch.nn as nn

from typing import Optional, Tuple
import matplotlib.pyplot as plt

from utils import model_device
from targets import PiecewiseLinearTarget


def model_visualization(
    model: nn.Module,
    domain: Tuple[float, float],
    num_points: int,
    target_function: Optional[PiecewiseLinearTarget] = None,
):
    assert model.output_dim == 1, (
        "Model must have a single output dimension for visualization"
    )
    assert model.input_dim == 1, (
        "Model must have a single input dimension for visualization"
    )

    device = model_device(model)
    x = torch.linspace(domain[0], domain[1], num_points, device=device).unsqueeze(-1)

    with torch.inference_mode():
        y_model = model(x)["predictions"]

    y_target = None
    if target_function is not None:
        y_target = target_function(x)

    fig, ax = plt.subplots()
    ax.plot(x.cpu(), y_model.cpu(), label="Model")

    if y_target is not None:
        ax.plot(x.cpu(), y_target.cpu(), label="Target")

    ax.legend()
    ax.set_xlim(domain)
    ax.grid(True)
    ax.set_title("Model Visualization")

    return {
        "figure": fig,
        "x": x,
        "y_model": y_model,
        "y_target": y_target,
    }


def top_expert_visualization(
    model: nn.Module,
    domain: Tuple[float, float],
    num_points: int,
    target_function: Optional[PiecewiseLinearTarget] = None,
) -> plt.Figure:
    assert model.input_dim == 1, (
        "Model must have a single input dimension for router visualization"
    )

    device = model_device(model)
    x = torch.linspace(domain[0], domain[1], num_points, device=device).unsqueeze(-1)

    with torch.inference_mode():
        output = model(x)
    top_expert = output["selected_experts"][:, 0]

    fig, ax = plt.subplots()
    ax.plot(x.detach().cpu(), top_expert.detach().cpu(), label="Top Expert")

    breakpoints = None
    if target_function is not None:
        inner_breakpoints = target_function.breakpoints.detach().cpu()
        inner_breakpoints = inner_breakpoints[1:-1]
        breakpoints = inner_breakpoints
        for i, bp in enumerate(inner_breakpoints.tolist()):
            ax.axvline(
                bp, color="red", linestyle="--", linewidth=1, label="GT Breakpoints"
            )

    ax.legend()
    ax.set_xlim(domain)
    ax.set_yticks(range(model.num_experts))
    ax.set_ylim(-0.1, model.num_experts - 0.9)
    ax.set_title("Top Expert Visualization")

    return {
        "figure": fig,
        "x": x,
        "top_expert": top_expert,
        "breakpoints": breakpoints,
    }


def router_visualization(
    model: nn.Module,
    domain: Tuple[float, float],
    num_points: int,
    target_function: Optional[PiecewiseLinearTarget] = None,
) -> plt.Figure:
    assert model.input_dim == 1, (
        "Model must have a single input dimension for router visualization"
    )

    device = model_device(model)
    x = torch.linspace(domain[0], domain[1], num_points, device=device).unsqueeze(-1)

    slopes = model.gating_function.weight.detach()
    intercepts = model.gating_function.bias.detach()

    fig, ax = plt.subplots()

    for i in range(model.num_experts):
        y = slopes[i] * x + intercepts[i]
        ax.plot(x.cpu(), y.cpu(), label=f"Expert {i}")

    breakpoints = None
    if target_function is not None:
        inner_breakpoints = target_function.breakpoints.detach().cpu()
        inner_breakpoints = inner_breakpoints[1:-1]
        breakpoints = inner_breakpoints
        for i, bp in enumerate(inner_breakpoints.tolist()):
            ax.axvline(
                bp, color="red", linestyle="--", linewidth=1, label="GT Breakpoints"
            )

    ax.legend()
    ax.set_xlim(domain)
    ax.set_title("Router Visualization")

    return {
        "figure": fig,
        "x": x,
        "slopes": slopes,
        "intercepts": intercepts,
        "breakpoints": breakpoints,
    }
