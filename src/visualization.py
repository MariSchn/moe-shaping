import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn

from typing import Optional, Tuple
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils import model_device
from targets import ModelTarget, TargetFunction


def model_visualization(
    model: nn.Module,
    domain: Tuple[float, float],
    num_points: int,
    target_function: Optional[TargetFunction] = None,
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
    target_function: Optional[TargetFunction] = None,
) -> plt.Figure:
    assert model.input_dim == 1, (
        "Model must have a single input dimension for router visualization"
    )

    device = model_device(model)
    x = torch.linspace(domain[0], domain[1], num_points, device=device).unsqueeze(-1)

    with torch.inference_mode():
        output = model(x)
    selected_experts = output["selected_experts"]
    top_k = selected_experts.shape[1]

    fig, ax = plt.subplots()
    for k in range(top_k):
        ax.plot(
            x.detach().cpu(),
            selected_experts[:, k].detach().cpu(),
            label=f"Top-{k + 1} Expert",
        )

    breakpoints = None
    if target_function is not None and target_function.breakpoints is not None:
        all_bps = target_function.breakpoints.detach().cpu()
        inner_breakpoints = all_bps[(all_bps > domain[0]) & (all_bps < domain[1])]
        breakpoints = inner_breakpoints
        for i, bp in enumerate(inner_breakpoints.tolist()):
            ax.axvline(
                bp,
                color="red",
                linestyle="--",
                linewidth=1,
                label="GT Breakpoints" if i == 0 else None,
            )

    ax.legend()
    ax.set_xlim(domain)
    ax.set_yticks(range(model.num_experts))
    ax.set_ylim(-0.1, model.num_experts - 0.9)
    ax.set_title("Top Expert Visualization")

    return {
        "figure": fig,
        "x": x,
        "selected_experts": selected_experts,
        "breakpoints": breakpoints,
    }


def router_visualization(
    model: nn.Module,
    domain: Tuple[float, float],
    num_points: int,
    target_function: Optional[TargetFunction] = None,
) -> plt.Figure:
    assert model.input_dim == 1, (
        "Model must have a single input dimension for router visualization"
    )

    device = model_device(model)
    x = torch.linspace(domain[0], domain[1], num_points, device=device).unsqueeze(-1)

    slopes = model.gating_function.weight.detach()
    intercepts = model.gating_function.bias.detach()

    routing_biases = model.routing_biases.detach()

    fig, ax = plt.subplots()

    for i in range(model.num_experts):
        y = slopes[i] * x + intercepts[i] + routing_biases[i]
        ax.plot(x.cpu(), y.cpu(), label=f"Expert {i} Vector")

    breakpoints = None
    if target_function is not None and target_function.breakpoints is not None:
        all_bps = target_function.breakpoints.detach().cpu()
        inner_breakpoints = all_bps[(all_bps > domain[0]) & (all_bps < domain[1])]
        breakpoints = inner_breakpoints
        for i, bp in enumerate(inner_breakpoints.tolist()):
            ax.axvline(
                bp,
                color="red",
                linestyle="--",
                linewidth=1,
                label="GT Breakpoints" if i == 0 else None,
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


def expert_visualization(
    model: nn.Module,
    domain: Tuple[float, float],
    num_points: int,
    target_function: Optional[TargetFunction] = None,
) -> plt.Figure:
    assert model.input_dim == 1, (
        "Model must have a single input dimension for expert visualization"
    )
    assert model.output_dim == 1, (
        "Model must have a single output dimension for expert visualization"
    )

    device = model_device(model)
    x = torch.linspace(domain[0], domain[1], num_points, device=device).unsqueeze(-1)

    with torch.inference_mode():
        output = model(x)
    expert_outputs = output["expert_outputs"]

    y_target = None
    if target_function is not None:
        y_target = target_function(x)

    fig, ax = plt.subplots()
    for i in range(model.num_experts):
        ax.plot(x.cpu(), expert_outputs[:, i, 0].cpu(), label=f"Expert {i}")
    if y_target is not None:
        ax.plot(x.cpu(), y_target.cpu(), label="Target", linestyle="--")
    ax.legend()
    ax.set_xlim(domain)
    ax.set_title("Expert Visualization")

    return {
        "figure": fig,
        "x": x,
        "expert_outputs": expert_outputs,
    }


def target_visualization(
    target_function: TargetFunction,
    domain: Tuple[float, float],
    num_points: int,
) -> dict:
    x = torch.linspace(domain[0], domain[1], num_points).unsqueeze(-1)
    y = target_function(x).cpu()

    fig, ax = plt.subplots()
    ax.plot(x, y, label="Target")

    bps = target_function.breakpoints
    if bps is not None:
        inner_bps = bps.detach().cpu()
        inner_bps = inner_bps[(inner_bps > domain[0]) & (inner_bps < domain[1])]
        for i, bp in enumerate(inner_bps.tolist()):
            ax.axvline(
                bp,
                color="red",
                linestyle="--",
                linewidth=1,
                label="Breakpoints" if i == 0 else None,
            )

    ax.legend()
    ax.set_xlim(domain)
    ax.grid(True)
    ax.set_title("Target Function")

    return {"figure": fig, "x": x, "y": y}


def target_router_visualization(
    target_function: TargetFunction,
    domain: Tuple[float, float],
    num_points: int,
) -> dict | None:
    """Visualizes the gating function of a ModelTarget. Returns None for other target types."""
    if not isinstance(target_function, ModelTarget):
        return None

    model = target_function.model
    device = model_device(model)
    x = torch.linspace(domain[0], domain[1], num_points, device=device).unsqueeze(-1)

    slopes = model.gating_function.weight.detach()
    intercepts = model.gating_function.bias.detach()
    routing_biases = model.routing_biases.detach()

    fig, ax = plt.subplots()
    for i in range(model.num_experts):
        y = slopes[i] * x + intercepts[i] + routing_biases[i]
        ax.plot(x.cpu(), y.cpu(), label=f"Expert {i} Vector")

    bps = target_function.breakpoints
    if bps is not None:
        inner_bps = bps.detach().cpu()
        inner_bps = inner_bps[(inner_bps > domain[0]) & (inner_bps < domain[1])]
        for i, bp in enumerate(inner_bps.tolist()):
            ax.axvline(
                bp,
                color="red",
                linestyle="--",
                linewidth=1,
                label="Breakpoints" if i == 0 else None,
            )

    ax.legend()
    ax.set_xlim(domain)
    ax.set_title("Target Router Visualization")

    return {"figure": fig, "x": x}


def routing_bias_visualization(model: nn.Module) -> dict:
    biases = model.routing_biases.detach().cpu().numpy()
    num_experts = len(biases)
    expert_indices = np.arange(num_experts)

    fig, ax = plt.subplots()
    ax.bar(expert_indices, biases)
    ax.set_xticks(expert_indices)
    ax.set_xticklabels([f"Expert {i}" for i in expert_indices])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Bias")
    ax.set_title("Routing Biases")

    return {"figure": fig, "biases": biases}


def _ylim(data, extra=None):
    ymin, ymax = float(data.min()), float(data.max())
    if extra is not None:
        ymin = min(ymin, float(extra.min()))
        ymax = max(ymax, float(extra.max()))
    margin = (ymax - ymin) * 0.1
    return (ymin - margin, ymax + margin)


def _add_frame_text(ax):
    return ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )


def _add_breakpoints(ax, breakpoints):
    if breakpoints is not None:
        for i, bp in enumerate(breakpoints):
            ax.axvline(
                bp,
                color="red",
                linestyle="--",
                linewidth=1,
                label="GT Breakpoints" if i == 0 else None,
            )


def _save_animation(fig, update_fn, init_fn, filepath, n_frames, fps):
    matplotlib.use("Agg")
    ani = FuncAnimation(
        fig,
        update_fn,
        frames=n_frames,
        init_func=init_fn,
        interval=50,
        blit=True,
    )
    ani.save(filepath, writer="ffmpeg", fps=fps)
    plt.close(fig)


# ---- Top-level animation builders (picklable for ProcessPoolExecutor) ----


def _build_model_animation(
    filepath, x_np, all_model_y, y_target, domain, model_ylim, steps, fps
):
    matplotlib.use("Agg")
    num_points = len(x_np)
    fig, ax = plt.subplots()
    (line,) = ax.plot(x_np, np.zeros(num_points), label="Model", color="blue")
    if y_target is not None:
        ax.plot(x_np, y_target, label="Target", color="orange", linestyle="--")
    ax.legend()
    ax.set_xlim(domain)
    ax.set_ylim(model_ylim)
    ax.grid(True)
    ax.set_title("Model Visualization")
    ft = _add_frame_text(ax)

    def init():
        line.set_ydata(np.full(num_points, np.nan))
        ft.set_text("")
        return (line, ft)

    def update(i):
        line.set_ydata(all_model_y[i])
        ft.set_text(f"Step: {steps[i]}")
        return (line, ft)

    _save_animation(fig, update, init, filepath, len(steps), fps)


def _build_top_expert_animation(
    filepath, x_np, all_top_experts, top_k, num_experts, breakpoints, domain, steps, fps
):
    matplotlib.use("Agg")
    num_points = len(x_np)
    fig, ax = plt.subplots()
    lines = []
    for k in range(top_k):
        (ln,) = ax.plot(x_np, np.zeros(num_points), label=f"Top-{k + 1} Expert")
        lines.append(ln)
    _add_breakpoints(ax, breakpoints)
    ax.legend()
    ax.set_xlim(domain)
    ax.set_yticks(range(num_experts))
    ax.set_ylim(-0.1, num_experts - 0.9)
    ax.set_title("Top-k Expert Visualization")
    ft = _add_frame_text(ax)

    def init():
        for ln in lines:
            ln.set_ydata(np.full(num_points, np.nan))
        ft.set_text("")
        return (*lines, ft)

    def update(i):
        for k, ln in enumerate(lines):
            ln.set_ydata(all_top_experts[i][:, k])
        ft.set_text(f"Step: {steps[i]}")
        return (*lines, ft)

    _save_animation(fig, update, init, filepath, len(steps), fps)


def _build_router_animation(
    filepath,
    x_np,
    all_router_y,
    num_experts,
    breakpoints,
    domain,
    router_ylim,
    steps,
    fps,
):
    matplotlib.use("Agg")
    num_points = len(x_np)
    fig, ax = plt.subplots()
    lines = []
    for i in range(num_experts):
        (ln,) = ax.plot(x_np, np.zeros(num_points), label=f"Expert {i} Vector")
        lines.append(ln)
    _add_breakpoints(ax, breakpoints)
    ax.legend()
    ax.set_xlim(domain)
    ax.set_ylim(router_ylim)
    ax.set_title("Router Visualization")
    ft = _add_frame_text(ax)

    def init():
        for ln in lines:
            ln.set_ydata(np.full(num_points, np.nan))
        ft.set_text("")
        return (*lines, ft)

    def update(i):
        gs = all_router_y[i]
        for j, ln in enumerate(lines):
            ln.set_ydata(gs[:, j])
        ft.set_text(f"Step: {steps[i]}")
        return (*lines, ft)

    _save_animation(fig, update, init, filepath, len(steps), fps)


def _build_expert_animation(
    filepath, x_np, all_expert_y, num_experts, domain, expert_ylim, y_target, steps, fps
):
    matplotlib.use("Agg")
    num_points = len(x_np)
    fig, ax = plt.subplots()
    lines = []
    for i in range(num_experts):
        (ln,) = ax.plot(x_np, np.zeros(num_points), label=f"Expert {i}")
        lines.append(ln)
    if y_target is not None:
        ax.plot(x_np, y_target, label="Target", color="red", linestyle="--")
    ax.legend()
    ax.set_xlim(domain)
    ax.set_ylim(expert_ylim)
    ax.set_title("Expert Visualization")
    ft = _add_frame_text(ax)

    def init():
        for ln in lines:
            ln.set_ydata(np.full(num_points, np.nan))
        ft.set_text("")
        return (*lines, ft)

    def update(i):
        eo = all_expert_y[i]
        for j, ln in enumerate(lines):
            ln.set_ydata(eo[:, j])
        ft.set_text(f"Step: {steps[i]}")
        return (*lines, ft)

    _save_animation(fig, update, init, filepath, len(steps), fps)


def _build_bar_animation(
    filepath, all_values, expert_indices, ylim, ylabel, title, steps, fps
):
    matplotlib.use("Agg")
    num_experts = len(expert_indices)
    fig, ax = plt.subplots()
    bars = ax.bar(expert_indices, np.zeros(num_experts))
    ax.set_xticks(expert_indices)
    ax.set_xticklabels([f"Expert {i}" for i in expert_indices])
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ft = _add_frame_text(ax)

    def init():
        for bar in bars:
            bar.set_height(0)
        ft.set_text("")
        return (*bars, ft)

    def update(i):
        for bar, val in zip(bars, all_values[i]):
            bar.set_height(val)
        ft.set_text(f"Step: {steps[i]}")
        return (*bars, ft)

    _save_animation(fig, update, init, filepath, len(steps), fps)


def export_training_animation_visualization(
    viz_frames: list[dict],
    viz_x: torch.Tensor,
    output_dir: str,
    domain: Tuple[float, float] = (-1, 1),
    target_function: Optional[TargetFunction] = None,
    fps: int = 24,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    x_np = viz_x.squeeze(-1).numpy()
    num_experts = viz_frames[0]["gating_scores"].shape[1]

    # Precompute target curve (fixed across frames)
    y_target = None
    breakpoints = None
    if target_function is not None:
        bps = target_function.breakpoints
        target_device = bps.device if bps is not None else viz_x.device
        y_target = target_function(viz_x.to(target_device)).squeeze(-1).cpu().numpy()
        if bps is not None:
            all_bps = bps.detach().cpu()
            breakpoints = all_bps[
                (all_bps > domain[0]) & (all_bps < domain[1])
            ].tolist()

    # Stack all frames into numpy arrays
    all_model_y = np.array([f["predictions"].squeeze(-1).numpy() for f in viz_frames])
    all_expert_y = np.array(
        [f["expert_outputs"].squeeze(-1).numpy() for f in viz_frames]
    )
    all_router_y = np.array(
        [f["gating_scores"].numpy() + f["routing_biases"].numpy() for f in viz_frames]
    )
    all_top_experts = [f["selected_experts"].numpy() for f in viz_frames]
    top_k = all_top_experts[0].shape[1]
    steps = [f["step"] for f in viz_frames]

    all_routing_biases = np.array([f["routing_biases"].numpy() for f in viz_frames])

    all_per_expert_loss = np.array(
        [
            f["per_expert_loss"].numpy()
            if isinstance(f["per_expert_loss"], torch.Tensor)
            else np.array(f["per_expert_loss"])
            for f in viz_frames
        ]
    )
    all_per_expert_grad_norm = np.array(
        [np.array(f["per_expert_grad_norm"]) for f in viz_frames]
    )
    all_expert_counts = np.array(
        [
            np.bincount(
                f["train_selected_experts"].numpy().ravel(), minlength=num_experts
            )
            for f in viz_frames
        ]
    )

    expert_indices = np.arange(num_experts)
    model_ylim = _ylim(all_model_y, y_target)
    expert_ylim = _ylim(all_expert_y, y_target)
    router_ylim = _ylim(all_router_y)

    def path(name):
        return os.path.join(output_dir, name)

    with ProcessPoolExecutor() as pool:
        futures = [
            pool.submit(
                _build_model_animation,
                path("model.mp4"),
                x_np,
                all_model_y,
                y_target,
                domain,
                model_ylim,
                steps,
                fps,
            ),
            pool.submit(
                _build_top_expert_animation,
                path("top_expert.mp4"),
                x_np,
                all_top_experts,
                top_k,
                num_experts,
                breakpoints,
                domain,
                steps,
                fps,
            ),
            pool.submit(
                _build_router_animation,
                path("router.mp4"),
                x_np,
                all_router_y,
                num_experts,
                breakpoints,
                domain,
                router_ylim,
                steps,
                fps,
            ),
            pool.submit(
                _build_expert_animation,
                path("expert.mp4"),
                x_np,
                all_expert_y,
                num_experts,
                domain,
                expert_ylim,
                y_target,
                steps,
                fps,
            ),
            pool.submit(
                _build_bar_animation,
                path("per_expert_loss.mp4"),
                all_per_expert_loss,
                expert_indices,
                (0, float(np.nanmax(all_per_expert_loss)) * 1.1),
                "Loss",
                "Per Expert Loss",
                steps,
                fps,
            ),
            pool.submit(
                _build_bar_animation,
                path("per_expert_grad_norm.mp4"),
                all_per_expert_grad_norm,
                expert_indices,
                (0, float(all_per_expert_grad_norm.max()) * 1.1),
                "Gradient Norm",
                "Per Expert Gradient Norm",
                steps,
                fps,
            ),
            pool.submit(
                _build_bar_animation,
                path("per_expert_sample_count.mp4"),
                all_expert_counts,
                expert_indices,
                (0, int(all_expert_counts.max()) + 1),
                "Sample Count",
                "Per Expert Sample Count",
                steps,
                fps,
            ),
            pool.submit(
                _build_bar_animation,
                path("routing_biases.mp4"),
                all_routing_biases,
                expert_indices,
                _ylim(all_routing_biases),
                "Bias",
                "Routing Biases",
                steps,
                fps,
            ),
        ]
        for f in futures:
            f.result()
