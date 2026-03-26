import os

import numpy as np
import torch
import torch.nn as nn

from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    if target_function is not None:
        inner_breakpoints = target_function.breakpoints.detach().cpu()
        inner_breakpoints = inner_breakpoints[1:-1]
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
        ax.plot(x.cpu(), y.cpu(), label=f"Expert {i} Vector")

    breakpoints = None
    if target_function is not None:
        inner_breakpoints = target_function.breakpoints.detach().cpu()
        inner_breakpoints = inner_breakpoints[1:-1]
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

    fig, ax = plt.subplots()
    for i in range(model.num_experts):
        ax.plot(x.cpu(), expert_outputs[:, i, 0].cpu(), label=f"Expert {i}")
    ax.legend()
    ax.set_xlim(domain)
    ax.set_title("Expert Visualization")

    return {
        "figure": fig,
        "x": x,
        "expert_outputs": expert_outputs,
    }


def export_training_animation_visualization(
    viz_frames: list[dict],
    viz_x: torch.Tensor,
    output_dir: str,
    domain: Tuple[float, float] = (-1, 1),
    target_function: Optional[PiecewiseLinearTarget] = None,
    fps: int = 20,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    n_frames = len(viz_frames)
    x_np = viz_x.squeeze(-1).numpy()
    num_points = len(x_np)
    num_experts = viz_frames[0]["gating_scores"].shape[1]

    # Precompute target curve (fixed across frames)
    y_target = None
    breakpoints = None
    if target_function is not None:
        target_device = target_function.breakpoints.device
        y_target = target_function(viz_x.to(target_device)).squeeze(-1).cpu().numpy()
        inner_bp = target_function.breakpoints.detach().cpu()
        breakpoints = inner_bp[1:-1].tolist()

    # Stack all frames into arrays for easy access and ylim computation
    # predictions: (n_frames, num_points)
    all_model_y = np.array([f["predictions"].squeeze(-1).numpy() for f in viz_frames])
    # expert_outputs: (n_frames, num_points, num_experts)
    all_expert_y = np.array(
        [f["expert_outputs"].squeeze(-1).numpy() for f in viz_frames]
    )
    # gating_scores: (n_frames, num_points, num_experts)
    all_router_y = np.array([f["gating_scores"].numpy() for f in viz_frames])
    # selected_experts: (n_frames, num_points, top_k)
    all_top_experts = [f["selected_experts"].numpy() for f in viz_frames]
    top_k = all_top_experts[0].shape[1]
    # step numbers
    steps = [f["step"] for f in viz_frames]

    def _step_label(frame_idx):
        return f"Step: {steps[frame_idx]}"

    def _ylim(data, extra=None):
        ymin, ymax = float(data.min()), float(data.max())
        if extra is not None:
            ymin = min(ymin, float(extra.min()))
            ymax = max(ymax, float(extra.max()))
        margin = (ymax - ymin) * 0.1
        return (ymin - margin, ymax + margin)

    model_ylim = _ylim(all_model_y, y_target)
    expert_ylim = _ylim(all_expert_y)
    router_ylim = _ylim(all_router_y)

    def _save_animation(fig, update_fn, init_fn, filename):
        ani = FuncAnimation(
            fig,
            update_fn,
            frames=n_frames,
            init_func=init_fn,
            interval=50,
            blit=True,
        )
        ani.save(os.path.join(output_dir, filename), writer="pillow", fps=fps)
        plt.close(fig)

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

    def _add_breakpoints(ax):
        if breakpoints is not None:
            for i, bp in enumerate(breakpoints):
                ax.axvline(
                    bp,
                    color="red",
                    linestyle="--",
                    linewidth=1,
                    label="GT Breakpoints" if i == 0 else None,
                )

    # ===== 1. MODEL VISUALIZATION =====
    fig, ax = plt.subplots()
    (line_model,) = ax.plot(x_np, np.zeros(num_points), label="Model", color="blue")
    if y_target is not None:
        ax.plot(x_np, y_target, label="Target", color="orange", linestyle="--")
    ax.legend()
    ax.set_xlim(domain)
    ax.set_ylim(model_ylim)
    ax.grid(True)
    ax.set_title("Model Visualization")
    ft = _add_frame_text(ax)

    def model_init():
        line_model.set_ydata(np.full(num_points, np.nan))
        ft.set_text("")
        return (line_model, ft)

    def model_update(i):
        line_model.set_ydata(all_model_y[i])
        ft.set_text(_step_label(i))
        return (line_model, ft)

    _save_animation(fig, model_update, model_init, "model.gif")

    # ===== 2. TOP-K EXPERT VISUALIZATION =====
    fig, ax = plt.subplots()
    te_lines = []
    for k in range(top_k):
        (ln,) = ax.plot(x_np, np.zeros(num_points), label=f"Top-{k + 1} Expert")
        te_lines.append(ln)
    _add_breakpoints(ax)
    ax.legend()
    ax.set_xlim(domain)
    ax.set_yticks(range(num_experts))
    ax.set_ylim(-0.1, num_experts - 0.9)
    ax.set_title("Top-k Expert Visualization")
    ft = _add_frame_text(ax)

    def te_init():
        for ln in te_lines:
            ln.set_ydata(np.full(num_points, np.nan))
        ft.set_text("")
        return (*te_lines, ft)

    def te_update(i):
        for k, ln in enumerate(te_lines):
            ln.set_ydata(all_top_experts[i][:, k])
        ft.set_text(_step_label(i))
        return (*te_lines, ft)

    _save_animation(fig, te_update, te_init, "top_expert.gif")

    # ===== 3. ROUTER VISUALIZATION =====
    fig, ax = plt.subplots()
    router_lines = []
    for i in range(num_experts):
        (ln,) = ax.plot(x_np, np.zeros(num_points), label=f"Expert {i} Vector")
        router_lines.append(ln)
    _add_breakpoints(ax)
    ax.legend()
    ax.set_xlim(domain)
    ax.set_ylim(router_ylim)
    ax.set_title("Router Visualization")
    ft = _add_frame_text(ax)

    def router_init():
        for ln in router_lines:
            ln.set_ydata(np.full(num_points, np.nan))
        ft.set_text("")
        return (*router_lines, ft)

    def router_update(i):
        gs = all_router_y[i]
        for j, ln in enumerate(router_lines):
            ln.set_ydata(gs[:, j])
        ft.set_text(_step_label(i))
        return (*router_lines, ft)

    _save_animation(fig, router_update, router_init, "router.gif")

    # ===== 4. EXPERT VISUALIZATION =====
    fig, ax = plt.subplots()
    expert_lines = []
    for i in range(num_experts):
        (ln,) = ax.plot(x_np, np.zeros(num_points), label=f"Expert {i}")
        expert_lines.append(ln)
    ax.legend()
    ax.set_xlim(domain)
    ax.set_ylim(expert_ylim)
    ax.set_title("Expert Visualization")
    ft = _add_frame_text(ax)

    def expert_init():
        for ln in expert_lines:
            ln.set_ydata(np.full(num_points, np.nan))
        ft.set_text("")
        return (*expert_lines, ft)

    def expert_update(i):
        eo = all_expert_y[i]
        for j, ln in enumerate(expert_lines):
            ln.set_ydata(eo[:, j])
        ft.set_text(_step_label(i))
        return (*expert_lines, ft)

    _save_animation(fig, expert_update, expert_init, "expert.gif")

    # ===== 5. PER EXPERT LOSS (BAR CHART) =====
    all_per_expert_loss = np.array(
        [
            f["per_expert_loss"].numpy()
            if isinstance(f["per_expert_loss"], torch.Tensor)
            else np.array(f["per_expert_loss"])
            for f in viz_frames
        ]
    )
    expert_indices = np.arange(num_experts)

    fig, ax = plt.subplots()
    bars_loss = ax.bar(expert_indices, np.zeros(num_experts))
    ax.set_xticks(expert_indices)
    ax.set_xticklabels([f"Expert {i}" for i in expert_indices])
    ax.set_ylim(0, float(np.nanmax(all_per_expert_loss)) * 1.1)
    ax.set_ylabel("Loss")
    ax.set_title("Per Expert Loss")
    ft = _add_frame_text(ax)

    def pel_init():
        for bar in bars_loss:
            bar.set_height(0)
        ft.set_text("")
        return (*bars_loss, ft)

    def pel_update(i):
        for bar, val in zip(bars_loss, all_per_expert_loss[i]):
            bar.set_height(val)
        ft.set_text(_step_label(i))
        return (*bars_loss, ft)

    _save_animation(fig, pel_update, pel_init, "per_expert_loss.gif")

    # ===== 6. PER EXPERT GRAD NORM (BAR CHART) =====
    all_per_expert_grad_norm = np.array(
        [np.array(f["per_expert_grad_norm"]) for f in viz_frames]
    )

    fig, ax = plt.subplots()
    bars_grad = ax.bar(expert_indices, np.zeros(num_experts))
    ax.set_xticks(expert_indices)
    ax.set_xticklabels([f"Expert {i}" for i in expert_indices])
    ax.set_ylim(0, float(all_per_expert_grad_norm.max()) * 1.1)
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Per Expert Gradient Norm")
    ft = _add_frame_text(ax)

    def pegn_init():
        for bar in bars_grad:
            bar.set_height(0)
        ft.set_text("")
        return (*bars_grad, ft)

    def pegn_update(i):
        for bar, val in zip(bars_grad, all_per_expert_grad_norm[i]):
            bar.set_height(val)
        ft.set_text(_step_label(i))
        return (*bars_grad, ft)

    _save_animation(fig, pegn_update, pegn_init, "per_expert_grad_norm.gif")

    # ===== 7. PER EXPERT SAMPLE COUNT (BAR CHART) =====
    all_expert_counts = np.array(
        [
            np.bincount(
                f["train_selected_experts"].numpy().ravel(), minlength=num_experts
            )
            for f in viz_frames
        ]
    )

    fig, ax = plt.subplots()
    bars_count = ax.bar(expert_indices, np.zeros(num_experts))
    ax.set_xticks(expert_indices)
    ax.set_xticklabels([f"Expert {i}" for i in expert_indices])
    ax.set_ylim(0, int(all_expert_counts.max()) + 1)
    ax.set_ylabel("Sample Count")
    ax.set_title("Per Expert Sample Count")
    ft = _add_frame_text(ax)

    def pesc_init():
        for bar in bars_count:
            bar.set_height(0)
        ft.set_text("")
        return (*bars_count, ft)

    def pesc_update(i):
        for bar, val in zip(bars_count, all_expert_counts[i]):
            bar.set_height(val)
        ft.set_text(_step_label(i))
        return (*bars_count, ft)

    _save_animation(fig, pesc_update, pesc_init, "per_expert_sample_count.gif")
