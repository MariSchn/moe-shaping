import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
)
from matplotlib.patches import Patch

from model.moe import Model, MoEOutput
from model.router import INDEX_TERMS
from monitoring.metrics import per_expert_loss
from targets import TargetFunction
from utils import sample_uniformly

GRID_POINTS = 200


def input_grid(target: TargetFunction) -> torch.Tensor:
    """``(GRID_POINTS, 1)`` inputs evenly spaced over the target's domain."""
    return torch.linspace(*target.domain, GRID_POINTS, device=target.device)[:, None]


def _numpy(tensors: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in tensors.items()}


@torch.no_grad()
def record_frame(
    step: int,
    model: Model,
    outputs: list[MoEOutput],
    targets: torch.Tensor | None = None,
    x_grid: torch.Tensor | None = None,
) -> dict:
    """The model's state to plot, from its ``outputs`` on a batch with ``targets``,
    and its outputs on the 1D inputs ``x_grid`` if given."""
    frame = {"step": step, "layers": []}
    if targets is not None:
        last = outputs[-1]
        frame["per_expert_loss"] = (
            per_expert_loss(last.expert_outputs, last.selected_experts, targets)
            .cpu()
            .numpy()
        )

    if x_grid is not None:
        predictions, grid_outputs = model(x_grid)
        frame["x"] = x_grid[:, 0].cpu().numpy()
        frame["predictions"] = predictions.cpu().numpy()

    for i, (layer, output) in enumerate(zip(model.layers, outputs)):
        layer_frame = {
            "sample_count": np.bincount(
                output.selected_experts.cpu().flatten(), minlength=model.num_experts
            )
        }
        if x_grid is None:
            layer_frame.update(_numpy(layer.router.snapshot(output.x)))
        else:
            grid_output = grid_outputs[i]
            layer_frame.update(
                _numpy(
                    {
                        "gating_scores": grid_output.gating_scores,
                        "selected_experts": grid_output.selected_experts,
                        "expert_outputs": grid_output.expert_outputs,
                        **layer.router.snapshot(grid_output.x),
                    }
                )
            )
        frame["layers"].append(layer_frame)
    return frame


@torch.no_grad()
def target_frame(target: TargetFunction, x_grid: torch.Tensor | None) -> dict:
    """`record_frame` of the target's model, or just its values on ``x_grid``."""
    if target.model is not None:
        x = sample_uniformly(target.domain, 1024, target.model.input_dim)
        _, outputs = target.model(x.to(target.device))
        return record_frame(0, target.model, outputs, x_grid=x_grid)
    frame = {"step": 0, "layers": []}
    if x_grid is not None:
        frame["x"] = x_grid[:, 0].cpu().numpy()
        frame["predictions"] = target(x_grid).cpu().numpy()
    return frame


# ---- Plots: each builds a figure from data stacked over frames ``(T, ...)`` and
# returns it with an ``update(i)`` that shows frame i. Top-level, so they can be
# run in a ProcessPoolExecutor.


def _ylim(*arrays) -> tuple[float, float]:
    values = np.concatenate([np.ravel(a) for a in arrays if a is not None] or [[]])
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (-1.0, 1.0)
    ymin, ymax = float(values.min()), float(values.max())
    margin = (ymax - ymin) * 0.1 or 1.0
    return (ymin - margin, ymax + margin)


def _add_breakpoints(ax, breakpoints, vertical=True):
    line = ax.axvline if vertical else ax.axhline
    for i, bp in enumerate(breakpoints):
        line(
            bp,
            color="red" if vertical else "black",
            linestyle="--",
            linewidth=1,
            label="GT Breakpoints" if i == 0 else None,
        )


def _expert_colormap(num_experts):
    return (
        ListedColormap([f"C{i}" for i in range(num_experts)]),
        BoundaryNorm(np.arange(-0.5, num_experts), num_experts),
    )


def _add_selection_strip(strip, x, selected_experts, num_experts):
    """Show the top-k experts ``(len(x), top_k)`` at every x on the axes ``strip``,
    in the experts' line colors. Returns the image."""
    top_k = selected_experts.shape[-1]
    strip.set_yticks(
        np.arange(top_k) + 0.5, [f"Top-{k + 1}" for k in range(top_k)], fontsize=8
    )
    cmap, norm = _expert_colormap(num_experts)
    return strip.imshow(
        selected_experts.T,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        extent=(x[0], x[-1], 0, top_k),
        origin="lower",
        interpolation="nearest",
    )


def _lines(x, ys, labels, title, ylim, ref=None, breakpoints=(), selected_experts=None):
    """One line over ``x`` per label, ``ys`` is ``(T, len(labels), len(x))``.

    With ``selected_experts`` ``(T, len(x), top_k)``, a strip below shows the top-k
    experts, the labels being the experts.
    """
    if selected_experts is None:
        fig, ax = plt.subplots()
    else:
        fig, (ax, strip_ax) = plt.subplots(
            2,
            1,
            sharex=True,
            height_ratios=(12, 1),
            gridspec_kw={"hspace": 0.05},
            figsize=(6.4, 5.2),
        )
    lines = [ax.plot(x, ys[0, i], label=label)[0] for i, label in enumerate(labels)]
    if ref is not None:
        ax.plot(x, ref, color="black", linestyle="--", label="Target")
    _add_breakpoints(ax, breakpoints)
    ax.legend()
    ax.set_xlim(x[0], x[-1])
    if selected_experts is not None:
        strip = _add_selection_strip(strip_ax, x, selected_experts[0], len(labels))
    ax.set_ylim(ylim)
    ax.grid(True)
    ax.set_title(title)

    def update(i):
        for line, y in zip(lines, ys[i]):
            line.set_ydata(y)
        if selected_experts is None:
            return lines
        strip.set_data(selected_experts[i].T)
        return (*lines, strip)

    return fig, update


def _bars(values, title, ylabel, ylim):
    """One bar per expert, ``values`` is ``(T, num_experts)``."""
    num_experts = values.shape[1]
    fig, ax = plt.subplots()
    bars = ax.bar(np.arange(num_experts), values[0])
    ax.set_xticks(np.arange(num_experts), [f"Expert {i}" for i in range(num_experts)])
    if ylim[0] < 0:
        ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    def update(i):
        for bar, value in zip(bars, values[i]):
            bar.set_height(value)
        return bars

    return fig, update


def _index(x, index_terms, selected_experts, title):
    """The bandit index terms ``(T, len(INDEX_TERMS), len(x), num_experts)``, their
    sum, and the selected experts ``(T, len(x), top_k)``."""
    num_experts = index_terms.shape[-1]
    domain = (x[0], x[-1])
    # The bonus is infinite until an expert has enough pulls.
    index_terms = np.where(np.isinf(index_terms), np.nan, index_terms)
    panels = np.concatenate(
        [index_terms, index_terms.sum(axis=1, keepdims=True)], axis=1
    )
    titles = [name.capitalize() for name in INDEX_TERMS] + ["Total Index"]
    # Quantiles, because the early bonus would stretch the axis.
    finite = panels[np.isfinite(panels)]
    ylim = _ylim(np.quantile(finite, [0.01, 0.99]) if len(finite) else None)

    fig, axes = plt.subplots(1, len(titles), sharey=True, figsize=(20, 4.8))
    lines = []
    for ax, panel_title in zip(axes, titles):
        lines.append(
            [
                ax.plot(x, panels[0, 0, :, i], label=f"Expert {i}")[0]
                for i in range(num_experts)
            ]
        )
        ax.set_xlim(domain)
        ax.set_ylim(ylim)
        ax.grid(True)
        ax.set_title(panel_title)
    axes[0].legend(loc="upper right")
    fig.suptitle(title)

    max_reward = axes[-1].axhline(
        1, color="red", linestyle="--", linewidth=1, label="Max Reward"
    )
    axes[-1].legend(handles=[max_reward], loc="upper left")
    strip = axes[-1].inset_axes([0, 0, 1, 0.1])
    strip.set_xticks([])
    selected = _add_selection_strip(strip, x, selected_experts[0], num_experts)
    fig.tight_layout()

    def update(i):
        for panel_lines, values in zip(lines, panels[i]):
            for j, line in enumerate(panel_lines):
                line.set_ydata(values[:, j])
        selected.set_data(selected_experts[i].T)
        return (*(line for panel in lines for line in panel), selected)

    return fig, update


def _save(path, build, kwargs, steps, fps):
    """A PNG of the single frame, or with ``fps`` an MP4 over all frames."""
    fig, update = build(**kwargs)
    text = fig.axes[0].text(
        0.02,
        0.98,
        "",
        transform=fig.axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    def show(i):
        text.set_text(f"Step: {steps[i]}")
        return (*update(i), text)

    if fps is None:
        show(0)
        fig.savefig(f"{path}.png", dpi=150, bbox_inches="tight")
    else:
        animation = FuncAnimation(fig, show, frames=len(steps), interval=50, blit=True)
        animation.save(f"{path}.mp4", writer="ffmpeg", fps=fps)
    plt.close(fig)


def _routing_map(path, x, selected_experts, num_experts, breakpoints, steps):
    """Selected experts ``(T, len(x), top_k)`` over steps and inputs."""
    top_k = selected_experts.shape[-1]
    cmap, norm = _expert_colormap(num_experts)
    fig, axes = plt.subplots(
        top_k, 1, sharex=True, squeeze=False, figsize=(9, 3 * top_k)
    )
    axes = axes[:, 0]
    for k, ax in enumerate(axes):
        ax.pcolormesh(
            steps,
            x,
            selected_experts[:, :, k].T,
            cmap=cmap,
            norm=norm,
            shading="nearest",
        )
        _add_breakpoints(ax, breakpoints, vertical=False)
        ax.set_title(f"Top-{k + 1} Expert")
        ax.set_ylabel("x")
    axes[-1].set_xlabel("Step")
    handles = [
        Patch(color=f"C{i}", label=f"Expert {i}") for i in range(num_experts)
    ] + axes[0].get_legend_handles_labels()[0]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("Routing Map")
    fig.tight_layout()
    fig.savefig(f"{path}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _expert_map(path, values, title, label, steps, baseline):
    """``values`` ``(T, num_experts)`` over steps and experts. White is the
    ``baseline``, blue above it and red below."""
    num_experts = values.shape[1]
    lo = min(float(np.nanmin(values)), baseline)
    hi = max(float(np.nanmax(values)), baseline)
    hi = hi if hi > lo else lo + 1.0
    # The white point sits at the baseline, each side spanning to its extreme.
    white = (baseline - lo) / (hi - lo)
    stops = [(white, "white")]
    if white > 0:
        stops.insert(0, (0.0, "#b2182b"))
    if white < 1:
        stops.append((1.0, "#2166ac"))
    fig, ax = plt.subplots(figsize=(9, 3))
    mesh = ax.pcolormesh(
        steps,
        np.arange(num_experts),
        np.ma.masked_invalid(values.T),
        cmap=LinearSegmentedColormap.from_list("baseline", stops),
        norm=Normalize(lo, hi),
        shading="nearest",
    )
    ax.set_yticks(np.arange(num_experts), [f"Expert {i}" for i in range(num_experts)])
    ax.set_xlabel("Step")
    fig.colorbar(mesh, ax=ax, label=label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f"{path}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _expert_lines(path, values, steps, title, ylabel):
    """One line per expert over steps, ``values`` is ``(T, num_experts)``."""
    fig, ax = plt.subplots(figsize=(9, 3))
    for e in range(values.shape[1]):
        ax.plot(steps, values[:, e], label=f"Expert {e}")
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(True)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f"{path}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render(
    frames: list[dict],
    output_dir: str,
    ref: np.ndarray | None = None,
    breakpoints: list[float] = (),
    animate: bool = False,
) -> None:
    """Plots of a single frame as PNGs, or with ``animate`` MP4s over the frames,
    into ``output_dir`` and a subdirectory per layer.

    ``ref`` is the target on the grid, drawn over the predictions and the last
    layer's experts, and ``breakpoints`` the target's.
    """
    steps = [frame["step"] for frame in frames]
    fps = max(1, len(frames) // 10) if animate else None
    jobs = []

    def plot(name, build, **kwargs):
        jobs.append(
            (_save, dict(path=name, build=build, kwargs=kwargs, steps=steps, fps=fps))
        )

    def bars(name, values, title, ylabel, ylim, baseline=0.0):
        """Bars of the single frame, or with ``animate`` an image over the steps."""
        if animate:
            jobs.append(
                (
                    _expert_map,
                    dict(
                        path=name,
                        values=values,
                        title=title,
                        label=ylabel,
                        steps=steps,
                        baseline=baseline,
                    ),
                )
            )
        else:
            plot(name, _bars, values=values, title=title, ylabel=ylabel, ylim=ylim)

    def stack(key, frames=frames):
        return np.stack([frame[key] for frame in frames])

    os.makedirs(output_dir, exist_ok=True)
    grid = "x" in frames[0]
    if grid:
        x = frames[0]["x"]
    if grid and frames[0]["predictions"].shape[-1] == 1:
        ys = stack("predictions").transpose(0, 2, 1)
        plot(
            os.path.join(output_dir, "model"),
            _lines,
            x=x,
            ys=ys,
            labels=["Model"],
            title="Model",
            ylim=_ylim(ys, ref),
            ref=ref,
            breakpoints=breakpoints,
        )
    if "per_expert_loss" in frames[0]:
        losses = stack("per_expert_loss")
        name = os.path.join(output_dir, "per_expert_loss")
        title = "Per Expert Loss (Last Layer)"
        if animate:
            jobs.append(
                (
                    _expert_lines,
                    dict(
                        path=name,
                        values=losses,
                        steps=steps,
                        title=title,
                        ylabel="Loss",
                    ),
                )
            )
        else:
            plot(
                name,
                _bars,
                values=losses,
                title=title,
                ylabel="Loss",
                ylim=(0, float(np.nanmax(losses)) * 1.1 or 1.0),
            )

    num_layers = len(frames[0]["layers"])
    for i in range(num_layers):
        layer_frames = [frame["layers"][i] for frame in frames]
        has = layer_frames[0].keys()
        layer_dir = os.path.join(output_dir, f"layer{i}")
        os.makedirs(layer_dir, exist_ok=True)

        def path(name):
            return os.path.join(layer_dir, name)

        def get(key):
            return stack(key, layer_frames)

        counts = get("sample_count")
        num_experts = counts.shape[1]
        bars(
            path("sample_count"),
            values=counts,
            title=f"Layer {i}: Per Expert Sample Count",
            ylabel="Sample Count",
            ylim=(0, int(counts.max()) + 1),
            # Uniform load.
            baseline=counts[0].sum() / num_experts,
        )
        routing_biases = 0
        if "routing_biases" in has:
            # Leave out the infinite bonus of unexplored experts.
            routing_biases = get("routing_biases")
            routing_biases = np.where(np.isinf(routing_biases), np.nan, routing_biases)
            bars(
                path("routing_biases"),
                values=routing_biases,
                title=f"Layer {i}: Routing Biases",
                ylabel="Bias",
                ylim=_ylim(routing_biases),
            )
            routing_biases = routing_biases[:, None, :]
        if "pulls" in has and animate:
            jobs.append(
                (
                    _expert_lines,
                    dict(
                        path=path("pulls"),
                        values=get("pulls"),
                        steps=steps,
                        title="Cumulative Pulls",
                        ylabel="Pulls",
                    ),
                )
            )

        if not grid:
            continue

        selected = get("selected_experts")
        scores = (get("gating_scores") + routing_biases).transpose(0, 2, 1)
        plot(
            path("router"),
            _lines,
            x=x,
            ys=scores,
            labels=[f"Expert {e}" for e in range(num_experts)],
            title=f"Layer {i}: Routing Scores",
            ylim=_ylim(scores),
            breakpoints=breakpoints,
            selected_experts=selected,
        )
        expert_outputs = get("expert_outputs")
        if expert_outputs.shape[-1] == 1:
            layer_ref = ref if i == num_layers - 1 else None
            ys = expert_outputs[..., 0].transpose(0, 2, 1)
            plot(
                path("expert"),
                _lines,
                x=x,
                ys=ys,
                labels=[f"Expert {e}" for e in range(num_experts)],
                title=f"Layer {i}: Experts",
                ylim=_ylim(ys, layer_ref),
                ref=layer_ref,
                selected_experts=selected,
            )
        if "index_terms" in has:
            plot(
                path("index"),
                _index,
                x=x,
                index_terms=get("index_terms"),
                selected_experts=selected,
                title=f"Layer {i}: Bandit Index",
            )
        if animate:
            jobs.append(
                (
                    _routing_map,
                    dict(
                        path=path("routing_map"),
                        x=x,
                        selected_experts=selected,
                        num_experts=num_experts,
                        breakpoints=breakpoints,
                        steps=steps,
                    ),
                )
            )

    if not animate:
        for fn, kwargs in jobs:
            fn(**kwargs)
        return
    with ProcessPoolExecutor() as pool:
        for future in [pool.submit(fn, **kwargs) for fn, kwargs in jobs]:
            future.result()


def _breakpoints(target: TargetFunction) -> list[float]:
    """The target's breakpoints inside its domain."""
    if target.breakpoints is None:
        return []
    bps = target.breakpoints.cpu()
    return bps[(bps > target.domain[0]) & (bps < target.domain[1])].tolist()


def plot_target(target: TargetFunction, output_dir: str, grid: bool) -> None:
    """PNGs of the target into ``target/`` of ``output_dir``, over the input grid
    if ``grid``."""
    render(
        [target_frame(target, input_grid(target) if grid else None)],
        os.path.join(output_dir, "target"),
        breakpoints=_breakpoints(target),
    )


def plot_run(frames: list[dict], target: TargetFunction, output_dir: str) -> None:
    """MP4s over the frames into ``anims/`` of ``output_dir``."""
    ref = None
    if "x" in frames[0]:
        y = target(input_grid(target))
        ref = y[:, 0].cpu().numpy() if y.shape[-1] == 1 else None
    breakpoints = _breakpoints(target)

    render(frames, os.path.join(output_dir, "anims"), ref, breakpoints, animate=True)
