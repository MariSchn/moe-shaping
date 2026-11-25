"""A target that a regular top-2 MoE cannot represent, but a MoE with separate selection
and gating learns: configs/target/separation_counterexample.yaml.

Trains four models on it with src/'s training loop, 3 seeds each: the regular MoE
(router=mlp), the regular MoE with aux-loss-free load balancing (router=load_balancing)
or discounted UCB (router=ucb) and the separated MoE (router=separated). Every seed
trains 4 restarts and keeps the one with the lowest training loss, then the script plots
one row of final fits per model.

Usage: python experiments/separation_counterexample.py [--out DIR]
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from training import train  # noqa: E402
from utils import sample_uniformly, set_seed  # noqa: E402

MODELS = {
    "joint": ["router=mlp"],
    # The best of gamma in (1e-4, 1e-3, 1e-2, 3e-2) on this target.
    "lb": ["router=load_balancing", "router.gamma=1e-4"],
    "ducb": ["router=ucb", "router.discount=0.5"],
    "separated": ["router=separated"],
}
COMMON = [
    "target=separation_counterexample",
    "model.num_experts=3",
    "training.lr_decay=true",
    "wandb.enabled=false",
]
GRID = torch.linspace(-1, 1, 801)[:, None]


def config(overrides, seed, steps):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(
            "config",
            overrides=[
                *COMMON,
                *overrides,
                f"training.num_steps={steps}",
                f"logging.frame_every={steps}",
                f"seed={seed}",
            ],
        )


def train_run(overrides, seed, steps):
    cfg = config(overrides, seed, steps)
    set_seed(seed)
    target = instantiate(cfg.target, device=torch.device("cpu"))
    model = instantiate(cfg.model)
    params = [p for p in model.parameters() if p.requires_grad]
    train(model, target, instantiate(cfg.optimizer)(params), cfg)
    return model, target


def errors(y, y_star):
    """MSE and the fraction of inputs where the fit is visibly off (|error| > 0.15)."""
    off = ((y - y_star).abs() > 0.15).float().mean().item()
    return ((y - y_star) ** 2).mean().item(), off


def run_seed(model_name, seed, restarts, steps):
    """The final fit on GRID of the restart with the lowest training loss."""
    torch.set_num_threads(1)
    runs = []
    for restart in range(restarts):
        model, target = train_run(MODELS[model_name], restarts * seed + restart, steps)
        with torch.no_grad():
            x = sample_uniformly(target.domain, 4096)
            loss = ((model(x)[0] - target(x)) ** 2).mean().item()
            y = model(GRID)[0][:, 0]
            runs.append((loss, y, *errors(y, target(GRID)[:, 0])))
    loss, y, mse, off = min(runs, key=lambda run: run[0])
    print(
        f"{model_name} seed {seed}: chosen mse {mse:.4f}, off {off:.1%}; all restarts: "
        + ", ".join(f"{m:.4f}/{o:.1%}" for _, _, m, o in runs),
        flush=True,
    )
    return y, off


def overview(rows, y_star, restarts, path):
    """One row of final fits per model and one column per seed."""
    ink, muted, model_color, surface = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb"
    seeds = len(rows["joint"])
    fig, axes = plt.subplots(
        len(rows),
        seeds,
        figsize=(3.4 * seeds + 1.4, 3 * len(rows)),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    fig.set_facecolor(surface)
    labels = {
        "joint": "Regular MoE",
        "lb": "Regular MoE\n+ aux-loss-free LB",
        "ducb": "Regular MoE\n+ discounted UCB",
        "separated": "Separated MoE",
    }
    for r, model in enumerate(rows):
        for c, (y, off) in enumerate(rows[model]):
            ax = axes[r, c]
            ax.set_facecolor(surface)
            ax.plot(GRID[:, 0], y_star, color=ink, ls="--", lw=1.8, label="target")
            ax.plot(GRID[:, 0], y, color=model_color, lw=2, label="trained model")
            ax.text(
                0.97,
                0.95,
                f"{off:.0%} of inputs off",
                color=muted,
                fontsize=10,
                ha="right",
                va="top",
                transform=ax.transAxes,
            )
            ax.set_ylim(-2.6, 2.6)
            ax.grid(color="#e4e3df", lw=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(colors=muted, labelsize=9)
            if r == 0:
                ax.set_title(f"Seed {c}", color=ink, fontsize=11)
        axes[r, 0].set_ylabel(labels[model], color=ink, fontsize=11)
    axes[0, 0].set_xticks([-1, -0.5, 0, 0.5, 1])
    legend = fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside upper center",
        ncol=2,
        frameon=False,
        fontsize=10,
        title="Same target, 3 experts with 2 active",
        title_fontsize=13,
    )
    legend.get_title().set_color(ink)
    fig.supxlabel(
        "Regular MoE: one router picks and weights the experts. Aux-loss-free LB: a "
        "per-expert bias on the selection only (γ = 1e-4, the best of 1e-4 to 3e-2).\n"
        "Discounted UCB: a per-expert exploration bonus on the selection only, its "
        "counts discounted by 0.5 per step.\nSeparated MoE: a selection model picks the "
        "experts, a gating model weights them. Off: |error| > 0.15. Each seed shows "
        f"the run with the lowest training loss of {restarts}.",
        color=muted,
        fontsize=9,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=surface)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/separation-counterexample")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--steps", type=int, default=60_000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jobs = [(model, seed) for model in MODELS for seed in range(args.seeds)]
    with ProcessPoolExecutor() as pool:
        futures = [
            pool.submit(run_seed, model, seed, args.restarts, args.steps)
            for model, seed in jobs
        ]
        results = [future.result() for future in futures]
    rows = {model: [] for model in MODELS}
    for (model, _), result in zip(jobs, results):
        rows[model].append(result)

    target = instantiate(config([], 0, args.steps).target, device=torch.device("cpu"))
    path = os.path.join(args.out, "overview.png")
    overview(rows, target(GRID)[:, 0], args.restarts, path)
    print("saved", path)


if __name__ == "__main__":
    main()
