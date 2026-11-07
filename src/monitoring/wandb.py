import os

import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def flatten(config: dict, prefix: str = "") -> dict:
    """Nested dicts as one dict with dotted keys, without Hydra's ``_target_``,
    ``_partial_`` and ``_convert_`` keys."""
    flat = {}
    for key, value in config.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            flat.update(flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def init_wandb(cfg: DictConfig) -> None:
    """A run named after the command line overrides, grouped with the runs whose
    overrides differ only in the seed."""
    hydra_cfg = HydraConfig.get()
    overrides = sorted(
        o for o in hydra_cfg.overrides.task if not o.startswith(("wandb.", "hydra."))
    )
    group = ",".join(o for o in overrides if not o.startswith("seed=")) or "default"
    # The config groups' choices (e.g. router=ucb) next to the flattened config.
    choices = {
        name: choice
        for name, choice in hydra_cfg.runtime.choices.items()
        if not name.startswith("hydra")
    }
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name or ",".join(overrides) or None,
        group=group,
        tags=cfg.wandb.run_tags,
        config={**choices, **flatten(OmegaConf.to_container(cfg, resolve=True))},
    )
    wandb.define_metric("loss", summary="min,last")
    wandb.define_metric("regret/*", summary="last")


def log_plots(output_dir: str, subdirs: list[str]) -> None:
    """Every PNG and MP4 in ``subdirs`` of ``output_dir``, keyed ``plots/<path>``."""
    log = {}
    for subdir in subdirs:
        root = os.path.join(output_dir, subdir)
        for dirpath, _, files in os.walk(root):
            for file in files:
                path = os.path.join(dirpath, file)
                name, ext = os.path.splitext(os.path.relpath(path, output_dir))
                log[f"plots/{name}"] = (
                    wandb.Video(path, format="mp4")
                    if ext == ".mp4"
                    else wandb.Image(path)
                )
    wandb.log(log)
