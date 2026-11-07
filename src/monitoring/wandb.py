import os

import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def flatten(config: dict, prefix: str = "") -> dict:
    """Nested dicts as one dict with dotted keys."""
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat.update(flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def init_wandb(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    # The config groups' choices (e.g. router=ucb) next to the flattened config,
    # whose keys then match those of sweep parameters like router.gamma.
    choices = {
        group: choice
        for group, choice in hydra_cfg.runtime.choices.items()
        if not group.startswith("hydra")
    }
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name or hydra_cfg.job.override_dirname or None,
        tags=cfg.wandb.run_tags,
        config={**choices, **flatten(OmegaConf.to_container(cfg, resolve=True))},
    )


def log_plots(output_dir: str) -> None:
    """Every PNG and MP4 of `plot_run`, keyed by its path."""
    log = {}
    for subdir, panel in (
        ("target", "target"),
        ("init", "init"),
        ("final", "final"),
        ("anims", "animations"),
    ):
        root = os.path.join(output_dir, subdir)
        for dirpath, _, files in os.walk(root):
            for file in files:
                path = os.path.join(dirpath, file)
                name, ext = os.path.splitext(os.path.relpath(path, root))
                log[f"{panel}/{name}"] = (
                    wandb.Video(path, format="mp4")
                    if ext == ".mp4"
                    else wandb.Image(path)
                )
    wandb.log(log)
