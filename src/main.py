import os

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import wandb
from training import train
from utils import get_device, set_seed
from visualization import plot_run


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


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    if cfg.wandb.enabled:
        init_wandb(cfg)
    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = get_device()
    target = instantiate(cfg.target, device=device)
    model = instantiate(cfg.model).to(device)

    if cfg.training.freeze_router:
        for layer in model.layers:
            layer.router.requires_grad_(False)
    if cfg.training.freeze_experts:
        for layer in model.layers:
            layer.experts.requires_grad_(False)
    optimizer = instantiate(cfg.optimizer)(
        [p for p in model.parameters() if p.requires_grad]
    )
    if cfg.wandb.enabled:
        wandb.summary["num_parameters"] = sum(p.numel() for p in model.parameters())
        wandb.summary["num_trainable_parameters"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

    frames = train(model, target, optimizer, cfg)

    plot_run(frames, target, output_dir)
    if cfg.wandb.enabled:
        log_plots(output_dir)


if __name__ == "__main__":
    main()
