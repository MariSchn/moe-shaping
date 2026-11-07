import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

import wandb
from monitoring.visualization import plot_run, plot_target
from monitoring.wandb import init_wandb, log_plots
from training import train
from utils import get_device, set_seed


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

    plot_target(target, output_dir, grid=model.input_dim == 1)
    if cfg.wandb.enabled:
        log_plots(output_dir, ["target"])

    frames = train(model, target, optimizer, cfg)

    plot_run(frames, target, output_dir)
    if cfg.wandb.enabled:
        log_plots(output_dir, ["init", "final", "anims"])


if __name__ == "__main__":
    main()
