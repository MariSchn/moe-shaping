import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm import tqdm

import wandb
from metrics import layer_metrics, load_balancing_loss, routing_regret
from models import Model
from targets import TargetFunction
from utils import sample_uniformly
from visualization import input_grid, record_frame


def train(
    model: Model,
    target: TargetFunction,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
) -> list[dict]:
    """Train ``model`` on ``target``, logging to wandb if enabled.

    Returns frames for plotting, every ``logging.frame_every`` steps and after the
    last step.
    """
    x_grid = input_grid(target) if model.input_dim == 1 else None
    frames = []

    # Running sums of per-step routing regret (cumulative bandit regret).
    regret_gating = regret_optimal = 0.0

    def sample():
        x = sample_uniformly(target.domain, cfg.training.batch_size, model.input_dim)
        x = x.to(target.device)
        return x, target(x)

    pbar = tqdm(range(cfg.training.num_steps), desc="Training")
    for step in pbar:
        x, y = sample()
        predictions, outputs = model(x)
        loss = F.mse_loss(predictions, y)

        # Always calculated for logging, only added to the loss with a positive weight.
        load_balancing_losses = [
            load_balancing_loss(out.gating_scores, out.selected_experts)
            for out in outputs
        ]
        if cfg.training.load_balancing_loss_weight > 0:
            loss = loss + cfg.training.load_balancing_loss_weight * sum(
                load_balancing_losses
            )

        optimizer.zero_grad()
        loss.backward()
        model.update_routers(outputs, (predictions.detach() - y).pow(2).mean(dim=-1))

        if cfg.wandb.enabled:
            last_layer, last = model.layers[-1], outputs[-1]
            step_regret_gating, step_regret_optimal = routing_regret(
                last.expert_outputs.detach(),
                last.gating_scores.detach(),
                predictions.detach(),
                y,
                last_layer.router.gate_weights,
                last_layer.router.top_k,
            )
            regret_gating += step_regret_gating
            regret_optimal += step_regret_optimal
            metrics = {
                "loss": loss.item(),
                "regret/gating": regret_gating,
                "regret/optimal": regret_optimal,
            }
            for i, (layer, out, lb_loss) in enumerate(
                zip(model.layers, outputs, load_balancing_losses)
            ):
                for name, value in layer_metrics(layer, out, lb_loss).items():
                    metrics[f"layer{i}/{name}"] = value
            wandb.log(metrics)

        if step % cfg.logging.frame_every == 0:
            frames.append(record_frame(step, model, outputs, y, x_grid))

        optimizer.step()
        pbar.set_postfix({"loss": loss.item()})

    x, y = sample()
    with torch.no_grad():
        _, outputs = model(x)
    frames.append(record_frame(cfg.training.num_steps, model, outputs, y, x_grid))
    return frames
