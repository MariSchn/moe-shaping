"""Trains the MoE router from semi-bandit feedback instead of the task gradient."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class BanditConfig:
    enabled: bool = False
    epsilon: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "BanditConfig":
        return cls(**raw) if raw else cls()


class BanditRouter:
    def __init__(self, config: BanditConfig, model: nn.Module):
        self.cfg = config
        self.model = model

    def selection_bonus(self, x: torch.Tensor) -> torch.Tensor | None:
        """Epsilon-greedy perturbation of the top-k scores, or None if greedy."""
        if not self.cfg.enabled or self.cfg.epsilon <= 0.0:
            return None

        with torch.no_grad():
            logits = self.model.gating_function(x)

        batch_size, num_experts = logits.shape
        device = logits.device

        exploring = torch.rand(batch_size, device=device) < self.cfg.epsilon
        expert = torch.randint(0, num_experts, (batch_size,), device=device)
        spread = logits.max() - logits.min() + 1.0

        bonus = torch.zeros_like(logits)
        bonus[torch.arange(batch_size, device=device), expert] = spread * exploring.to(
            logits.dtype
        )
        return bonus

    def compute_router_gradient(
        self, x: torch.Tensor, output: dict, targets: torch.Tensor
    ) -> dict[str, float]:
        """Write the update into the gating parameters' .grad. Call after backward."""
        if not self.cfg.enabled:
            return {}

        expert_losses = (
            (output["expert_outputs"].detach() - targets.unsqueeze(1)) ** 2
        ).mean(dim=-1)
        selected = output["selected_experts"]
        selected_losses = torch.gather(expert_losses, 1, selected)
        advantage = selected_losses / selected_losses.std().clamp_min(1e-8)

        logits = self.model.gating_function(x)
        gates = torch.softmax(torch.gather(logits, 1, selected), dim=-1)

        surrogate = (gates * advantage).sum(dim=-1).mean()
        surrogate.backward()

        return {"bandit/surrogate": surrogate.item()}
