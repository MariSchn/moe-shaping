import torch.nn as nn


class BilevelOptimizer:
    """Drives separate router and expert optimizers.

    With ``router_steps``/``expert_steps`` > 0 it alternates router-only and
    expert-only gradient steps in cycles. When both step counts are 0 it runs in
    *simultaneous* mode: router and experts are both updated every step, each at
    its own learning rate (no alternation).

    Exposes the same zero_grad() / step() interface as a regular PyTorch
    optimizer so the training loop needs no changes.
    """

    def __init__(
        self, router_optimizer, expert_optimizer, router_steps: int, expert_steps: int
    ):
        self._opts = (router_optimizer, expert_optimizer)
        self._cycle = router_steps + expert_steps
        self._router_steps = router_steps
        self._simultaneous = self._cycle == 0
        self._step = 0

    @property
    def in_router_phase(self) -> bool:
        if self._simultaneous:
            return False
        return (self._step % self._cycle) < self._router_steps

    def zero_grad(self):
        for opt in self._opts:
            opt.zero_grad()

    def step(self):
        if self._simultaneous:
            for opt in self._opts:
                opt.step()
        else:
            active = self._opts[0] if self.in_router_phase else self._opts[1]
            active.step()
        self._step += 1

    def metrics(self) -> dict[str, int]:
        return {"bilevel/phase": int(not self.in_router_phase)}


def build_bilevel_optimizer(
    model: nn.Module, optimizer_cls: type, learning_rate: float, bilevel_cfg
) -> BilevelOptimizer | None:
    """A BilevelOptimizer for `model`, or None if `bilevel_cfg` leaves it off.

    Bilevel training is on if either step count is > 0 or either learning rate
    is set. Unset learning rates fall back to `learning_rate`.
    """
    router_steps = bilevel_cfg.router_steps
    expert_steps = bilevel_cfg.expert_steps
    router_lr = bilevel_cfg.get("router_lr")
    expert_lr = bilevel_cfg.get("expert_lr")

    use_bilevel = (
        router_steps > 0
        or expert_steps > 0
        or router_lr is not None
        or expert_lr is not None
    )
    if not use_bilevel:
        return None

    router_opt = optimizer_cls(
        [p for p in model.gating_function.parameters() if p.requires_grad],
        lr=router_lr or learning_rate,
    )
    expert_opt = optimizer_cls(
        [p for p in model.experts.parameters() if p.requires_grad],
        lr=expert_lr or learning_rate,
    )
    return BilevelOptimizer(router_opt, expert_opt, router_steps, expert_steps)
