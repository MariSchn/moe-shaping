import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

INDEX_TERMS = ("payoff", "increment", "bonus", "covariance")


def mlp(
    input_dim: int, output_dim: int, hidden_dim: int | None = None, num_layers: int = 1
) -> nn.Sequential:
    """``num_layers`` linear layers with ReLUs in between."""
    if num_layers > 1 and hidden_dim is None:
        raise ValueError("hidden_dim must be set when num_layers > 1")
    dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
    layers = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(d_in, d_out), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


class Router(nn.Module, ABC):
    """Picks ``top_k`` of ``num_experts`` experts per input and weights their outputs."""

    # Whether training batches must hold every input twice, in consecutive tokens.
    paired_inputs = False

    def __init__(self, input_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Route a batch ``x`` of shape ``(B, input_dim)``.

        Returns:
            gating_scores: (B, num_experts) score of every expert.
            selected_experts: (B, top_k) indices of the selected experts.
            gate_weights: (B, top_k) weights of the selected experts' outputs.
            explored: (B,) whether each token explored a set of experts the router
                would not select, so that it must not train the experts or the gating;
                None for routers that do not explore this way.
        """
        gating_scores, selection_scores = self.scores(x)
        selected_experts = selection_scores.topk(self.top_k, dim=-1).indices
        gate_weights = self.gate_weights(gating_scores.gather(-1, selected_experts))
        return gating_scores, selected_experts, gate_weights, None

    @abstractmethod
    def scores(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gating scores and selection scores, each ``(B, num_experts)``.

        The top-k experts by selection score are selected, and `gate_weights` turns
        their gating scores into the weights of their outputs.
        """

    def gate_weights(self, scores: torch.Tensor) -> torch.Tensor:
        """Weights of experts' outputs from their gating scores, over the last dim."""
        return torch.softmax(scores, dim=-1)

    def combo_weights(
        self, x: torch.Tensor, gating_scores: torch.Tensor, combos: torch.Tensor
    ) -> torch.Tensor:
        """Weights ``(B, C, top_k)`` the router would give each set of experts in
        ``combos`` ``(C, top_k)``, for the inputs ``x`` with the ``gating_scores``
        `forward` gave them."""
        return self.gate_weights(gating_scores[:, combos])

    def optimizers(self) -> list[torch.optim.Optimizer]:
        """The optimizers the router steps itself in `update`, for the learning-rate
        schedule."""
        return []

    def update(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        losses: torch.Tensor,
    ) -> None:
        """Learn from the per-token losses ``(B,)`` of the final predictions for ``x``.

        For anything not learned from the gradients of the task loss.
        """

    def metrics(self) -> dict[str, float]:
        return {}

    def snapshot(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """State to plot, for the inputs ``x``."""
        return {}


def rising_bandit_bonus(
    pulls: torch.Tensor, t: int, epsilon: float, sigma: float, alpha: float
) -> torch.Tensor:
    """R-ed-UCB's exploration bonus per expert (Metelli et al., 2022)

        sigma (t - N + h - 1) sqrt(10 log(1 / delta_t) / h^3)

    with N its pulls, h = floor(epsilon * N) and delta_t = t^-alpha. Infinite while h = 0.
    """
    h = torch.floor(epsilon * pulls)
    bonus = sigma * (t - pulls + h - 1) * torch.sqrt(10 * alpha * math.log(t) / h**3)
    return torch.where(h > 0, bonus, math.inf)


class MLPRouter(Router):
    """Gating scores from an MLP, linear for ``num_layers == 1``.

    Selects the top-k experts by score and normalizes their scores into gate weights.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int,
        activation: str = "softmax",
        hidden_dim: int | None = None,
        num_layers: int = 1,
    ):
        super().__init__(input_dim, num_experts, top_k)

        if activation not in ("softmax", "sigmoid"):
            raise ValueError(
                f"activation must be 'softmax' or 'sigmoid', got {activation!r}"
            )
        self.activation = activation

        self.scorer = mlp(input_dim, num_experts, hidden_dim, num_layers)

    def scores(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gating_scores = self.scorer(x)
        return gating_scores, gating_scores

    def gate_weights(self, scores: torch.Tensor) -> torch.Tensor:
        if self.activation == "softmax":
            return torch.softmax(scores, dim=-1)
        sig = torch.sigmoid(scores)
        return sig / sig.sum(dim=-1, keepdim=True).clamp(min=1e-9)


class MLPRouterWithBonus(MLPRouter, ABC):
    """`MLPRouter` whose selection scores get an untrained bonus, updated after each
    batch from the experts' pulls (tokens routed to them) so far.

    The bonus is per expert, in ``routing_biases``, unless `bonus` is overridden.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tokens = 0
        self.register_buffer("pulls", torch.zeros(self.num_experts))
        self.register_buffer("routing_biases", torch.zeros(self.num_experts))

    def bonus(self, x: torch.Tensor) -> torch.Tensor:
        """The bonus of each expert for each row of ``x``, broadcastable to
        ``(B, num_experts)``."""
        return self.routing_biases

    def scores(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gating_scores = self.scorer(x)
        return gating_scores, gating_scores + self.bonus(x)

    def update(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        losses: torch.Tensor,
    ) -> None:
        counts = torch.bincount(selected_experts.flatten(), minlength=self.num_experts)
        self.pulls += counts
        self.num_tokens += len(selected_experts)
        self.update_bonus(x, selected_experts, counts)

    @abstractmethod
    def update_bonus(
        self, x: torch.Tensor, selected_experts: torch.Tensor, counts: torch.Tensor
    ) -> None:
        """Update the bonus after a batch ``x`` with ``counts`` pulls per expert."""

    def snapshot(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "routing_biases": self.routing_biases.clone(),
            "pulls": self.pulls.clone(),
        }


class LoadBalancingRouter(MLPRouterWithBonus):
    """Auxiliary loss-free load balancing (DeepSeek-V3): raises the bonus of
    underloaded experts by gamma and lowers that of overloaded ones.

    Gamma is constant until step ``decay_start``, decays to 0 along a cosine until
    step ``decay_end`` and stays 0 after (Beam, Reflection AI).
    """

    def __init__(
        self,
        *args,
        gamma: float = 1e-3,
        decay_start: int,
        decay_end: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.decay_start = decay_start
        self.decay_end = decay_end
        self.step = 0

    def update_bonus(
        self, x: torch.Tensor, selected_experts: torch.Tensor, counts: torch.Tensor
    ) -> None:
        progress = (self.step - self.decay_start) / max(
            1, self.decay_end - self.decay_start
        )
        progress = min(1.0, max(0.0, progress))
        gamma = self.gamma * (1 + math.cos(math.pi * progress)) / 2
        loads = counts / counts.sum()
        self.routing_biases -= gamma * torch.sign(loads - 1 / self.num_experts)
        self.step += 1


class UCBRouter(MLPRouterWithBonus):
    """UCB's exploration bonus ucb_weight * sqrt(2 log t / N), with t the tokens and
    N the expert's pulls so far. Infinite while N = 0.

    With ``per_step``, t counts steps and N the expert's share of each batch's
    tokens, making the bonus independent of the batch size. With ``discount`` < 1,
    t and N are discounted by it every step (Discounted UCB), so the bonus levels
    off instead of decaying to zero.
    """

    def __init__(
        self,
        *args,
        ucb_weight: float = 1.0,
        per_step: bool = False,
        discount: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ucb_weight = ucb_weight
        self.per_step = per_step
        self.discount = discount
        self.routing_biases.fill_(math.inf)
        self.rounds = 0.0
        self.register_buffer("round_pulls", torch.zeros(self.num_experts))

    def update_bonus(
        self, x: torch.Tensor, selected_experts: torch.Tensor, counts: torch.Tensor
    ) -> None:
        batch_size = counts.sum().item() / self.top_k
        unit = 1 / batch_size if self.per_step else 1.0
        self.rounds = self.discount * self.rounds + batch_size * unit
        self.round_pulls.mul_(self.discount).add_(counts * unit)
        t = self.rounds + 1
        self.routing_biases.copy_(
            self.ucb_weight * torch.sqrt(2 * math.log(t) / self.round_pulls)
        )


class RisingBanditBonusRouter(MLPRouterWithBonus):
    """R-ed-UCB's exploration bonus, see `rising_bandit_bonus`, with t the tokens and
    N the expert's pulls so far."""

    def __init__(
        self,
        *args,
        epsilon: float = 0.25,
        sigma: float = 0.1,
        alpha: float = 3.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.sigma = sigma
        self.alpha = alpha
        self.routing_biases.fill_(math.inf)

    def update_bonus(
        self, x: torch.Tensor, selected_experts: torch.Tensor, counts: torch.Tensor
    ) -> None:
        t = self.num_tokens + 1
        self.routing_biases.copy_(
            rising_bandit_bonus(self.pulls, t, self.epsilon, self.sigma, self.alpha)
        )


class LinUCBRouter(MLPRouterWithBonus):
    """LinUCB's contextual exploration bonus

        linucb_weight * sqrt(z^T A_i^-1 z)

    with z = [x, 1] and A_i = linucb_ridge I + sum_l z_l z_l^T the design matrix of
    expert i, with z_l the features of its l-th pull. Unlike the other bonuses, it
    differs per token.

    With ``per_step``, each pull counts 1 / batch size, making the bonus independent
    of the batch size. With ``discount`` < 1, A_i is discounted by it every step
    towards linucb_ridge I (D-LinUCB, Russac et al., 2019), so the bonus levels off
    instead of decaying to zero.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int,
        *args,
        linucb_weight: float = 1.0,
        linucb_ridge: float = 1.0,
        per_step: bool = False,
        discount: float = 1.0,
        **kwargs,
    ):
        super().__init__(input_dim, num_experts, top_k, *args, **kwargs)
        self.linucb_weight = linucb_weight
        self.per_step = per_step
        self.discount = discount

        # Per expert, A_i and a factor L_i with A_i^-1 = L_i L_i^T, so
        # sqrt(z^T A_i^-1 z) = ||L_i^T z||. A_i is float64 on the CPU because it gets
        # ill-conditioned when an expert only sees a narrow range of x, and MPS has
        # no float64.
        eye = torch.eye(input_dim + 1).expand(num_experts, -1, -1)
        self.ridge = linucb_ridge * eye.double()
        self.design = self.ridge
        self.register_buffer("design_factor", eye / math.sqrt(linucb_ridge))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, torch.ones_like(x[:, :1])], dim=1)

    def bonus(self, x: torch.Tensor) -> torch.Tensor:
        z = self._features(x)
        return self.linucb_weight * (z @ self.design_factor).norm(dim=-1).T

    @torch.no_grad()
    def update_bonus(
        self, x: torch.Tensor, selected_experts: torch.Tensor, counts: torch.Tensor
    ) -> None:
        z = self._features(x).cpu().double()
        routed = torch.zeros(x.shape[0], self.num_experts, dtype=z.dtype)
        routed.scatter_(1, selected_experts.cpu(), 1.0)
        unit = 1 / x.shape[0] if self.per_step else 1.0
        self.design = (
            self.discount * self.design
            + (1 - self.discount) * self.ridge
            + unit * torch.einsum("be,bi,bj->eij", routed, z, z)
        )
        self.design_factor.copy_(torch.linalg.cholesky(torch.linalg.inv(self.design)))

    def snapshot(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        bonus = self.bonus(x)
        return {
            # The mean over x, to compare with the other routers' per-expert bonus.
            "routing_biases": bonus.mean(dim=0),
            "selection_bonus": bonus,
            "pulls": self.pulls.clone(),
        }


class RandomRouter(Router):
    """Selects ``top_k`` distinct experts uniformly at random per token and weights
    them equally. Has no parameters; its gating scores are all zero."""

    def scores(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        return (
            torch.zeros(B, self.num_experts, device=x.device),
            torch.rand(B, self.num_experts, device=x.device),
        )


class BanditRouter(Router):
    """Contextual version of R-ed-UCB (Metelli et al., 2022), a rested rising bandit.

    Each expert is an arm and each token routed to it a pull. Experts are selected
    by the paper's index

        B_i(x, t) = 1/h sum_{l=N-h+1}^{N} [ r_i^l(x) + (t - l) (r_i^l(x) - r_i^{l-h}(x)) / h ]
                    + sigma (t - N + h - 1) sqrt(10 log(1 / delta_t) / h^3)

    with N the pulls of expert i, t the current token, h = floor(epsilon * N) and
    delta_t = t^-alpha. The index is +inf while h = 0.

    The one change to the paper: the reward of the l-th pull is replaced by
    r_i^l(x), the reward predicted for the current x by a linear reward model as
    it was at that pull.

    With linucb_weight > 0, the index also gets LinUCB's covariance term

        linucb_weight * sqrt(z^T A_i^-1 z),  A_i = linucb_ridge I + sum_l z_l z_l^T

    with z = [x, 1] and z_l the features of the l-th pull of expert i.

    Every expert selected for a token receives the reward exp(-loss) of the final
    prediction. The gating scores are the current predicted rewards and the gate
    weights a softmax over those of the selected experts. The router learns only
    from rewards, through `update`.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int,
        epsilon: float = 0.25,
        sigma: float = 0.1,
        alpha: float = 3.0,
        lr: float = 1e-2,
        linucb_weight: float = 0.0,
        linucb_ridge: float = 1.0,
    ):
        super().__init__(input_dim, num_experts, top_k)
        self.epsilon = epsilon
        self.sigma = sigma
        self.alpha = alpha
        self.linucb_weight = linucb_weight

        self.reward_model = nn.Linear(input_dim, num_experts, bias=True)
        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=lr)

        self.num_tokens = 0

        # Per expert, the reward model's row [W_i, b_i] after each of its pulls.
        self.history = [torch.empty(0, input_dim + 1) for _ in range(num_experts)]

        # r_i^l is linear in z = [x, 1], so the sums over pulls in the index reduce
        # to one row per expert. The payoff and increment are the two summands.
        self.register_buffer("payoff", torch.zeros(num_experts, input_dim + 1))
        self.register_buffer("increment", torch.zeros(num_experts, input_dim + 1))
        self.register_buffer("bonus", torch.full((num_experts,), float("inf")))

        # Per expert, the design matrix A_i and a factor L_i with A_i^-1 = L_i L_i^T,
        # so sqrt(z^T A_i^-1 z) = ||L_i^T z||. A_i is float64 on the CPU because it
        # gets ill-conditioned when an expert only sees a narrow range of x, and MPS
        # has no float64.
        eye = torch.eye(input_dim + 1).expand(num_experts, -1, -1)
        self.design = linucb_ridge * eye.double()
        self.register_buffer("design_factor", eye / math.sqrt(linucb_ridge))

        self._metrics: dict[str, float] = {}

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, torch.ones_like(x[:, :1])], dim=1)

    def index_terms(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Payoff, increment, bonus and covariance term, each ``(B, num_experts)``."""
        z = self._features(x)
        payoff = z @ self.payoff.T
        covariance = self.linucb_weight * (z @ self.design_factor).norm(dim=-1).T
        return payoff, z @ self.increment.T, self.bonus.expand_as(payoff), covariance

    @torch.no_grad()
    def scores(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.reward_model(x), sum(self.index_terms(x))

    def update(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        losses: torch.Tensor,
    ) -> None:
        rewards = torch.exp(-losses).unsqueeze(1).expand_as(selected_experts)

        # One regression step of the reward model on the observed rewards.
        predicted = torch.gather(self.reward_model(x), 1, selected_experts)
        reward_model_loss = (predicted - rewards).pow(2).mean()
        reward_model_loss.backward()
        self.optimizer.step()
        # The main optimizer holds these parameters too: leave it no gradient to apply.
        self.optimizer.zero_grad()

        index_terms = self.index_terms(x)
        payoff, increment, bonus, covariance = (
            torch.gather(term, 1, selected_experts) for term in index_terms
        )
        explored = bonus.isfinite()
        self._metrics = {
            "reward_model_loss": reward_model_loss.item(),
            "index_payoff": payoff[explored].mean().item(),
            "index_increment": increment[explored].mean().item(),
            "index_bonus": bonus[explored].mean().item(),
            "index_covariance": covariance[explored].mean().item(),
        }

        self.num_tokens += x.shape[0]
        self._record_pulls(selected_experts)
        self._update_index()
        if self.linucb_weight > 0:
            self._update_design(x, selected_experts)

    @torch.no_grad()
    def _record_pulls(self, selected_experts: torch.Tensor) -> None:
        """Store the reward model's row for each new pull."""
        counts = torch.bincount(selected_experts.flatten(), minlength=self.num_experts)
        rows = torch.cat(
            [self.reward_model.weight, self.reward_model.bias.unsqueeze(1)], dim=1
        ).cpu()
        for i, count in enumerate(counts.tolist()):
            self.history[i] = torch.cat([self.history[i], rows[i].expand(count, -1)])

    @torch.no_grad()
    def _update_design(self, x: torch.Tensor, selected_experts: torch.Tensor) -> None:
        """Add the new pulls to the design matrices."""
        z = self._features(x).cpu().double()
        routed = torch.zeros(x.shape[0], self.num_experts, dtype=z.dtype)
        routed.scatter_(1, selected_experts.cpu(), 1.0)
        self.design += torch.einsum("be,bi,bj->eij", routed, z, z)
        self.design_factor.copy_(torch.linalg.cholesky(torch.linalg.inv(self.design)))

    def _update_index(self) -> None:
        """Evaluate the index's sums over pulls for the next token."""
        t = self.num_tokens + 1
        num_pulls = torch.tensor(
            [len(rows) for rows in self.history], dtype=torch.float
        )
        self.bonus.copy_(
            rising_bandit_bonus(num_pulls, t, self.epsilon, self.sigma, self.alpha)
        )
        for i, rows in enumerate(self.history):
            n = len(rows)
            h = math.floor(self.epsilon * n)
            if h == 0:
                continue
            recent = rows[n - h :]  # pulls l = n - h + 1, ..., n
            previous = rows[n - 2 * h : n - h]  # pulls l - h
            pulls = torch.arange(n - h + 1, n + 1).unsqueeze(1)
            self.payoff[i] = recent.mean(dim=0)
            self.increment[i] = ((t - pulls) * (recent - previous) / h).mean(dim=0)

    def metrics(self) -> dict[str, float]:
        """Means over the last `update`'s batch and its selected experts."""
        return self._metrics

    def optimizers(self) -> list[torch.optim.Optimizer]:
        return [self.optimizer]

    def snapshot(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "index_terms": torch.stack(self.index_terms(x)),
            "pulls": torch.tensor([len(rows) for rows in self.history]),
        }


class SeparatedRouter(Router):
    """Separate models for which experts to select and how to weight them.

    A selection head scores the experts and its top-k are selected. It learns only from
    the losses of its own picks: in training the inputs come in pairs, the first token
    of a pair takes the top-k and the second a uniformly random set, and the set with
    the lower loss should score (the sum of its experts' scores) at least ``margin``
    higher, a hinge weighted by the loss gap and stepped by the router's own Adam. The
    second token does not train the experts or the gating.

    The gating is conditioned on the selected set: every set has its own logits for its
    experts but the last (whose logit is 0), softmaxed into the weights. They start at 0
    and learn from the task loss's gradients. Its gating scores are the selection
    scores, for plots.
    """

    paired_inputs = True

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int,
        hidden_dim: int | None = None,
        num_layers: int = 1,
        lr: float = 0.3,
        margin: float = 0.3,
    ):
        super().__init__(input_dim, num_experts, top_k)
        self.margin = margin
        self.selector = mlp(input_dim, num_experts, hidden_dim, num_layers)
        self.optimizer = torch.optim.Adam(self.selector.parameters(), lr=lr)

        # Every set of top_k experts, increasing within and across sets.
        self.register_buffer(
            "sets", torch.combinations(torch.arange(num_experts), top_k)
        )
        self.gate = mlp(input_dim, len(self.sets) * (top_k - 1), hidden_dim, num_layers)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        self._metrics: dict[str, float] = {}

    @torch.no_grad()
    def scores(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        selection_scores = self.selector(x)
        return selection_scores, selection_scores

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        selection_scores, _ = self.scores(x)
        selected = selection_scores.topk(self.top_k, dim=-1).indices
        explored = None
        if self.training:
            # The second token of every pair explores a uniformly random set.
            random_sets = torch.rand(x.shape[0], self.num_experts, device=x.device)
            random_sets = random_sets.topk(self.top_k, dim=-1).indices
            second = (torch.arange(x.shape[0], device=x.device) % 2 == 1)[:, None]
            greedy, selected = selected, torch.where(second, random_sets, selected)
            explored = (selected.sort(-1).values != greedy.sort(-1).values).any(-1)
        return selection_scores, selected, self.set_weights(x, selected), explored

    def set_weights(self, x: torch.Tensor, sets: torch.Tensor) -> torch.Tensor:
        """Weights ``(B, ..., top_k)`` of the sets of experts ``(B, ..., top_k)``."""
        order = sets.sort(dim=-1)
        ids = (order.values.unsqueeze(-2) == self.sets).all(-1).float().argmax(-1)
        logits = self.gate(x).view(len(x), len(self.sets), self.top_k - 1)
        index = ids.view(len(x), -1, 1).expand(-1, -1, self.top_k - 1)
        logits = logits.gather(1, index)
        logits = torch.cat([logits, torch.zeros_like(logits[..., :1])], dim=-1)
        weights = torch.softmax(logits, dim=-1).view(sets.shape)
        # From increasing expert order back to the order of `sets`.
        return torch.empty_like(weights).scatter_(-1, order.indices, weights)

    @torch.no_grad()
    def combo_weights(
        self, x: torch.Tensor, gating_scores: torch.Tensor, combos: torch.Tensor
    ) -> torch.Tensor:
        return self.set_weights(x, combos.expand(len(x), -1, -1))

    def update(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        losses: torch.Tensor,
    ) -> None:
        """One step of the hinge over the pairs of tokens (2i, 2i + 1)."""
        set_scores = self.selector(x).gather(-1, selected_experts).sum(-1).view(-1, 2)
        losses = losses.view(-1, 2)
        sets = selected_experts.sort(-1).values.view(-1, 2, self.top_k)
        differ = (sets[:, 0] != sets[:, 1]).any(-1)
        second_better = torch.sign(losses[:, 0] - losses[:, 1])
        gap = (losses[:, 1] - losses[:, 0]).abs()
        lead = second_better * (set_scores[:, 1] - set_scores[:, 0])
        loss = (torch.relu(self.margin - lead) * gap * differ).mean()
        loss.backward()
        self.optimizer.step()
        # The main optimizer holds these parameters too: leave it no gradient to apply.
        self.optimizer.zero_grad()
        self._metrics = {"duel_loss": loss.item()}

    def metrics(self) -> dict[str, float]:
        return self._metrics

    def optimizers(self) -> list[torch.optim.Optimizer]:
        return [self.optimizer]
