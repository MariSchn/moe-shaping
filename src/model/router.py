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

    def __init__(self, input_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route a batch ``x`` of shape ``(B, input_dim)``.

        Returns:
            gating_scores: (B, num_experts) score of every expert.
            selected_experts: (B, top_k) indices of the selected experts.
            gate_weights: (B, top_k) weights of the selected experts' outputs.
        """
        gating_scores, selection_scores = self.scores(x)
        selected_experts = selection_scores.topk(self.top_k, dim=-1).indices
        gate_weights = self.gate_weights(gating_scores.gather(-1, selected_experts))
        return gating_scores, selected_experts, gate_weights

    @abstractmethod
    def scores(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gating scores and selection scores, each ``(B, num_experts)``.

        The top-k experts by selection score are selected, and `gate_weights` turns
        their gating scores into the weights of their outputs.
        """

    def gate_weights(self, scores: torch.Tensor) -> torch.Tensor:
        """Weights of experts' outputs from their gating scores, over the last dim."""
        return torch.softmax(scores, dim=-1)

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

    Selects the top-k experts by score, shifted by the routing biases, and
    normalizes their scores into gate weights.
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

        # Only used for the top-k selection, zero unless set by MLPRouterWithBonus.
        self.register_buffer("routing_biases", torch.zeros(num_experts))

    def scores(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gating_scores = self.scorer(x)
        return gating_scores, gating_scores + self.routing_biases

    def gate_weights(self, scores: torch.Tensor) -> torch.Tensor:
        if self.activation == "softmax":
            return torch.softmax(scores, dim=-1)
        sig = torch.sigmoid(scores)
        return sig / sig.sum(dim=-1, keepdim=True).clamp(min=1e-9)


class MLPRouterWithBonus(MLPRouter, ABC):
    """`MLPRouter` whose routing biases are an untrained per-expert bonus, set
    after each batch from the experts' pulls (tokens routed to them) so far."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tokens = 0
        self.register_buffer("pulls", torch.zeros(self.num_experts))

    def update(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        losses: torch.Tensor,
    ) -> None:
        counts = torch.bincount(selected_experts.flatten(), minlength=self.num_experts)
        self.pulls += counts
        self.num_tokens += len(selected_experts)
        self.update_bonus(counts)

    @abstractmethod
    def update_bonus(self, counts: torch.Tensor) -> None:
        """Update ``routing_biases`` after a batch with ``counts`` pulls per expert."""

    def snapshot(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"routing_biases": self.routing_biases.clone()}


class LoadBalancingRouter(MLPRouterWithBonus):
    """Auxiliary loss-free load balancing (DeepSeek-V3): raises the bonus of
    underloaded experts by gamma and lowers that of overloaded ones."""

    def __init__(self, *args, gamma: float = 1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def update_bonus(self, counts: torch.Tensor) -> None:
        loads = counts / counts.sum()
        self.routing_biases -= self.gamma * torch.sign(loads - 1 / self.num_experts)


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

    def update_bonus(self, counts: torch.Tensor) -> None:
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

    def update_bonus(self, counts: torch.Tensor) -> None:
        t = self.num_tokens + 1
        self.routing_biases.copy_(
            rising_bandit_bonus(self.pulls, t, self.epsilon, self.sigma, self.alpha)
        )


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

    def snapshot(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"index_terms": torch.stack(self.index_terms(x))}
