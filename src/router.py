import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

INDEX_TERMS = ("payoff", "increment", "bonus", "covariance")


class Router(nn.Module, ABC):
    """Picks ``top_k`` of ``num_experts`` experts per input and weights their outputs."""

    def __init__(self, input_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k

    @abstractmethod
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route a batch ``x`` of shape ``(B, input_dim)``.

        Returns:
            gating_scores: (B, num_experts) score of every expert.
            selected_experts: (B, top_k) indices of the selected experts.
            gate_probs: (B, top_k) weights of the selected experts' outputs.
        """

    def update(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        losses: torch.Tensor,
    ) -> None:
        """Learn from the per-token losses ``(B,)`` of the final predictions for ``x``.

        Only for routers that don't learn from the gradients of the task loss.
        """

    def metrics(self) -> dict[str, float]:
        return {}


class LinearRouter(Router):
    """Linear gating scores ``x @ W.T + b``, one row of ``W`` per expert.

    Selects the top-k experts by score, shifted by the auxiliary loss-free
    routing biases, and normalizes their scores into gate weights.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int,
        activation: str = "softmax",
        initial_weights: list[list[float]] = None,
        initial_biases: list[float] = None,
    ):
        super().__init__(input_dim, num_experts, top_k)

        if activation not in ("softmax", "sigmoid"):
            raise ValueError(
                f"activation must be 'softmax' or 'sigmoid', got {activation!r}"
            )
        self.activation = activation

        self.linear = nn.Linear(input_dim, num_experts, bias=True)

        # Auxiliary loss-free load balancing (ignored if cfg.training.load_balancing_loss_weight < 0)
        self.register_buffer("routing_biases", torch.zeros(num_experts))

        # Insert hard-coded weights and biases if provided
        if initial_weights is not None:
            assert len(initial_weights) == num_experts, (
                f"Initial gating weights must have {num_experts} elements, got {len(initial_weights)}"
            )
            self.linear.weight.data = torch.tensor(initial_weights, dtype=torch.float32)
        if initial_biases is not None:
            assert len(initial_biases) == num_experts, (
                f"Initial gating biases must have {num_experts} elements, got {len(initial_biases)}"
            )
            self.linear.bias.data = torch.tensor(initial_biases, dtype=torch.float32)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gating_scores = self.linear(x)
        _, selected_experts = torch.topk(
            gating_scores + self.routing_biases, self.top_k, dim=1
        )
        selected_scores = torch.gather(gating_scores, dim=1, index=selected_experts)

        # Normalize only over the top-k experts, not all experts.
        if self.activation == "softmax":
            gate_probs = torch.softmax(selected_scores, dim=1)
        else:  # "sigmoid": sigmoid + sum-normalization
            sig = torch.sigmoid(selected_scores)
            gate_probs = sig / sig.sum(dim=1, keepdim=True).clamp(min=1e-9)

        return gating_scores, selected_experts, gate_probs

    def update_routing_biases(
        self,
        selected_experts: torch.Tensor,
        gamma: float = 1e-3,
        target_load: float | None = None,
    ):
        if gamma <= 0:
            return

        B, _ = selected_experts.shape

        # Use uniform load as default target load
        target_load = target_load or self.top_k / self.num_experts

        # Calculate the mean load of each expert
        expert_counts = torch.zeros(self.num_experts, device=selected_experts.device)
        expert_counts.scatter_add_(
            0,
            selected_experts.flatten(),
            torch.ones(selected_experts.numel(), device=selected_experts.device),
        )
        expert_loads = expert_counts / B

        # Calculate the difference between the expert load and the target load
        load_differences = expert_loads - target_load

        bias_updates = torch.zeros(self.num_experts, device=selected_experts.device)
        bias_updates[load_differences < 0] = gamma
        bias_updates[load_differences > 0] = -gamma

        # Update the routing biases
        self.routing_biases += bias_updates


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
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, selected_experts = torch.topk(sum(self.index_terms(x)), self.top_k, dim=1)
        predicted_rewards = self.reward_model(x)
        gate_probs = torch.softmax(
            torch.gather(predicted_rewards, 1, selected_experts), dim=1
        )
        return predicted_rewards, selected_experts, gate_probs

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
            "bandit/reward": rewards.mean().item(),
            "bandit/reward_model_loss": reward_model_loss.item(),
            "bandit/index_payoff": payoff[explored].mean().item(),
            "bandit/index_increment": increment[explored].mean().item(),
            "bandit/index_bonus": bonus[explored].mean().item(),
            "bandit/index_covariance": covariance[explored].mean().item(),
        }
        for name, term in zip(INDEX_TERMS, index_terms):
            for i, value in enumerate(term.mean(dim=0).tolist()):
                self._metrics[f"bandit_index/{name}_{i}"] = value

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
        for i, rows in enumerate(self.history):
            n = len(rows)
            h = math.floor(self.epsilon * n)
            if h == 0:
                # For window size 0, keep infinite bonus
                continue
            recent = rows[n - h :]  # pulls l = n - h + 1, ..., n
            previous = rows[n - 2 * h : n - h]  # pulls l - h
            pulls = torch.arange(n - h + 1, n + 1).unsqueeze(1)
            self.payoff[i] = recent.mean(dim=0)
            self.increment[i] = ((t - pulls) * (recent - previous) / h).mean(dim=0)
            self.bonus[i] = (
                self.sigma
                * (t - n + h - 1)
                * math.sqrt(10 * self.alpha * math.log(t) / h**3)
            )

    def metrics(self) -> dict[str, float]:
        """Means over the last `update`'s batch: `bandit/` over the selected experts,
        `bandit_index/` per expert over all tokens, selected or not."""
        return self._metrics
