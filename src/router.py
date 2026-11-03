import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


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
    ):
        super().__init__(input_dim, num_experts, top_k)
        self.epsilon = epsilon
        self.sigma = sigma
        self.alpha = alpha

        self.reward_model = nn.Linear(input_dim, num_experts, bias=True)
        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=lr)

        self.num_tokens = 0
        self.pulls = [0] * num_experts

        # History of past reward models, used for the window to estimate increment
        self.history = torch.zeros(num_experts, 1024, input_dim + 1)

        # r_i^l is linear in z = [x, 1], so the sums over pulls in the index reduce
        # to one row per expert. The payoff and increment are the two summands.
        self.register_buffer("payoff", torch.zeros(num_experts, input_dim + 1))
        self.register_buffer("increment", torch.zeros(num_experts, input_dim + 1))
        self.register_buffer("bonus", torch.full((num_experts,), float("inf")))

        self._metrics: dict[str, float] = {}

    def _index_terms(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Payoff, increment and bonus of the index, each ``(B, num_experts)``."""
        z = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        payoff = z @ self.payoff.T
        return payoff, z @ self.increment.T, self.bonus.expand_as(payoff)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            payoff, increment, bonus = self._index_terms(x)
            _, selected_experts = torch.topk(
                payoff + increment + bonus, self.top_k, dim=1
            )
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
        """Learn from the per-token losses ``(B,)`` of the final predictions for ``x``."""
        rewards = torch.exp(-losses).unsqueeze(1).expand_as(selected_experts)

        with torch.no_grad():
            payoff, increment, bonus = (
                torch.gather(term, 1, selected_experts) for term in self._index_terms(x)
            )
            explored = bonus.isfinite()

        # One regression step of the reward model on the observed rewards.
        predicted = torch.gather(self.reward_model(x), 1, selected_experts)
        reward_model_loss = (predicted - rewards).pow(2).mean()

        self.optimizer.zero_grad()
        reward_model_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self._metrics = {
            "bandit/reward": rewards.mean().item(),
            "bandit/reward_model_loss": reward_model_loss.item(),
            "bandit/index_payoff": payoff[explored].mean().item(),
            "bandit/index_increment": increment[explored].mean().item(),
            "bandit/index_bonus": bonus[explored].mean().item(),
        }

        with torch.no_grad():
            self.num_tokens += x.shape[0]
            self._record_pulls(selected_experts)
            self._update_index()

    def _record_pulls(self, selected_experts: torch.Tensor) -> None:
        """Count the new pulls and store the reward model's row for each of them."""
        routed = torch.zeros(
            selected_experts.shape[0], self.num_experts, device=selected_experts.device
        )
        routed.scatter_(1, selected_experts, 1.0)
        counts = routed.sum(dim=0).int().tolist()

        while max(n + c for n, c in zip(self.pulls, counts)) > self.history.shape[1]:
            self.history = torch.cat(
                [self.history, torch.zeros_like(self.history)], dim=1
            )

        rows = torch.cat(
            [self.reward_model.weight, self.reward_model.bias.unsqueeze(1)], dim=1
        ).cpu()
        for i, count in enumerate(counts):
            self.history[i, self.pulls[i] : self.pulls[i] + count] = rows[i]
            self.pulls[i] += count

    def _update_index(self) -> None:
        """Evaluate the index's sums over pulls for the next token."""
        t = self.num_tokens + 1
        payoff = torch.zeros(self.payoff.shape)
        increment = torch.zeros(self.increment.shape)
        bonus = torch.full(self.bonus.shape, float("inf"))
        for i, n in enumerate(self.pulls):
            h = math.floor(self.epsilon * n)
            if h == 0:
                # For window size 0, keep infinite bonus
                continue
            recent = self.history[i, n - h : n]  # pulls l = n - h + 1, ..., n
            previous = self.history[i, n - 2 * h : n - h]  # pulls l - h
            pulls = torch.arange(n - h + 1, n + 1).unsqueeze(1)
            payoff[i] = recent.mean(dim=0)
            increment[i] = ((t - pulls) * (recent - previous) / h).mean(dim=0)
            bonus[i] = (
                self.sigma
                * (t - n + h - 1)
                * math.sqrt(10 * self.alpha * math.log(t) / h**3)
            )
        self.payoff.copy_(payoff)
        self.increment.copy_(increment)
        self.bonus.copy_(bonus)

    def metrics(self) -> dict[str, float]:
        """Means over the selected experts of the last `update`'s batch."""
        return self._metrics
