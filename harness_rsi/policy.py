"""A tiny contextual router; only its parameters, never the LLM, are trained.

The training objective uses a complete, observed reward matrix. Its gradient is
the exact gradient of the softmax policy's expected utility, not a REINFORCE
estimate. Deployment uses ``predict`` (deterministic argmax), so validation must
also evaluate that rule rather than treating the smooth objective as accuracy.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np


FEATURE_NAMES = (
    "log_characters",
    "log_words",
    "log_numbers",
    "log_largest_number",
    "log_sentences",
    "question_marks",
    "addition_words",
    "subtraction_words",
    "multiplication_words",
    "division_words",
    "percentage_words",
    "sequence_words",
    "money_words",
    "comparison_words",
    "decimal_numbers",
)

_NUMBER = re.compile(r"(?<!\w)-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_features(question: str) -> np.ndarray:
    """Return fixed lexical features, with no answer, split, or task ID input."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    lower = question.lower()
    words = re.findall(r"[a-z]+", lower)
    word_set = set(words)
    numbers = _NUMBER.findall(lower)
    finite_numbers = []
    for number in numbers:
        try:
            value = abs(float(number.replace(",", "")))
        except ValueError:
            continue
        if math.isfinite(value):
            finite_numbers.append(value)

    def count(vocabulary: tuple[str, ...]) -> float:
        return float(len(word_set.intersection(vocabulary)))

    features = [
        math.log1p(len(question)),
        math.log1p(len(words)),
        math.log1p(len(numbers)),
        math.log1p(max(finite_numbers, default=0.0)),
        math.log1p(len(re.findall(r"[.!?]", question))),
        float(question.count("?")),
        count(("total", "altogether", "combined", "together", "sum", "added")),
        count(("left", "remaining", "remain", "lost", "gave", "spent", "difference")),
        count(("times", "twice", "triple", "double", "product", "multiply")),
        count(("each", "per", "split", "share", "equally", "divide", "average")),
        count(("percent", "percentage", "discount")) + float("%" in question),
        count(("after", "before", "then", "first", "second", "rest", "finally")),
        count(("dollars", "dollar", "cents", "cost", "price", "paid"))
        + float("$" in question),
        count(("more", "less", "fewer", "than", "extra")),
        float(sum("." in value for value in numbers)),
    ]
    return np.asarray(features, dtype=np.float64)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Stable softmax over the final dimension."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] == 0:
        raise ValueError("logits need a nonempty action dimension")
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")
    exponentials = np.exp(values - np.max(values, axis=-1, keepdims=True))
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def objective_and_gradient(
    weights: np.ndarray,
    bias: np.ndarray,
    features: np.ndarray,
    rewards: np.ndarray,
    *,
    entropy: float = 0.01,
    l2: float = 0.001,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact full-information objective and ascent gradients.

    W has shape (features, actions); rewards has shape (tasks, actions).
    J = mean_i[sum_a pi_ia r_ia + entropy * H(pi_i)] - l2/2 * ||W||^2.
    Biases are not regularized. All entries in rewards must be observed.
    """
    w = np.asarray(weights, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    x = np.asarray(features, dtype=np.float64)
    r = np.asarray(rewards, dtype=np.float64)
    if w.ndim != 2 or b.shape != (w.shape[1],):
        raise ValueError("weights and bias have incompatible shapes")
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != w.shape[0]:
        raise ValueError("features must be a nonempty (tasks, features) matrix")
    if r.shape != (x.shape[0], w.shape[1]):
        raise ValueError("rewards must have shape (tasks, actions)")
    if entropy < 0 or l2 < 0 or not np.isfinite([entropy, l2]).all():
        raise ValueError("entropy and l2 must be finite and nonnegative")
    if not all(np.isfinite(v).all() for v in (w, b, x, r)):
        raise ValueError("all features, parameters, and observed rewards must be finite")

    logits = x @ w + b
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_pi = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    pi = np.exp(log_pi)
    expected = np.sum(pi * r, axis=1, keepdims=True)
    policy_entropy = -np.sum(pi * log_pi, axis=1, keepdims=True)
    logit_gradient = pi * (
        r - expected - entropy * (log_pi + policy_entropy)
    )
    objective = float(
        np.mean(expected + entropy * policy_entropy) - 0.5 * l2 * np.sum(w * w)
    )
    grad_w = x.T @ logit_gradient / len(x) - l2 * w
    grad_b = np.mean(logit_gradient, axis=0)
    return objective, grad_w, grad_b


class LinearPolicy:
    """Small softmax router trained by full-batch gradient ascent.

    The first fit estimates feature normalization from that training batch.
    Normalization is frozen thereafter, including across warm-start fits and
    serialization. Validation/test data must only be passed to probs/predict.
    """

    def __init__(self, n_features: int, n_actions: int, seed: int = 0):
        if n_features < 1 or n_actions < 2:
            raise ValueError("n_features >= 1 and n_actions >= 2 are required")
        self.n_features = int(n_features)
        self.n_actions = int(n_actions)
        self.seed = int(seed)
        self.weights = np.random.default_rng(seed).normal(
            0.0, 0.01, size=(self.n_features, self.n_actions)
        )
        self.bias = np.zeros(self.n_actions, dtype=np.float64)
        self.mean = np.zeros(self.n_features, dtype=np.float64)
        self.scale = np.ones(self.n_features, dtype=np.float64)
        self.scaler_fitted = False
        self.total_steps = 0

    def _features(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"features must have shape (tasks, {self.n_features})")
        if not np.isfinite(x).all():
            raise ValueError("features must be finite")
        return x

    def probs(self, features: np.ndarray) -> np.ndarray:
        x = self._features(features)
        if len(x) == 0:
            return np.empty((0, self.n_actions), dtype=np.float64)
        return softmax(((x - self.mean) / self.scale) @ self.weights + self.bias)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Choose one route per task; ties resolve to the lowest action index."""
        return np.argmax(self.probs(features), axis=1)

    def fit(
        self,
        features: np.ndarray,
        rewards: np.ndarray,
        *,
        steps: int = 300,
        lr: float = 0.2,
        entropy: float = 0.01,
        l2: float = 0.001,
    ) -> list[float]:
        """Warm-start gradient ascent; return objective after every update."""
        x = self._features(features)
        r = np.asarray(rewards, dtype=np.float64)
        if len(x) == 0 or r.shape != (len(x), self.n_actions):
            raise ValueError("training needs a nonempty complete reward matrix")
        if not np.isfinite(r).all():
            raise ValueError("all rewards must be observed and finite")
        if isinstance(steps, bool) or int(steps) != steps or steps < 1:
            raise ValueError("steps must be a positive integer")
        if lr <= 0 or not np.isfinite(lr):
            raise ValueError("lr must be finite and positive")
        if not self.scaler_fitted:
            self.mean = x.mean(axis=0)
            self.scale = x.std(axis=0)
            self.scale[self.scale < 1e-12] = 1.0
            self.scaler_fitted = True
        normalized = (x - self.mean) / self.scale
        history = []
        for _ in range(int(steps)):
            _, grad_w, grad_b = objective_and_gradient(
                self.weights, self.bias, normalized, r, entropy=entropy, l2=l2
            )
            self.weights += lr * grad_w
            self.bias += lr * grad_b
            self.total_steps += 1
            objective, _, _ = objective_and_gradient(
                self.weights, self.bias, normalized, r, entropy=entropy, l2=l2
            )
            history.append(objective)
        return history

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "n_features": self.n_features,
            "n_actions": self.n_actions,
            "seed": self.seed,
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "scaler_fitted": self.scaler_fitted,
            "total_steps": self.total_steps,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "LinearPolicy":
        if state.get("version") != 1:
            raise ValueError("unsupported policy format version")
        policy = cls(state["n_features"], state["n_actions"], seed=state["seed"])
        expected_shapes = {
            "weights": (policy.n_features, policy.n_actions),
            "bias": (policy.n_actions,),
            "mean": (policy.n_features,),
            "scale": (policy.n_features,),
        }
        for name, shape in expected_shapes.items():
            value = np.asarray(state[name], dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid serialized {name}")
            setattr(policy, name, value.copy())
        if np.any(policy.scale <= 0):
            raise ValueError("serialized scales must be positive")
        policy.scaler_fitted = bool(state["scaler_fitted"])
        policy.total_steps = int(state["total_steps"])
        return policy
