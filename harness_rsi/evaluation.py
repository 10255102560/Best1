"""Matched-task metrics and uncertainty for harness routing experiments.

Bootstrap intervals describe sampling uncertainty across the evaluated tasks.
They do not account for LLM sampling variance or repeated model selection on the
same validation set; the final comparison must use a held-out test set.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _cost_parameters(cost_weight: float, token_unit: float) -> None:
    if not np.isfinite(cost_weight) or cost_weight < 0:
        raise ValueError("cost_weight must be finite and nonnegative")
    if not np.isfinite(token_unit) or token_unit <= 0:
        raise ValueError("token_unit must be finite and positive")


def _numeric_array(value: Any, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or array.size == 0:
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def evaluate_matrix(
    correct: Any,
    tokens: Any,
    calls: Any,
    latency: Any,
    actions: Any = None,
    *,
    probs: Any = None,
    cost_weight: float = 0.05,
    token_unit: float = 1000.0,
) -> dict[str, float]:
    """Evaluate per-task route choices against observed (tasks, actions) tables.

    Supply integer ``actions`` of shape (N,) for an actual deterministic policy.
    Supply ``probs`` of shape (N,A) for the expected performance of a randomized
    policy. A two-dimensional fifth positional argument is also interpreted as
    probabilities. Expected performance does not represent one realized rollout.
    Token input may already be a fixed, documented input/output weighted cost;
    this function applies no hidden token weighting or currency conversion.
    """
    _cost_parameters(cost_weight, token_unit)
    matrices = [
        _numeric_array(correct, "correct", 2),
        _numeric_array(tokens, "tokens", 2),
        _numeric_array(calls, "calls", 2),
        _numeric_array(latency, "latency", 2),
    ]
    shape = matrices[0].shape
    if any(matrix.shape != shape for matrix in matrices[1:]):
        raise ValueError("correct, tokens, calls, and latency must have the same shape")
    if np.any((matrices[0] < 0) | (matrices[0] > 1)):
        raise ValueError("correct values must lie between 0 and 1")
    if any(np.any(matrix < 0) for matrix in matrices[1:]):
        raise ValueError("tokens, calls, and latency must be nonnegative")
    if actions is not None and probs is not None:
        raise ValueError("provide either actions or probs, not both")
    if actions is None and probs is None:
        raise ValueError("actions or probs is required")
    if actions is not None and np.asarray(actions).ndim == 2:
        probs = actions
        actions = None

    if probs is not None:
        probability = _numeric_array(probs, "probs", 2)
        if probability.shape != shape:
            raise ValueError("probs must have the same (tasks, actions) shape as metrics")
        if np.any(probability < 0) or np.any(probability > 1):
            raise ValueError("probabilities must lie between 0 and 1")
        if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-8, rtol=0):
            raise ValueError("each probability row must sum to 1")
        means = [float(np.mean(np.sum(probability * matrix, axis=1))) for matrix in matrices]
    else:
        chosen = _numeric_array(actions, "actions", 1)
        if chosen.shape != (shape[0],):
            raise ValueError("actions must have one entry per task")
        if np.any(chosen != np.floor(chosen)) or np.any(chosen < 0) or np.any(chosen >= shape[1]):
            raise ValueError("actions must be valid integer action indices")
        chosen = chosen.astype(np.int64)
        row = np.arange(shape[0])
        means = [float(np.mean(matrix[row, chosen])) for matrix in matrices]
    accuracy, mean_tokens, mean_calls, mean_latency = means
    return {
        "accuracy": accuracy,
        "mean_tokens": mean_tokens,
        "mean_calls": mean_calls,
        "mean_latency_s": mean_latency,
        "utility": accuracy - cost_weight * mean_tokens / token_unit,
    }


def compare_paired(
    selected_correct: Any,
    selected_tokens: Any,
    baseline_correct: Any,
    baseline_tokens: Any,
    *,
    cost_weight: float = 0.05,
    token_unit: float = 1000.0,
    seed: int = 123,
    n_boot: int = 5000,
) -> dict[str, Any]:
    """Paired percentile bootstrap for selected minus baseline task metrics.

    Inputs are aligned vectors over the same tasks. Resampling uses the same
    task indices for both policies and all metrics. Positive accuracy/utility
    differences favor the selected policy; negative token differences indicate
    savings. Intervals are marginal 95% intervals, not simultaneous guarantees.
    """
    _cost_parameters(cost_weight, token_unit)
    arrays = [
        _numeric_array(selected_correct, "selected_correct", 1),
        _numeric_array(selected_tokens, "selected_tokens", 1),
        _numeric_array(baseline_correct, "baseline_correct", 1),
        _numeric_array(baseline_tokens, "baseline_tokens", 1),
    ]
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("all paired metric vectors must have the same shape")
    if any(np.any((arrays[index] < 0) | (arrays[index] > 1)) for index in (0, 2)):
        raise ValueError("correct values must lie between 0 and 1")
    if any(np.any(arrays[index] < 0) for index in (1, 3)):
        raise ValueError("token counts must be nonnegative")
    if isinstance(n_boot, bool) or int(n_boot) != n_boot or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    n_boot = int(n_boot)
    accuracy_difference = arrays[0] - arrays[2]
    token_difference = arrays[1] - arrays[3]
    differences = np.column_stack(
        [
            accuracy_difference,
            token_difference,
            accuracy_difference - cost_weight * token_difference / token_unit,
        ]
    )
    rng = np.random.default_rng(seed)
    samples = np.empty((n_boot, 3), dtype=np.float64)
    # Batching bounds memory even when this helper is used on a large benchmark.
    batch_size = max(1, min(256, 1_000_000 // len(differences)))
    for start in range(0, n_boot, batch_size):
        stop = min(start + batch_size, n_boot)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        samples[start:stop] = np.mean(differences[indices], axis=1)
    interval = np.quantile(samples, [0.025, 0.975], axis=0)
    points = differences.mean(axis=0)
    result: dict[str, Any] = {"n": len(differences), "n_boot": n_boot, "seed": int(seed)}
    for index, name in enumerate(("accuracy_difference", "token_difference", "utility_difference")):
        result[name] = {
            "point": float(points[index]),
            "ci95": [float(interval[0, index]), float(interval[1, index])],
        }
    return result
