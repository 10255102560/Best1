"""Verify saved benchmark results and regenerate reports, entirely offline.

Usage from any working directory:
    python /path/to/repo/scripts/replay_report.py
    python /path/to/repo/scripts/replay_report.py --run /path/to/artifacts/run

Relative --run paths are resolved against this repository, not the shell's cwd.
No credentials are read and no model requests are made. This verifies saved
observations and computation; it does not independently rerun the language model.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from harness_rsi.evaluation import compare_paired, evaluate_matrix
from harness_rsi.policy import FEATURE_NAMES, LinearPolicy, extract_features
from harness_rsi.report import write_report
from harness_rsi.strategies import STRATEGIES
from harness_rsi.tasks import extract_answer, is_correct


def _load(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"Run incomplete or file missing: {path}. No report regenerated.")
    return json.loads(path.read_text(encoding="utf-8"))


def _same(actual: Any, expected: Any, location: str) -> None:
    """Recursive assertion with tolerance only for finite floating-point values."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"{location}: dictionary keys differ")
        for key in expected:
            _same(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{location}: sequence lengths differ")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _same(left, right, f"{location}[{index}]")
    elif isinstance(expected, float):
        if not np.isfinite(expected) or not np.isfinite(actual) or not np.isclose(actual, expected, rtol=1e-10, atol=1e-10):
            raise AssertionError(f"{location}: expected {expected!r}, reproduced {actual!r}")
    elif actual != expected:
        raise AssertionError(f"{location}: expected {expected!r}, reproduced {actual!r}")


def _matrices(rows: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray([[row["outcomes"][action][field] for action in STRATEGIES] for row in rows], dtype=float)
        for field in ("correct", "total_tokens", "calls", "latency_s")
    )


def replay(run_dir: str | Path) -> dict[str, str]:
    directory = Path(run_dir)
    if not directory.is_absolute():
        directory = REPO_ROOT / directory
    directory = directory.resolve()
    # Require completion before inspecting any inputs or overwriting a report.
    summary = _load(directory / "summary.json")
    config_path = directory / "config.json"
    config = _load(config_path)
    manifest = _load(directory / "split_manifest.json")
    freeze = _load(directory / "frozen_policy.json")
    _same(summary["config"], config, "summary.config")
    _same(config["strategies"], list(STRATEGIES), "config.strategies")
    _same(config["features"], list(FEATURE_NAMES), "config.features")
    _same(hashlib.sha256(config_path.read_bytes()).hexdigest(), freeze["config_sha256"], "frozen.config_sha256")
    for name, digest in config.get("implementation_sha256", {}).items():
        if Path(name).name != name:
            raise ValueError("Invalid implementation file name in config")
        source = REPO_ROOT / "harness_rsi" / name
        _same(hashlib.sha256(source.read_bytes()).hexdigest(), digest, f"implementation.{name}")

    ledger_path = directory / "requests.jsonl"
    events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    successes = [event for event in events if event["event"] == "success"]
    by_request = {event["request_id"]: event for event in successes}
    if len(by_request) != len(successes):
        raise AssertionError("Duplicate successful request IDs in ledger")
    ledger = {
        "attempts": sum(event["event"] == "attempt" for event in events),
        "successes": len(successes),
        "failures": sum(event["event"] == "failure" for event in events),
        **{key: sum(event[key] for event in successes) for key in ("input_tokens", "output_tokens", "total_tokens")},
        "failed_request_cost": "unknown; any provider-billed failed requests are not measurable here",
    }
    _same(ledger, summary["ledger"], "ledger")
    if ledger["attempts"] > config["max_requests"]:
        raise AssertionError("Recorded request count exceeds configured cap")
    frozen_at = datetime.fromisoformat(freeze["frozen_at"])
    for event in events:
        if event.get("tag", "").startswith("test:") and "timestamp" in event:
            if datetime.fromisoformat(event["timestamp"]) < frozen_at:
                raise AssertionError("A test request predates policy freezing")

    rows_by_split = {}
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for split in ("train", "val", "test"):
        tasks = manifest[split]
        _same(len(tasks), config[f"{split}_size"], f"manifest.{split}.size")
        rows = []
        for task in tasks:
            task_id = task["id"]
            if Path(task_id).name != task_id:
                raise ValueError("Invalid task ID in manifest")
            if task_id in seen_ids or task["question"] in seen_questions:
                raise AssertionError("Duplicate task ID or question across manifest splits")
            seen_ids.add(task_id)
            seen_questions.add(task["question"])
            row = _load(directory / "observations" / split / f"{task_id}.json")
            _same(row["task"], task, f"{split}.{task_id}.task")
            _same(row["split"], split, f"{split}.{task_id}.split")
            for action in STRATEGIES:
                outcome = row["outcomes"][action]
                location = f"{split}.{task_id}.{action}"
                prediction = extract_answer(outcome["text"])
                _same(str(prediction) if prediction is not None else None, outcome["prediction"], location + ".prediction")
                _same(int(is_correct(outcome["text"], task["answer"])), outcome["correct"], location + ".correct")
                request_ids = outcome["request_ids"]
                _same(len(request_ids), 2 if action == "review" else 1, location + ".calls")
                _same(outcome["calls"], len(request_ids), location + ".calls.recorded")
                if action == "review":
                    _same(request_ids[0], row["outcomes"]["reason"]["request_ids"][0], location + ".shared_draft")
                for field in ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens", "latency_s"):
                    _same(sum(by_request[request_id][field] for request_id in request_ids), outcome[field], location + "." + field)
            rows.append(row)
        rows_by_split[split] = rows

    policy = LinearPolicy.from_dict(freeze["policy"])
    test = rows_by_split["test"]
    matrix = _matrices(test)
    features = np.asarray([extract_features(row["task"]["question"]) for row in test])
    chosen = policy.predict(features)
    n = len(test)
    best_fixed = STRATEGIES.index(freeze["best_fixed"])
    options = {"cost_weight": config["cost_weight"], "token_unit": config["token_unit"]}
    actions = {
        "initial_reason": np.ones(n, dtype=int),
        **{f"fixed_{name}": np.full(n, index) for index, name in enumerate(STRATEGIES)},
        "best_fixed": np.full(n, best_fixed),
        "learned": chosen,
        "uniform": np.full((n, len(STRATEGIES)), 1 / len(STRATEGIES)),
        "oracle": np.argmax(matrix[0] - config["cost_weight"] * matrix[1] / config["token_unit"], axis=1),
    }
    measured = {name: evaluate_matrix(*matrix, actions=selection, **options) for name, selection in actions.items()}
    _same(measured, summary["metrics"], "metrics")
    comparisons = {}
    indices = np.arange(n)
    for label, baseline in (("vs_initial", 1), ("vs_best_fixed", best_fixed)):
        comparisons[label] = compare_paired(
            matrix[0][indices, chosen], matrix[1][indices, chosen],
            matrix[0][:, baseline], matrix[1][:, baseline], seed=config["seed"], **options,
        )
    _same(comparisons, summary["comparisons"], "comparisons")
    _same({name: int(np.sum(chosen == index)) for index, name in enumerate(STRATEGIES)}, summary["selected_actions"], "selected_actions")
    _same({
        "fixes": int(np.sum((matrix[0][:, 1] == 0) & (matrix[0][:, 2] == 1))),
        "breaks": int(np.sum((matrix[0][:, 1] == 1) & (matrix[0][:, 2] == 0))),
    }, summary["review_effect"], "review_effect")
    for output_key, field, value in (("format_failures", "prediction", None), ("truncated_outputs", "finish_reason", "length")):
        reproduced = {name: sum(row["outcomes"][name][field] == value for row in test) for name in STRATEGIES}
        _same(reproduced, summary[output_key], output_key)
    predictions = [{"task_id": row["task"]["id"], "action": STRATEGIES[int(action)], **row["outcomes"][STRATEGIES[int(action)]]} for row, action in zip(test, chosen)]
    _same(predictions, _load(directory / "test_predictions.json"), "test_predictions")

    validation = rows_by_split["val"]
    validation_matrix = _matrices(validation)
    fixed_metrics = [evaluate_matrix(*validation_matrix, actions=np.full(len(validation), index), **options) for index in range(len(STRATEGIES))]
    _same(int(np.argmax([item["utility"] for item in fixed_metrics])), best_fixed, "validation.best_fixed")
    _same(fixed_metrics[best_fixed], summary["validation_best_fixed_metrics"], "validation_best_fixed_metrics")
    _same(freeze["best_fixed"], summary["best_fixed_strategy"], "best_fixed_strategy")
    _same(freeze["selected_version"], summary["selected_version"], "selected_version")
    _same(_load(directory / "rounds.json"), summary["rounds"], "rounds")

    # Recompute the selected candidate's validation gate at every saved round.
    vx = np.asarray([extract_features(row["task"]["question"]) for row in validation])
    incumbent = fixed_metrics[1]
    initial_accuracy = incumbent["accuracy"]
    selected_version = 0
    selected_policy = LinearPolicy(len(FEATURE_NAMES), len(STRATEGIES), seed=config["seed"])
    selected_policy.weights[:] = 0
    selected_policy.bias[1] = 0.5
    for entry in summary["rounds"]:
        candidate_state = _load(directory / "versions" / f"candidate_{entry['round']}.json")
        candidate = LinearPolicy.from_dict(candidate_state)
        candidate_metrics = evaluate_matrix(*validation_matrix, actions=candidate.predict(vx), **options)
        _same(incumbent, entry["val_incumbent"], f"round.{entry['round']}.incumbent")
        _same(candidate_metrics, entry["val_candidate"], f"round.{entry['round']}.candidate")
        accepted = bool(
            candidate_metrics["utility"] > incumbent["utility"] + config["gate_utility_margin"]
            and candidate_metrics["accuracy"] >= max(initial_accuracy, incumbent["accuracy"]) - config["gate_accuracy_tolerance"]
        )
        _same(accepted, entry["accepted"], f"round.{entry['round']}.accepted")
        if accepted:
            selected_policy = candidate
            selected_version = entry["round"]
            incumbent = candidate_metrics
    _same(selected_version, freeze["selected_version"], "gate.selected_version")
    _same(selected_policy.to_dict(), freeze["policy"], "gate.frozen_policy")
    paths = write_report(summary, directory)
    print(json.dumps({
        "status": "verified_offline", "tasks_verified": len(seen_ids), "test_tasks": n,
        "successful_requests_verified": len(successes), "selected_version": selected_version,
        "verified": ["implementation hashes", "split membership", "answer grading", "request token accounting", "frozen policy", "validation gates", "test metrics", "paired bootstrap", "prediction export"],
        **paths,
    }, ensure_ascii=False, indent=2))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="artifacts/gsm8k_flash_v1", help="Saved run directory; relative paths are repository-relative")
    args = parser.parse_args()
    try:
        replay(args.run)
    except (AssertionError, ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Offline verification failed: {error}\n")


if __name__ == "__main__":
    main()
