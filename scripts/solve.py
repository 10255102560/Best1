"""Answer a new question with the frozen harness; never read benchmark labels."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from harness_rsi.client import CachedChatClient
from harness_rsi.policy import FEATURE_NAMES, LinearPolicy, extract_features
from harness_rsi.strategies import STRATEGIES, build_messages, build_review_messages


def normalize_base_url(value: str) -> str:
    value = value.rstrip("/")
    for suffix in ("/chat/completions", "/messages"):
        if value.endswith(suffix):
            value = value[:-len(suffix)]
    return value


def load_frozen(run: Path) -> tuple[dict, dict, LinearPolicy]:
    frozen_path = run / "frozen_policy.json"
    if not frozen_path.is_file():
        raise ValueError(
            f"Frozen policy not found: {frozen_path}. "
            "Finish the experiment's policy-freezing stage before inference."
        )
    config_path = run / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Frozen run is missing its config.json: {run}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    config_bytes = config_path.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != frozen.get("config_sha256"):
        raise ValueError("config.json does not match the frozen policy hash")
    config = json.loads(config_bytes)
    if config.get("features") != list(FEATURE_NAMES):
        raise ValueError("Frozen feature definitions do not match this implementation")
    if config.get("strategies") != list(STRATEGIES):
        raise ValueError("Frozen routes do not match this implementation")
    if config.get("deployment_rule") != "argmax" or config.get("temperature") != 0:
        raise ValueError("This inference script requires frozen argmax routing and temperature 0")
    hashes = config.get("implementation_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Frozen configuration is missing implementation hashes")
    for name, expected in hashes.items():
        if Path(name).name != name:
            raise ValueError("Invalid filename in the frozen implementation hashes")
        source = PROJECT_ROOT / "harness_rsi" / name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Implementation differs from the frozen experiment: {name}")
    policy = LinearPolicy.from_dict(frozen["policy"])
    if policy.n_features != len(FEATURE_NAMES) or policy.n_actions != len(STRATEGIES):
        raise ValueError("Frozen policy dimensions do not match the configured routes")
    return config, frozen, policy


@contextmanager
def ledger_lock(directory: Path):
    """Prevent concurrent processes from overspending the same persistent budget."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "inference.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Another inference process is using this run's request budget") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def solve(args: argparse.Namespace) -> dict:
    question = args.question.strip()
    if not question:
        raise ValueError("--question must contain a nonempty question")
    if args.max_requests < 0:
        raise ValueError("--max-requests must be nonnegative")
    run = Path(args.run).resolve()
    config, frozen, policy = load_frozen(run)
    credentials = {}
    if args.credentials:
        credentials = json.loads(Path(args.credentials).read_text(encoding="utf-8-sig"))
        if not isinstance(credentials, dict):
            raise ValueError("Credentials must be a JSON object")
    model = os.getenv("HARNESS_MODEL") or credentials.get("model") or config["model"]
    base_url = normalize_base_url(
        os.getenv("HARNESS_BASE_URL") or credentials.get("base_url") or config["base_url"]
    )
    if model != config["model"] or base_url != normalize_base_url(config["base_url"]):
        raise ValueError(
            "HARNESS_MODEL/HARNESS_BASE_URL or credential values differ from the frozen run. "
            "Use the model and endpoint recorded in its config.json."
        )
    key = os.getenv("HARNESS_API_KEY") or credentials.get("api_key")
    if not key:
        raise ValueError("Set HARNESS_API_KEY or supply --credentials with an api_key")

    features = np.asarray([extract_features(question)])
    route = STRATEGIES[int(policy.predict(features)[0])]
    directory = PROJECT_ROOT / "artifacts" / "inference" / run.name
    tag = "inference:" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    with ledger_lock(directory):
        client = CachedChatClient(
            base_url, model, key, directory / "cache", directory / "requests.jsonl",
            max_requests=args.max_requests, max_tokens=config["max_tokens"],
            timeout=args.timeout, max_parallel=1,
        )
        before = client.requests_made
        first_route = "reason" if route == "review" else route
        steps = [client.complete(build_messages(question, first_route), tag=f"{tag}:{first_route}")]
        if route == "review":
            steps.append(client.complete(
                build_review_messages(question, steps[0]["text"]), tag=f"{tag}:review"
            ))
        return {
            "route": route,
            "text": steps[-1]["text"],
            "input_tokens": sum(step["input_tokens"] for step in steps),
            "output_tokens": sum(step["output_tokens"] for step in steps),
            "total_tokens": sum(step["total_tokens"] for step in steps),
            "reasoning_tokens": sum(step["reasoning_tokens"] for step in steps),
            "calls": len(steps),
            "all_cached": all(step["cached"] for step in steps),
            "new_request_attempts": client.requests_made - before,
            "total_request_attempts": client.requests_made,
            "request_limit": args.max_requests,
            "new_successful_request_tokens": sum(
                step["total_tokens"] for step in steps if not step["cached"]
            ),
            "finish_reason": steps[-1]["finish_reason"],
            "selected_version": frozen["selected_version"],
            "ledger": str(directory / "requests.jsonl"),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="New arithmetic question; no reference answer is used")
    parser.add_argument("--run", default="artifacts/gsm8k_flash_v1", help="Directory containing the frozen policy")
    parser.add_argument("--credentials", help="Optional local JSON with api_key, base_url and model")
    parser.add_argument("--max-requests", type=int, default=2,
                        help="Lifetime attempt cap for this inference ledger, including retries (default: 2)")
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    try:
        result = solve(args)
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
