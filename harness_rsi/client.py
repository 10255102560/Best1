"""Cached, budgeted chat calls using the operating system's TLS-enabled curl.

The bearer token lives only in a short-lived header file, never in command-line
arguments, cache keys, cached responses, or the persistent request ledger.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from typing import Any
from urllib.parse import urlsplit
import uuid


class BudgetExceeded(RuntimeError):
    """No additional paid request may be attempted under this run's cap."""


class ProviderError(RuntimeError):
    """A provider response could not be used as a metered chat result."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProviderError(f"Provider returned invalid or missing {name}")
    return value


class CachedChatClient:
    """Thread-safe within one process; use a distinct ledger for other processes.

    ``max_requests`` includes all prior attempts in this ledger and all failed
    attempts.  Cache hits consume no request budget.  A request gets at most
    two retries, and only for network, HTTP 429, or HTTP 5xx failures.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        cache_dir: str | Path,
        ledger_path: str | Path,
        max_requests: int = 950,
        max_tokens: int = 2048,
        timeout: float = 90,
        max_parallel: int = 6,
        min_interval: float = 0.8,
        rate_limit_backoff: float = 25,
    ):
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("https", "http") or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain credentials, query parameters, or a fragment")
        if not api_key or any(character in api_key for character in "\r\n"):
            raise ValueError("api_key must be nonempty and must not contain newlines")
        for name, value, minimum in (("max_requests", max_requests, 0),
                                     ("max_tokens", max_tokens, 1),
                                     ("max_parallel", max_parallel, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be a finite positive number")
        for name, value in (("min_interval", min_interval), ("rate_limit_backoff", rate_limit_backoff)):
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite nonnegative number")
        if not model:
            raise ValueError("model must not be empty")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.ledger_path = Path(ledger_path)
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.min_interval = min_interval
        self.rate_limit_backoff = rate_limit_backoff
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(max_parallel)
        self._schedule = threading.Condition()
        self._next_start = 0.0
        self._backoff_until = 0.0
        self._cache_locks: dict[str, threading.Lock] = {}
        self._requests_made = self._read_attempt_count()

    @property
    def requests_made(self) -> int:
        with self._lock:
            return self._requests_made

    @property
    def request_count(self) -> int:
        return self.requests_made

    def _read_attempt_count(self) -> int:
        if not self.ledger_path.exists():
            return 0
        attempts = 0
        with self.ledger_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed request ledger at line {line_number}; refusing to reset budget") from exc
                if not isinstance(row, dict) or row.get("event") not in ("attempt", "success", "failure"):
                    raise ValueError(f"Unknown request ledger event at line {line_number}; refusing to reset budget")
                attempts += row["event"] == "attempt"
        return attempts

    def _append_locked(self, event: dict[str, Any]) -> None:
        # Flush before the HTTP call: a crash must not erase attempted spending.
        with self.ledger_path.open("a", encoding="utf-8", newline="\n") as target:
            target.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            target.flush()
            os.fsync(target.fileno())

    def _record(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._append_locked({"timestamp": _utc_now(), **event})

    def _reserve(self, cache_key: str, tag: str, attempt: int) -> str:
        with self._lock:
            if self._requests_made >= self.max_requests:
                raise BudgetExceeded(
                    f"Request budget exhausted: {self._requests_made}/{self.max_requests} attempts "
                    "already recorded (including failures)"
                )
            request_id = uuid.uuid4().hex
            self._append_locked({
                "event": "attempt", "timestamp": _utc_now(), "request_id": request_id,
                "cache_key": cache_key, "tag": tag, "attempt": attempt,
                "model": self.model, "max_tokens": self.max_tokens,
            })
            self._requests_made += 1
            return request_id

    def _wait_for_start(self) -> None:
        """Pace all threads and respect provider cooldowns before spending budget."""
        with self._schedule:
            while True:
                now = time.monotonic()
                delay = max(self._next_start, self._backoff_until) - now
                if delay <= 0:
                    self._next_start = now + self.min_interval
                    return
                # Condition.wait releases the lock so another thread can extend
                # a shared 429 cooldown while this thread is waiting.
                self._schedule.wait(timeout=delay)

    def _defer_requests(self) -> None:
        with self._schedule:
            self._backoff_until = max(self._backoff_until, time.monotonic() + self.rate_limit_backoff)
            self._schedule.notify_all()

    def _curl_request(self, payload: dict[str, Any]) -> tuple[int, str, float]:
        """Return (HTTP status, response body, elapsed seconds); mockable in tests."""
        executable = "curl.exe" if os.name == "nt" else "curl"
        marker = "\n__HARNESS_RSI_HTTP_STATUS__"
        with tempfile.TemporaryDirectory(prefix="harness-rsi-auth-") as directory:
            header_path = Path(directory) / "headers.txt"
            # A newly created file has mode 0600 on POSIX.  TemporaryDirectory
            # uses the current user's temporary directory on Windows.
            descriptor = os.open(header_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as headers:
                headers.write(f"Authorization: Bearer {self._api_key}\n")
                headers.write("Content-Type: application/json\n")
            command = [
                executable, "--silent", "--show-error", "--request", "POST",
                "--url", self.base_url + "/chat/completions",
                "--header", "@" + str(header_path), "--data-binary", "@-",
                "--max-time", str(self.timeout), "--write-out", marker + "%{http_code}",
            ]
            started = time.perf_counter()
            try:
                process = subprocess.run(
                    command,
                    input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    capture_output=True,
                    timeout=self.timeout + 5,
                    check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            except subprocess.TimeoutExpired as exc:
                raise ProviderError("curl request timed out", retryable=True) from exc
            except FileNotFoundError as exc:
                raise ProviderError("curl executable was not found") from exc
            elapsed = time.perf_counter() - started
            if process.returncode != 0:
                # Do not copy subprocess stderr or provider text into exceptions:
                # neither is required for diagnostics and either may echo data.
                raise ProviderError(f"curl transport failed with exit code {process.returncode}", retryable=True)
            body, separator, status = process.stdout.decode("utf-8", errors="replace").rpartition(marker)
            if not separator or not status.strip().isdigit():
                raise ProviderError("curl did not return an HTTP status", retryable=True)
            return int(status.strip()), body, elapsed

    @staticmethod
    def _decode_response(status: int, body: str, latency_s: float, request_id: str) -> dict[str, Any]:
        if status < 200 or status >= 300:
            raise ProviderError(f"Provider returned HTTP {status}", status=status,
                                retryable=status == 429 or 500 <= status < 600)
        if body.lstrip().startswith("<"):
            raise ProviderError("Provider returned HTML/XML instead of a JSON chat response", status=status)
        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ProviderError("Provider returned malformed JSON", status=status) from exc
        if not isinstance(result, dict) or "error" in result:
            raise ProviderError("Provider returned an error or non-object JSON response", status=status)
        try:
            choice = result["choices"][0]
            text = choice["message"]["content"]
            usage = result["usage"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Provider response is missing choices, content, or metered usage", status=status) from exc
        # A reasoning model can spend its completion allowance before emitting
        # visible text. It is still a charged result and must score as an empty
        # answer, with its real usage retained rather than silently reissued.
        if text is None and choice.get("finish_reason") == "length":
            text = ""
        if not isinstance(text, str) or not isinstance(usage, dict):
            raise ProviderError("Provider response has invalid content or metered usage", status=status)
        input_tokens = _nonnegative_int(usage.get("prompt_tokens"), "prompt_tokens")
        output_tokens = _nonnegative_int(usage.get("completion_tokens"), "completion_tokens")
        total_tokens = _nonnegative_int(usage.get("total_tokens", input_tokens + output_tokens), "total_tokens")
        if total_tokens == 0:
            raise ProviderError("Provider reported zero metered usage for a chat request", status=status)
        if total_tokens < input_tokens + output_tokens:
            raise ProviderError("Provider total_tokens is smaller than input plus output usage", status=status)
        details = usage.get("completion_tokens_details") or {}
        if not isinstance(details, dict):
            raise ProviderError("Provider completion_tokens_details is invalid", status=status)
        reasoning_tokens = _nonnegative_int(details.get("reasoning_tokens", 0), "reasoning_tokens")
        if reasoning_tokens > output_tokens:
            raise ProviderError("Provider reasoning_tokens exceeds completion_tokens", status=status)
        return {
            "text": text, "input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": total_tokens, "reasoning_tokens": reasoning_tokens,
            "latency_s": latency_s, "cached": False, "request_id": request_id,
            "finish_reason": choice.get("finish_reason"),
        }

    def complete(self, messages: list[dict[str, str]], tag: str = "") -> dict[str, Any]:
        payload = {"model": self.model, "messages": messages,
                   "max_tokens": self.max_tokens, "temperature": 0}
        cache_identity = {"base_url": self.base_url, **payload}
        cache_key = hashlib.sha256(json.dumps(
            cache_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / (cache_key + ".json")
        with self._lock:
            cache_lock = self._cache_locks.setdefault(cache_key, threading.Lock())
        # Prevent simultaneous identical prompts from creating duplicate charges.
        with cache_lock:
            if cache_path.exists():
                try:
                    saved = json.loads(cache_path.read_text(encoding="utf-8"))
                    response = saved["response"]
                    required = {"text", "input_tokens", "output_tokens", "total_tokens", "reasoning_tokens",
                                "latency_s", "request_id", "finish_reason"}
                    if saved["cache_key"] != cache_key or not required.issubset(response):
                        raise ValueError("Invalid cache schema")
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid cache file {cache_path.name}; refusing an automatic paid replacement") from exc
                return {**response, "cached": True}
            with self._semaphore:
                for attempt in range(1, 4):
                    self._wait_for_start()
                    request_id = self._reserve(cache_key, tag, attempt)
                    try:
                        status, body, elapsed = self._curl_request(payload)
                        response = self._decode_response(status, body, elapsed, request_id)
                    except ProviderError as exc:
                        self._record({"event": "failure", "request_id": request_id,
                                      "tag": tag, "attempt": attempt, "error": str(exc),
                                      "http_status": exc.status, "retryable": exc.retryable,
                                      "usage_unknown": True})
                        if exc.status == 429:
                            self._defer_requests()
                        if not exc.retryable or attempt == 3:
                            raise
                        if exc.status != 429:
                            time.sleep(min(2 ** (attempt - 1), 4))
                        continue
                    self._record({"event": "success", "request_id": request_id,
                                  "tag": tag, "attempt": attempt,
                                  **{key: value for key, value in response.items()
                                     if key not in ("text", "cached", "request_id")}})
                    temporary_path = cache_path.with_suffix("." + uuid.uuid4().hex + ".tmp")
                    try:
                        temporary_path.write_text(json.dumps(
                            {"cache_key": cache_key, "response": response},
                            ensure_ascii=False, sort_keys=True,
                        ), encoding="utf-8")
                        os.replace(temporary_path, cache_path)
                    finally:
                        temporary_path.unlink(missing_ok=True)
                    return response
        raise AssertionError("Unreachable retry state")
