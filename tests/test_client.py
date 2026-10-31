from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from harness_rsi.client import BudgetExceeded, CachedChatClient, ProviderError


MESSAGES = [{"role": "user", "content": "What is 2 plus 2?"}]


def valid_response(text="<answer>4</answer>"):
    return 200, json.dumps({
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20,
                  "completion_tokens_details": {"reasoning_tokens": 5}},
    }), 0.25


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def make_client(self, max_requests=10):
        return CachedChatClient(
            "https://example.invalid/v1", "test-model", "test-secret-key",
            self.root / "cache", self.root / "ledger.jsonl", max_requests=max_requests,
            min_interval=0, rate_limit_backoff=0,
        )

    def read_ledger(self):
        return [json.loads(line) for line in (self.root / "ledger.jsonl").read_text().splitlines()]

    def test_cache_hit_is_free_and_budget_survives_restart(self):
        client = self.make_client(max_requests=1)
        with patch.object(client, "_curl_request", return_value=valid_response()) as request:
            first = client.complete(MESSAGES, tag="train/direct")
            second = client.complete(MESSAGES, tag="test/direct")
            self.assertEqual(request.call_count, 1)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(first["output_tokens"], 8)
        self.assertEqual(first["reasoning_tokens"], 5)
        self.assertEqual(client.requests_made, 1)
        restarted = self.make_client(max_requests=1)
        self.assertTrue(restarted.complete(MESSAGES)["cached"])
        with self.assertRaises(BudgetExceeded):
            restarted.complete([{"role": "user", "content": "Another question"}])
        self.assertEqual([row["event"] for row in self.read_ledger()], ["attempt", "success"])
        self.assertNotIn("test-secret-key", (self.root / "ledger.jsonl").read_text())

    def test_concurrent_budget_cannot_be_overspent(self):
        client = self.make_client(max_requests=2)
        def invoke(index):
            try:
                return client.complete([{"role": "user", "content": str(index)}])
            except BudgetExceeded:
                return None
        with patch.object(client, "_curl_request", return_value=valid_response()) as request:
            with ThreadPoolExecutor(max_workers=6) as executor:
                results = list(executor.map(invoke, range(6)))
        self.assertEqual(sum(result is not None for result in results), 2)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(client.requests_made, 2)

    def test_concurrent_identical_prompts_share_one_call(self):
        client = self.make_client()
        with patch.object(client, "_curl_request", return_value=valid_response()) as request:
            with ThreadPoolExecutor(max_workers=6) as executor:
                results = list(executor.map(lambda _: client.complete(MESSAGES), range(6)))
        self.assertEqual(request.call_count, 1)
        self.assertEqual(sum(result["cached"] for result in results), 5)

    def test_missing_usage_is_a_paid_failure_not_zero_cost_success(self):
        client = self.make_client()
        body = json.dumps({"choices": [{"message": {"content": "<answer>4</answer>"}}]})
        with patch.object(client, "_curl_request", return_value=(200, body, 0.1)) as request:
            with self.assertRaises(ProviderError):
                client.complete(MESSAGES)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(client.requests_made, 1)
        self.assertEqual([row["event"] for row in self.read_ledger()], ["attempt", "failure"])
        self.assertEqual(list((self.root / "cache").glob("*.json")), [])

    def test_only_transient_errors_are_retried_and_each_counts(self):
        client = self.make_client(max_requests=3)
        with patch.object(client, "_curl_request", side_effect=[
            (429, "rate limited", 0.1), (503, "unavailable", 0.1), valid_response(),
        ]) as request, patch("harness_rsi.client.time.sleep"):
            result = client.complete(MESSAGES)
        self.assertEqual(result["text"], "<answer>4</answer>")
        self.assertEqual(request.call_count, 3)
        self.assertEqual(client.requests_made, 3)
        self.assertEqual(sum(row["event"] == "failure" for row in self.read_ledger()), 2)

    def test_retry_obeys_budget(self):
        client = self.make_client(max_requests=1)
        with patch.object(client, "_curl_request", return_value=(500, "error", 0.1)) as request, \
                patch("harness_rsi.client.time.sleep"):
            with self.assertRaises(BudgetExceeded):
                client.complete(MESSAGES)
        self.assertEqual(request.call_count, 1)

    def test_html_and_auth_failure_are_not_retried(self):
        client = self.make_client()
        for index, response in enumerate(((200, "<html>error</html>", 0.1), (401, "bad key", 0.1))):
            with patch.object(client, "_curl_request", return_value=response) as request:
                with self.assertRaises(ProviderError):
                    client.complete([{"role": "user", "content": str(index)}])
                self.assertEqual(request.call_count, 1)

    def test_zero_metered_usage_is_not_accepted_as_free_success(self):
        client = self.make_client()
        body = json.dumps({
            "choices": [{"message": {"content": "<answer>4</answer>"}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })
        with patch.object(client, "_curl_request", return_value=(200, body, 0.1)):
            with self.assertRaises(ProviderError):
                client.complete(MESSAGES)
        self.assertEqual(client.requests_made, 1)
        self.assertEqual(self.read_ledger()[-1]["event"], "failure")

    def test_null_content_at_length_limit_retains_cost_and_is_not_retried(self):
        client = self.make_client()
        body = json.loads(valid_response()[1])
        body["choices"][0].update(message={"content": None}, finish_reason="length")
        with patch.object(client, "_curl_request", return_value=(200, json.dumps(body), 0.1)) as request:
            response = client.complete(MESSAGES)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(response["text"], "")
        self.assertEqual(response["output_tokens"], 8)
        self.assertEqual(response["reasoning_tokens"], 5)
        self.assertEqual(response["finish_reason"], "length")
        self.assertEqual(self.read_ledger()[-1]["event"], "success")
        self.assertTrue(client.complete(MESSAGES)["cached"])

    def test_null_content_without_length_limit_remains_an_error(self):
        client = self.make_client()
        body = json.loads(valid_response()[1])
        body["choices"][0]["message"]["content"] = None
        with patch.object(client, "_curl_request", return_value=(200, json.dumps(body), 0.1)):
            with self.assertRaises(ProviderError):
                client.complete(MESSAGES)

    def test_request_start_pacing_is_shared_by_threads(self):
        client = self.make_client()
        client.min_interval = 0.015
        starts = []
        def request(payload):
            starts.append(time.monotonic())
            return valid_response()
        with patch.object(client, "_curl_request", side_effect=request):
            with ThreadPoolExecutor(max_workers=3) as executor:
                list(executor.map(lambda index: client.complete([
                    {"role": "user", "content": str(index)}]), range(3)))
        gaps = [right - left for left, right in zip(starts, starts[1:])]
        self.assertTrue(all(gap >= 0.012 for gap in gaps), gaps)

    def test_429_cooldown_defers_subsequent_requests_globally(self):
        client = self.make_client()
        client.rate_limit_backoff = 0.025
        with patch.object(client, "_curl_request", side_effect=[
            (429, "rate limited", 0.1), valid_response(),
        ]) as request:
            start = time.monotonic()
            client.complete(MESSAGES)
            elapsed = time.monotonic() - start
        self.assertEqual(request.call_count, 2)
        self.assertGreaterEqual(elapsed, 0.023)

    def test_auth_header_is_temporary_and_not_in_command_line(self):
        client = self.make_client()
        def fake_run(command, **kwargs):
            self.assertNotIn("test-secret-key", " ".join(command))
            self.assertNotIn("--insecure", command)
            header_path = Path(command[command.index("--header") + 1][1:])
            self.assertIn("Authorization: Bearer test-secret-key", header_path.read_text())
            self.header_path = header_path
            self.assertEqual(json.loads(kwargs["input"])["temperature"], 0)
            body = valid_response()[1] + "\n__HARNESS_RSI_HTTP_STATUS__200"
            class Result:
                returncode = 0
                stdout = body.encode()
            return Result()
        with patch("harness_rsi.client.subprocess.run", side_effect=fake_run):
            client.complete(MESSAGES)
        self.assertFalse(self.header_path.exists())


if __name__ == "__main__":
    unittest.main()
