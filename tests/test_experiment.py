import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from harness_rsi.experiment import run


class ExperimentTests(unittest.TestCase):
    def test_final_test_calls_require_frozen_policy_and_review_counts_both_steps(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, count in [('train', 12), ('test', 4)]:
                (root / f'{name}.jsonl').write_text('\n'.join(json.dumps({
                    'question': f'{name} item {i}: What is 2 plus 3?', 'answer': '#### 5'
                }) for i in range(count)), encoding='utf-8')
            (root / 'key.json').write_text(json.dumps({'base_url': 'https://example.invalid/v1',
                                                      'model': 'test', 'api_key': 'dummy'}))
            output = root / 'out'
            observed_tags = []

            class FakeClient:
                def __init__(self, base, model, key, cache, ledger, **kwargs):
                    self.requests_made = 0
                    self.ledger = ledger

                def complete(self, messages, tag=''):
                    if tag.startswith('test:'):
                        self_frozen = json.loads((output / 'frozen_policy.json').read_text())
                        assert self_frozen['policy']['version'] == 1
                        assert sum(t.startswith('train:') for t in observed_tags) == 24
                    self.requests_made += 1
                    observed_tags.append(tag)
                    result = {'text': '<answer>5</answer>', 'input_tokens': 30, 'output_tokens': 70,
                              'total_tokens': 100, 'reasoning_tokens': 0, 'latency_s': .1,
                              'cached': False, 'request_id': str(self.requests_made), 'finish_reason': 'stop'}
                    with self.ledger.open('a') as f:
                        f.write(json.dumps({'event': 'attempt'}) + '\n')
                        f.write(json.dumps({'event': 'success', **result}) + '\n')
                    return result

            args = argparse.Namespace(output=str(output), credentials=str(root / 'key.json'),
                                      base_url=None, model=None, key_env='NONEXISTENT_HARNESS_TEST_KEY',
                                      train_size=8, val_size=4, test_size=4, seed=42, rounds=2,
                                      steps=4, cost_weight=.05, max_requests=60, max_tokens=256,
                                      train_data=str(root / 'train.jsonl'), test_data=str(root / 'test.jsonl'),
                                      timeout=1, workers=1)
            with patch('harness_rsi.experiment.CachedChatClient', FakeClient), contextlib.redirect_stdout(io.StringIO()):
                run(args)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['ledger']['attempts'], 48)
            self.assertEqual(summary['metrics']['fixed_review']['mean_tokens'], 200)
            self.assertEqual(summary['metrics']['fixed_review']['mean_calls'], 2)
            self.assertEqual(summary['metrics']['fixed_reason']['mean_tokens'], 100)
            self.assertTrue((output / 'REPORT.md').exists())
            self.assertEqual(summary['config']['deployment_rule'], 'argmax')


if __name__ == '__main__':
    unittest.main()
