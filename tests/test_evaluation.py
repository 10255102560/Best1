import unittest

import numpy as np

from harness_rsi.evaluation import compare_paired, evaluate_matrix


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.correct = np.asarray([[1, 0], [0, 1]], dtype=float)
        self.tokens = np.asarray([[100, 300], [200, 400]], dtype=float)
        self.calls = np.asarray([[1, 2], [1, 2]], dtype=float)
        self.latency = np.asarray([[0.5, 1.5], [0.8, 2.0]], dtype=float)

    def test_deterministic_metrics_and_cost_formula(self):
        metrics = evaluate_matrix(
            self.correct, self.tokens, self.calls, self.latency, [0, 1],
            cost_weight=0.2, token_unit=100,
        )
        self.assertEqual(metrics["accuracy"], 1)
        self.assertEqual(metrics["mean_tokens"], 250)
        self.assertEqual(metrics["mean_calls"], 1.5)
        self.assertEqual(metrics["mean_latency_s"], 1.25)
        self.assertAlmostEqual(metrics["utility"], 0.5)

    def test_probability_metrics_are_expectations(self):
        probs = np.asarray([[0.75, 0.25], [0.25, 0.75]])
        metrics = evaluate_matrix(
            self.correct, self.tokens, self.calls, self.latency, probs=probs
        )
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["mean_tokens"], 250)
        self.assertAlmostEqual(metrics["utility"], 0.7375)
        positional = evaluate_matrix(
            self.correct, self.tokens, self.calls, self.latency, probs
        )
        self.assertEqual(metrics, positional)

    def test_identical_policies_have_exact_zero_paired_intervals(self):
        correct = [1, 0, 1, 0, 0, 1]
        tokens = [12, 30, 24, 44, 20, 15]
        comparison = compare_paired(correct, tokens, correct, tokens)
        for metric in ("accuracy_difference", "token_difference", "utility_difference"):
            self.assertEqual(comparison[metric]["point"], 0)
            self.assertEqual(comparison[metric]["ci95"], [0, 0])

    def test_paired_differences_cost_sign_and_seed(self):
        parameters = dict(
            selected_correct=[1, 1, 1, 0], selected_tokens=[100, 100, 100, 100],
            baseline_correct=[1, 0, 1, 0], baseline_tokens=[200, 200, 200, 200],
            cost_weight=0.2, token_unit=100, seed=17, n_boot=1000,
        )
        comparison = compare_paired(**parameters)
        self.assertEqual(comparison, compare_paired(**parameters))
        self.assertEqual(comparison["accuracy_difference"]["point"], 0.25)
        self.assertEqual(comparison["token_difference"]["point"], -100)
        self.assertEqual(comparison["token_difference"]["ci95"], [-100, -100])
        self.assertAlmostEqual(comparison["utility_difference"]["point"], 0.45)

    def test_rejects_invalid_choices_and_unpaired_shapes(self):
        matrices = (self.correct, self.tokens, self.calls, self.latency)
        with self.assertRaises(ValueError):
            evaluate_matrix(*matrices, [0, 0.5])
        with self.assertRaises(ValueError):
            evaluate_matrix(*matrices, probs=[[0.5, 0.3], [0.5, 0.5]])
        with self.assertRaises(ValueError):
            evaluate_matrix(*matrices, [0, 1], probs=np.ones((2, 2)) / 2)
        with self.assertRaises(ValueError):
            compare_paired([1, 0], [100], [1, 0], [100, 200])


if __name__ == "__main__":
    unittest.main()
