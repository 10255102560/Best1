import json
import unittest

import numpy as np

from harness_rsi.policy import (
    FEATURE_NAMES,
    LinearPolicy,
    extract_features,
    objective_and_gradient,
    softmax,
)


class PolicyTests(unittest.TestCase):
    def test_exact_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(41)
        x = rng.normal(size=(8, 4))
        rewards = rng.normal(size=(8, 3))
        weights = rng.normal(scale=0.3, size=(4, 3))
        bias = rng.normal(scale=0.3, size=3)
        options = {"entropy": 0.13, "l2": 0.07}
        _, grad_w, grad_b = objective_and_gradient(weights, bias, x, rewards, **options)
        epsilon = 1e-6
        for parameter, analytical in ((weights, grad_w), (bias, grad_b)):
            numerical = np.empty_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                upper = objective_and_gradient(weights, bias, x, rewards, **options)[0]
                parameter[index] = original - epsilon
                lower = objective_and_gradient(weights, bias, x, rewards, **options)[0]
                parameter[index] = original
                numerical[index] = (upper - lower) / (2 * epsilon)
            np.testing.assert_allclose(analytical, numerical, atol=2e-8, rtol=2e-6)

    def test_learns_context_dependent_routing_on_unseen_inputs(self):
        rng = np.random.default_rng(3)
        train_x = rng.uniform(-2, 2, size=(180, 2))
        # Action 0 wins to the left; action 1 wins to the right. Action 2 is weak.
        train_rewards = np.column_stack(
            [train_x[:, 0] < 0, train_x[:, 0] >= 0, np.full(len(train_x), 0.2)]
        ).astype(float)
        policy = LinearPolicy(2, 3, seed=7)
        history = policy.fit(train_x, train_rewards, steps=350, lr=0.2)
        test_x = rng.uniform(-2, 2, size=(300, 2))
        choices = policy.predict(test_x)
        correct_choices = (test_x[:, 0] >= 0).astype(int)
        self.assertGreater(np.mean(choices == correct_choices), 0.97)
        self.assertGreater(history[-1], history[0] + 0.3)

    def test_normalization_is_frozen_and_serialization_preserves_predictions(self):
        x = np.asarray([[-2.0, 10.0], [-1.0, 10.0], [1.0, 10.0], [2.0, 10.0]])
        rewards = np.asarray([[1.0, 0], [1.0, 0], [0, 1.0], [0, 1.0]])
        policy = LinearPolicy(2, 2)
        policy.fit(x, rewards, steps=20)
        original_mean = policy.mean.copy()
        original_scale = policy.scale.copy()
        policy.fit(x + 0.5, rewards, steps=20)
        np.testing.assert_array_equal(policy.mean, original_mean)
        np.testing.assert_array_equal(policy.scale, original_scale)
        restored = LinearPolicy.from_dict(json.loads(json.dumps(policy.to_dict())))
        np.testing.assert_array_equal(restored.probs(x), policy.probs(x))
        self.assertEqual(restored.total_steps, 40)

    def test_softmax_handles_large_logits(self):
        actual = softmax(np.asarray([[1000.0, 1001.0, -1000.0]]))
        np.testing.assert_allclose(actual.sum(axis=1), 1)
        self.assertTrue(np.isfinite(actual).all())
        self.assertEqual(int(np.argmax(actual[0])), 1)

    def test_features_are_fixed_finite_and_answer_independent(self):
        features = extract_features(
            "After paying $2.50 each for 3 books, Jo has 25% less money. How much is left?"
        )
        self.assertEqual(features.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features[FEATURE_NAMES.index("decimal_numbers")], 1)
        self.assertGreater(features[FEATURE_NAMES.index("percentage_words")], 0)
        self.assertTrue(np.isfinite(extract_features("")).all())

    def test_rejects_partial_reward_matrix(self):
        policy = LinearPolicy(2, 3)
        with self.assertRaises(ValueError):
            policy.fit(np.ones((2, 2)), np.asarray([[1, 0, np.nan], [1, 0, 1]]))


if __name__ == "__main__":
    unittest.main()
