import json
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from harness_rsi.tasks import (
    Task, build_demo_splits, extract_answer, is_correct, load_gsm8k,
    parse_number, question_hash,
)
from harness_rsi.strategies import STRATEGIES, build_messages, build_review_messages


class AnswerTests(unittest.TestCase):
    def test_numeric_formats_and_exactness(self):
        self.assertEqual(extract_answer("work\n#### 1,234.50"), Decimal("1234.5"))
        self.assertEqual(extract_answer("<answer>-0.25</answer>"), Decimal("-0.25"))
        self.assertTrue(is_correct("<answer>4.0</answer>", "4"))
        self.assertFalse(is_correct("<answer>4.000000000000000001</answer>", "4"))
        self.assertIsNone(parse_number("1,2"))
        self.assertIsNone(parse_number("NaN"))

    def test_reasoning_numbers_are_not_answers(self):
        self.assertIsNone(extract_answer("We computed 4 times 6 and got 24."))
        self.assertIsNone(extract_answer("The answer is 24"))
        self.assertIsNone(extract_answer("<answer>24 dollars</answer>"))
        self.assertIsNone(extract_answer("<answer>1/2</answer>"))

    def test_last_explicit_answer_wins(self):
        self.assertEqual(extract_answer("<answer>4</answer>\n#### 5"), Decimal(5))
        self.assertIsNone(extract_answer("<answer>4</answer><answer>unknown</answer>"))
        self.assertIsNone(extract_answer("<answer>4</answer><answer>"))


class DatasetTests(unittest.TestCase):
    def test_gsm8k_official_boundary_dedup_and_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            train_path, test_path = Path(directory) / "train.jsonl", Path(directory) / "test.jsonl"
            train_rows = [{"question": f"Train question {i}?", "answer": f"work\n#### {i}"} for i in range(8)]
            train_rows += [train_rows[0], {"question": " SHARED question ", "answer": "#### 1"}]
            test_rows = [{"question": "Shared QUESTION", "answer": "#### 1"}]
            test_rows += [{"question": f"Test question {i}?", "answer": f"#### {i}"} for i in range(4)]
            for path, rows in ((train_path, train_rows), (test_path, test_rows)):
                path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            splits = load_gsm8k(train_path, test_path, 4, 4, 5, seed=17)
            self.assertEqual(splits, load_gsm8k(train_path, test_path, 4, 4, 5, seed=17))
            hashes = [question_hash(task.question) for tasks in splits.values() for task in tasks]
            self.assertEqual(len(hashes), len(set(hashes)))
            self.assertTrue(all(task.source_split == "gsm8k_test" for task in splits["test"]))
            with self.assertRaises(ValueError):
                load_gsm8k(train_path, test_path, 5, 4, 5)

    def test_demo_is_disjoint_reproducible_and_labeled(self):
        splits = build_demo_splits(30, 15, 20, seed=91)
        self.assertEqual(splits, build_demo_splits(30, 15, 20, seed=91))
        tasks = [task for rows in splits.values() for task in rows]
        self.assertEqual(len({task.id for task in tasks}), 65)
        for task in tasks:
            self.assertTrue(task.id.startswith("demo-"))
            self.assertTrue(task.source_split.startswith("demo_"))
            self.assertTrue(is_correct(f"<answer>{task.answer}</answer>", task))
            self.assertEqual(Task.from_dict(task.to_dict()), task)

    def test_strategies_are_bounded_and_do_not_need_labels(self):
        for strategy in STRATEGIES:
            messages = build_messages("How much is 2 plus 3?", strategy)
            self.assertEqual(messages[1]["content"], "How much is 2 plus 3?")
            self.assertIn("<answer>number</answer>", messages[0]["content"])
        review = build_review_messages("A question", "A draft")
        self.assertIn("A draft", review[1]["content"])
        with self.assertRaises(ValueError):
            build_messages("A question", "unbounded_retry")


if __name__ == "__main__":
    unittest.main()
