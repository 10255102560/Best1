"""Numeric benchmark loading and strict, deterministic answer checking.

GSM8K is loaded from its original JSONL files.  The generated ``demo`` tasks
are local arithmetic fixtures, not an alternative name for GSM8K.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any


_NUMBER = re.compile(
    r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?\Z"
)
_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_GSM_MARKER = re.compile(r"####\s*([^\r\n]*)")


def parse_number(value: Any) -> Decimal | None:
    """Parse a finite numeric value without accepting units or malformed commas."""
    text = str(value).strip().replace("\u2212", "-")
    if not _NUMBER.fullmatch(text):
        return None
    try:
        number = Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def extract_answer(text: str) -> Decimal | None:
    """Read an explicitly marked answer; never guess from reasoning numbers.

    The last explicit marker wins, including an invalid marker.  A response
    ending with an invalid correction therefore cannot pass using an earlier
    valid answer.  Tags and the original GSM8K ``####`` syntax are accepted.
    """
    candidates = [(match.start(), match.group(1)) for match in _ANSWER_TAG.finditer(text)]
    candidates.extend((match.start(), match.group(1)) for match in _GSM_MARKER.finditer(text))
    if not candidates:
        return None
    start, value = max(candidates, key=lambda candidate: candidate[0])
    # An unfinished later tag is a malformed final answer, not permission to
    # reuse the earlier tagged answer.
    if text.lower().rfind("<answer>") > start:
        return None
    return parse_number(value)


def _number_string(number: Decimal) -> str:
    # Avoid Decimal.normalize(), which can round using the active context.
    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if number == 0 else text


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    answer: str
    source_split: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, str]) -> "Task":
        return cls(**value)


def is_correct(response: str, gold: str | Decimal | Task) -> bool:
    """Exact numeric equality after strict final-answer extraction."""
    expected = parse_number(gold.answer if isinstance(gold, Task) else gold)
    actual = extract_answer(response)
    return actual is not None and expected is not None and actual == expected


def question_hash(question: str) -> str:
    """Stable identity ignoring superficial case and whitespace differences."""
    canonical = " ".join(question.casefold().split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_gsm8k(path: str | Path, source_split: str) -> list[Task]:
    tasks: list[Task] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                question = row["question"].strip()
                answer = extract_answer(row["answer"])
            except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid GSM8K row at {path}:{line_number}") from exc
            if not question or answer is None:
                raise ValueError(f"Missing question or numeric marked answer at {path}:{line_number}")
            digest = question_hash(question)
            if digest in seen:
                continue
            seen.add(digest)
            tasks.append(Task(f"gsm8k-{digest}", question, _number_string(answer), source_split))
    return tasks


def _check_sizes(train_size: int, val_size: int, test_size: int) -> None:
    if any(not isinstance(size, int) or isinstance(size, bool) or size < 1
           for size in (train_size, val_size, test_size)):
        raise ValueError("train_size, val_size and test_size must be positive integers")


def load_gsm8k(
    train_path: str | Path,
    test_path: str | Path,
    train_size: int = 40,
    val_size: int = 20,
    test_size: int = 40,
    seed: int = 42,
) -> dict[str, list[Task]]:
    """Draw disjoint development subsets and a held-out official test subset.

    Any question occurring in official test is removed from official train
    before sampling.  Official test labels are never used for sample selection.
    """
    _check_sizes(train_size, val_size, test_size)
    test_pool = _read_gsm8k(test_path, "gsm8k_test")
    test_hashes = {question_hash(task.question) for task in test_pool}
    train_pool = [task for task in _read_gsm8k(train_path, "gsm8k_train")
                  if question_hash(task.question) not in test_hashes]
    if len(train_pool) < train_size + val_size:
        raise ValueError(f"Need {train_size + val_size} distinct training questions; have {len(train_pool)}")
    if len(test_pool) < test_size:
        raise ValueError(f"Need {test_size} distinct test questions; have {len(test_pool)}")
    random.Random(seed).shuffle(train_pool)
    random.Random(seed + 1).shuffle(test_pool)
    return {
        "train": train_pool[:train_size],
        "val": train_pool[train_size:train_size + val_size],
        "test": test_pool[:test_size],
    }


def _demo_question(rng: random.Random, family: int) -> tuple[str, int]:
    if family == 0:
        boxes, items, given = rng.randint(2, 30), rng.randint(3, 25), rng.randint(1, 5)
        return (f"A store has {boxes} boxes with {items} pencils in each box. "
                f"It gives away {given} pencils. How many pencils remain?", boxes * items - given)
    if family == 1:
        count_a, price_a, count_b, price_b = (rng.randint(2, 20) for _ in range(4))
        return (f"Mira buys {count_a} notebooks at {price_a} dollars each and "
                f"{count_b} pens at {price_b} dollars each. What is the total cost in dollars?",
                count_a * price_a + count_b * price_b)
    if family == 2:
        speed, hours, walking = rng.randint(15, 100), rng.randint(2, 8), rng.randint(1, 20)
        return (f"A traveler cycles at {speed} kilometers per hour for {hours} hours, "
                f"then walks another {walking} kilometers. What is the total distance in kilometers?",
                speed * hours + walking)
    if family == 3:
        denominator, share = rng.randint(2, 9), rng.randint(4, 80)
        extra = rng.randint(1, share - 1)
        return (f"A club has {denominator * share} tickets. One out of every {denominator} "
                f"tickets is reserved for members. Of those reserved tickets, {extra} are cancelled. "
                "How many reserved tickets remain?", share - extra)
    workers, daily, days, damaged = rng.randint(2, 12), rng.randint(3, 30), rng.randint(2, 9), rng.randint(1, 20)
    return (f"There are {workers} workers. Each makes {daily} parts per day for {days} days. "
            f"Afterward {damaged} parts fail inspection. How many parts pass inspection?",
            workers * daily * days - damaged)


def build_demo_splits(
    train_size: int = 40,
    val_size: int = 20,
    test_size: int = 40,
    seed: int = 42,
) -> dict[str, list[Task]]:
    """Generate exact arithmetic fixtures, clearly identified as synthetic demo."""
    _check_sizes(train_size, val_size, test_size)
    rng = random.Random(seed)
    seen: set[str] = set()
    result: dict[str, list[Task]] = {}
    for split, size in (("train", train_size), ("val", val_size), ("test", test_size)):
        rows: list[Task] = []
        while len(rows) < size:
            question, answer = _demo_question(rng, rng.randrange(5))
            digest = question_hash(question)
            if digest not in seen:
                seen.add(digest)
                rows.append(Task(f"demo-{digest}", question, str(answer), f"demo_{split}"))
        result[split] = rows
    return result
