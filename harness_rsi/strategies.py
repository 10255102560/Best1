"""Frozen execution choices whose routing is optimized by the harness."""

from __future__ import annotations


STRATEGIES = ("direct", "reason", "review")

_FORMAT = (
    "End your response with exactly one final answer in the form "
    "<answer>number</answer>. Use a plain numeric value, without units, "
    "currency symbols, or fractions inside the answer tags."
)


def build_messages(question: str, strategy: str) -> list[dict[str, str]]:
    """Create the first call, without access to the benchmark reference answer.

    Review begins with the same reasoning instruction as reason, then makes
    a second call with ``build_review_messages``.  Each separately measured
    strategy pays for all of its calls.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    if strategy == "direct":
        instruction = "Solve the arithmetic problem. Give only the final answer. "
    else:
        instruction = "Solve the arithmetic problem carefully, showing concise calculation steps. "
    return [
        {"role": "system", "content": instruction + _FORMAT},
        {"role": "user", "content": question},
    ]


def build_review_messages(question: str, draft: str) -> list[dict[str, str]]:
    """Request one bounded self-review, without exposing labels or grader feedback."""
    return [
        {"role": "system", "content": (
            "Independently check the proposed solution to the arithmetic problem. "
            "Check the interpretation, arithmetic, and requested quantity. Correct any errors; "
            "otherwise keep its answer. Show only concise verification. " + _FORMAT
        )},
        {"role": "user", "content": (
            f"Problem:\n{question}\n\nProposed solution:\n{draft}\n\n"
            "Return your checked solution and final numeric answer."
        )},
    ]
