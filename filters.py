"""The two semantically equivalent filter expressions from issue 54195."""

from __future__ import annotations


ORIGINAL_FILTER = "category == 1"
TRUE_PREDICATE = "(age >= 1) and (age <= 1000)"


def equivalent_filter(repeats: int = 128) -> str:
    """Return ``F AND TRUE AND ... AND TRUE`` with ``repeats`` true predicates."""

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    return f"({ORIGINAL_FILTER}) and " + " and ".join(
        f"({TRUE_PREDICATE})" for _ in range(repeats)
    )
