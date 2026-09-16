from __future__ import annotations

from collections.abc import Callable

from redundancy.pruning.gradient_pruning import select_gradient_pruning_plan
from redundancy.pruning.plan import PruningPlan, apply_pruning_plan
from redundancy.pruning.random_pruning import select_random_pruning_plan
from redundancy.pruning.similarity_pruning import select_similarity_pruning_plan


PRUNING_METHODS: dict[str, Callable[..., PruningPlan]] = {
    "gradient": select_gradient_pruning_plan,
    "random": select_random_pruning_plan,
    "similarity": select_similarity_pruning_plan,
}


def register_pruning_method(
    name: str,
    selector: Callable[..., PruningPlan],
) -> None:
    if not name or name in PRUNING_METHODS:
        raise ValueError(
            f"Pruning method is empty or already registered: {name!r}"
        )

    PRUNING_METHODS[name] = selector


def select_pruning_plan(
    method: str,
    **kwargs,
) -> PruningPlan:
    try:
        selector = PRUNING_METHODS[method]

    except KeyError as error:
        choices = ", ".join(
            sorted(PRUNING_METHODS)
        )

        raise ValueError(
            f"Unknown pruning method {method!r}; "
            f"choose one of: {choices}"
        ) from error

    plan = selector(**kwargs)

    plan.validate()

    return plan


__all__ = [
    "PRUNING_METHODS",
    "PruningPlan",
    "apply_pruning_plan",
    "register_pruning_method",
    "select_pruning_plan",
]