from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = 2

ALL_HARNESS_TASKS = [
    "hellaswag",
    "lambada",
    "piqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
]


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "--")


def ratio_seed_suffix(ratio: float, seed: int) -> str:
    return f"r{ratio:.4f}_seed{seed}"


def resolve_seeds(seed: int | None, seeds: list[int] | None) -> list[int]:
    if seed is not None and seeds:
        raise ValueError("Use either --seed or --seeds, not both")
    if seeds:
        return list(dict.fromkeys(seeds))
    return [42 if seed is None else seed]


def parse_float_values(values: Iterable[str | float] | None, default: list[float]) -> list[float]:
    if values is None:
        return list(default)
    parsed: list[float] = []
    for value in values:
        if isinstance(value, float):
            parsed.append(value)
            continue
        for part in str(value).split(","):
            part = part.strip()
            if part:
                parsed.append(float(part))
    return parsed


def resolve_harness_tasks(tasks: list[str] | None) -> list[str]:
    if not tasks:
        return list(ALL_HARNESS_TASKS)

    flattened: list[str] = []
    for item in tasks:
        flattened.extend(part.strip() for part in item.split(",") if part.strip())

    if "all" in flattened:
        return list(ALL_HARNESS_TASKS)

    unknown = sorted(set(flattened) - set(ALL_HARNESS_TASKS))
    if unknown:
        choices = ", ".join(ALL_HARNESS_TASKS)
        raise ValueError(f"Unknown lm-eval task(s): {unknown}. Available: {choices}, or 'all'.")

    return list(dict.fromkeys(flattened))


def ratio_selected(requested_ratio: float, selected_ratios: Iterable[float], tolerance: float = 1e-9) -> bool:
    return any(abs(requested_ratio - candidate) <= tolerance for candidate in selected_ratios)


def current_command() -> str:
    return shlex.join([sys.executable, *sys.argv])


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: str | Path, value) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return destination
