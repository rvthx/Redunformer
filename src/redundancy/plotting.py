from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def plot_redundancy_heatmap(measurement: dict, path: str | Path) -> Path:
    layers = [int(layer) for layer in measurement["layers"]]
    scores = np.asarray(
        [measurement["layers"][str(layer)]["head_scores"] for layer in layers],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(max(7, scores.shape[1] * 0.55), max(5, len(layers) * 0.28)))
    image = ax.imshow(scores, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Attention head")
    ax.set_ylabel("Transformer layer")
    ax.set_title("Head redundancy by layer")
    ax.set_xticks(range(scores.shape[1]))
    if len(layers) <= 40:
        ax.set_yticks(range(len(layers)), labels=layers)
    fig.colorbar(image, ax=ax, label="Maximum centered-cosine similarity")
    return _save(fig, path)


def most_redundant_layer(measurement: dict) -> int:
    return max(
        (int(layer) for layer in measurement["layers"]),
        key=lambda layer: measurement["layers"][str(layer)]["layer_summary"]["mean_redundancy"],
    )


def plot_similarity_matrix(
    measurement: dict,
    path: str | Path,
    layer: int | None = None,
) -> Path:
    selected_layer = most_redundant_layer(measurement) if layer is None else layer
    matrix = np.asarray(
        measurement["layers"][str(selected_layer)]["similarity_matrix"],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Head")
    ax.set_ylabel("Head")
    ax.set_title(f"Head-output similarity — layer {selected_layer}")
    ax.set_xticks(range(matrix.shape[0]))
    ax.set_yticks(range(matrix.shape[0]))
    fig.colorbar(image, ax=ax, label="Centered-cosine similarity")
    return _save(fig, path)


def plot_depth_profile(measurement: dict, path: str | Path) -> Path:
    layers = [int(layer) for layer in measurement["layers"]]
    values = [
        measurement["layers"][str(layer)]["layer_summary"]["mean_redundancy"]
        for layer in layers
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(layers, values, marker="o")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Mean head redundancy")
    ax.set_title("Redundancy across model depth")
    ax.grid(alpha=0.25)
    return _save(fig, path)
