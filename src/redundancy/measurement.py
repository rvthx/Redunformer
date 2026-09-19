from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch

from redundancy.hooks import HeadOutputStatsHook
from redundancy.pruning.gradient_pruning import compute_head_importance
from redundancy.pruning.similarity_pruning import compute_centered_cosine_similarity
from redundancy.utils import get_model_metadata, get_output_projection

MEASUREMENT_SCHEMA_VERSION = 1


def collect_head_measurements(
    model,
    tokenizer,
    dataset,
    device,
    num_batches: int = 16,
    max_length: int = 512,
    seed: int = 42,
    absolute: bool = True,
) -> dict[str, Any]:
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")

    metadata = get_model_metadata(model)
    collectors = {
        layer: HeadOutputStatsHook(metadata.num_heads, metadata.head_dim)
        for layer in metadata.eligible_layers
    }
    handles = [
        get_output_projection(model, layer, metadata.model_type).register_forward_pre_hook(
            collectors[layer]
        )
        for layer in metadata.eligible_layers
    ]

    texts = [text for text in dataset["text"] if text and text.strip()]
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
    sequence_length = encodings.input_ids.size(1)
    if sequence_length < max_length:
        for handle in handles:
            handle.remove()
        raise ValueError(
            f"Dataset is too short ({sequence_length} tokens) for max_length={max_length}"
        )

    max_start = sequence_length - max_length
    starts = torch.randint(
        0,
        max_start + 1,
        (num_batches,),
        generator=torch.Generator().manual_seed(seed),
    )

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in starts.tolist():
                input_ids = encodings.input_ids[:, start : start + max_length].to(device)
                model(input_ids=input_ids)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    layers: dict[str, Any] = {}
    for layer, collector in collectors.items():
        similarity = compute_centered_cosine_similarity(collector, absolute=absolute)
        count = float(collector.count)
        mean = collector.sum / count
        second_moment = collector.sum_sq / count
        variance = (second_moment - mean.square()).clamp_min(0.0)
        std = torch.sqrt(variance)
        rms = torch.sqrt(second_moment.clamp_min(0.0))
        head_scores, partners = similarity.max(dim=1)

        layers[str(layer)] = {
            "similarity_matrix": similarity.tolist(),
            "head_scores": head_scores.tolist(),
            "most_similar_head": partners.tolist(),
            "activation_mean": mean.tolist(),
            "activation_std": std.tolist(),
            "activation_rms": rms.tolist(),
            "layer_summary": {
                "mean_redundancy": float(head_scores.mean()),
                "max_redundancy": float(head_scores.max()),
                "std_redundancy": float(head_scores.std(unbiased=False)),
            },
        }

    return {
        "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
        "measurement": "head_output_centered_cosine",
        "absolute_similarity": absolute,
        "seed": seed,
        "num_batches": num_batches,
        "max_length": max_length,
        "model_type": metadata.model_type,
        "num_layers": metadata.num_layers,
        "num_heads": metadata.num_heads,
        "head_dim": metadata.head_dim,
        "eligible_layers": list(metadata.eligible_layers),
        "layers": layers,
    }


def save_head_measurement(
    measurement: dict[str, Any],
    json_path: str | Path,
    csv_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    json_destination = Path(json_path)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(
        json.dumps(measurement, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    if csv_path is None:
        return json_destination, None

    csv_destination = Path(csv_path)
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    with csv_destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "layer",
                "head",
                "redundancy_score",
                "most_similar_head",
                "activation_mean",
                "activation_std",
                "activation_rms",
            ],
        )
        writer.writeheader()
        for layer_key, layer_data in measurement["layers"].items():
            for head, score in enumerate(layer_data["head_scores"]):
                writer.writerow(
                    {
                        "layer": int(layer_key),
                        "head": head,
                        "redundancy_score": score,
                        "most_similar_head": layer_data["most_similar_head"][head],
                        "activation_mean": layer_data["activation_mean"][head],
                        "activation_std": layer_data["activation_std"][head],
                        "activation_rms": layer_data["activation_rms"][head],
                    }
                )

    return json_destination, csv_destination


def load_head_measurement(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("measurement_schema_version") != MEASUREMENT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported measurement schema in {path}")
    return data


def similarities_from_measurement(measurement: dict[str, Any]) -> dict[int, torch.Tensor]:
    return {
        int(layer): torch.tensor(layer_data["similarity_matrix"], dtype=torch.float64)
        for layer, layer_data in measurement["layers"].items()
    }


def compute_and_save_gradient_importance(
    model,
    tokenizer,
    dataset,
    device,
    path: str | Path,
    num_batches: int = 16,
    max_length: int = 512,
    seed: int = 42,
) -> dict[int, torch.Tensor]:
    importance = compute_head_importance(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        num_batches=num_batches,
        max_length=max_length,
        seed=seed,
    )
    metadata = get_model_metadata(model)
    payload = {
        "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
        "measurement": "first_order_taylor_head_importance",
        "seed": seed,
        "num_batches": num_batches,
        "max_length": max_length,
        "eligible_layers": list(metadata.eligible_layers),
        "num_heads": metadata.num_heads,
        "importance": {
            str(layer): scores.tolist()
            for layer, scores in importance.items()
        },
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return importance


def load_gradient_importance(path: str | Path) -> dict[int, torch.Tensor]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("measurement_schema_version") != MEASUREMENT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported measurement schema in {path}")
    if data.get("measurement") != "first_order_taylor_head_importance":
        raise ValueError(f"{path} is not a gradient-importance cache")
    return {
        int(layer): torch.tensor(scores, dtype=torch.float32)
        for layer, scores in data["importance"].items()
    }
