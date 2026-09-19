from __future__ import annotations

import argparse
from pathlib import Path

from redundancy.data import get_wikitext_dataset
from redundancy.experiment import model_slug
from redundancy.measurement import collect_head_measurements, save_head_measurement
from redundancy.models import RedundancyModel
from redundancy.plotting import (
    plot_depth_profile,
    plot_redundancy_heatmap,
    plot_similarity_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Measure attention-head redundancy")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--signed-similarity", action="store_true")
    parser.add_argument("--quantization", choices=["auto", "4bit", "none"], default="auto")
    parser.add_argument("--output-root", default="configs/experiments")
    return parser.parse_args()


def main():
    args = parse_args()
    redundancy_model = RedundancyModel(
        args.model,
        quantization=args.quantization,
        revision=args.model_revision,
    )
    dataset = get_wikitext_dataset(subset=args.dataset, split=args.split)

    measurement = collect_head_measurements(
        model=redundancy_model.model,
        tokenizer=redundancy_model.tokenizer,
        dataset=dataset,
        device=redundancy_model.device,
        num_batches=args.batches,
        max_length=args.max_length,
        seed=args.seed,
        absolute=not args.signed_similarity,
    )
    measurement.update(
        {
            "model": args.model,
            "model_revision": args.model_revision,
            "dataset": args.dataset,
            "split": args.split,
            "quantization": "4bit" if redundancy_model.is_quantized else "none",
        }
    )

    output_dir = (
        Path(args.output_root)
        / model_slug(args.model)
        / "measurements"
    )
    stem = f"head_redundancy_seed{args.seed}"
    json_path, csv_path = save_head_measurement(
        measurement,
        output_dir / f"{stem}.json",
        output_dir / f"{stem}.csv",
    )
    plot_redundancy_heatmap(measurement, output_dir / f"{stem}_heatmap.png")
    plot_similarity_matrix(measurement, output_dir / f"{stem}_similarity_matrix.png")
    plot_depth_profile(measurement, output_dir / f"{stem}_depth_profile.png")

    print(f"Measurement JSON: {json_path}")
    print(f"Measurement CSV: {csv_path}")
    print(f"Plots: {output_dir}")


if __name__ == "__main__":
    main()
