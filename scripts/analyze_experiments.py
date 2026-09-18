from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


METHOD_LABELS = {
    "random_head_pruning": "Random",
    "gradient_head_pruning": "Gradient",
    "similarity_head_pruning": "Similarity",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate and plot pruning experiments")
    parser.add_argument("--experiments-root", default="configs/experiments")
    parser.add_argument("--output-root", default="configs/experiments/analysis")
    return parser.parse_args()


def load_pruning_results(root: Path):
    records = []
    for path in root.glob("*/*_pruned_results_*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        method = data.get("pruning_method")
        if method not in METHOD_LABELS:
            continue
        actual = data.get("actual_pruning_ratio")
        if actual is None:
            actual = data.get("pruning_ratio")
        records.append(
            {
                "path": str(path),
                "model": data["model"],
                "method": method,
                "label": METHOD_LABELS[method],
                "requested_ratio": data.get(
                    "requested_pruning_ratio", data.get("pruning_ratio")
                ),
                "actual_ratio": actual,
                "seed": data.get("seed", data.get("random_seed")),
                "perplexity": data["pruned_perplexity"],
                "loss": data["pruned_loss"],
                "lm_harness_metrics": data.get("lm_harness_metrics", {}),
            }
        )
    return records


def aggregate(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["model"], row["label"], row["actual_ratio"])].append(row)

    summary = []
    for (model, label, ratio), rows in sorted(grouped.items()):
        ppls = [row["perplexity"] for row in rows]
        losses = [row["loss"] for row in rows]
        summary.append(
            {
                "model": model,
                "method": label,
                "actual_ratio": ratio,
                "runs": len(rows),
                "perplexity_mean": mean(ppls),
                "perplexity_std": stdev(ppls) if len(ppls) > 1 else 0.0,
                "loss_mean": mean(losses),
                "loss_std": stdev(losses) if len(losses) > 1 else 0.0,
            }
        )
    return summary


def plot_curves(summary, output_root: Path):
    models = sorted({row["model"] for row in summary})
    for model in models:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in ("Random", "Gradient", "Similarity"):
            rows = sorted(
                [
                    row
                    for row in summary
                    if row["model"] == model and row["method"] == method
                ],
                key=lambda row: row["actual_ratio"],
            )
            if not rows:
                continue
            x = [row["actual_ratio"] for row in rows]
            y = [row["perplexity_mean"] for row in rows]
            yerr = [row["perplexity_std"] for row in rows]
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=method)

        ax.set_title(f"Head pruning: {model}")
        ax.set_xlabel("Actual per-layer pruning ratio")
        ax.set_ylabel("WikiText perplexity")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        model_slug = model.replace("/", "--")
        fig.savefig(output_root / f"{model_slug}_perplexity_vs_ratio.png", dpi=200)
        plt.close(fig)


def analyze_depth(experiments_root: Path, output_root: Path):
    measurement_root = experiments_root / "redundancy"
    for path in measurement_root.glob("*/*_head_redundancy.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        layers = data["layer_summaries"]
        if not layers:
            continue

        indices = [row["layer"] for row in layers]
        scores = [row["mean_head_redundancy"] for row in layers]
        n = len(layers)
        shallow_cut = max(1, n // 3)
        deep_start = max(shallow_cut + 1, 2 * n // 3)

        bins = {
            "shallow": layers[:shallow_cut],
            "middle": layers[shallow_cut:deep_start],
            "deep": layers[deep_start:],
        }
        depth_summary = {
            name: mean(row["mean_head_redundancy"] for row in rows) if rows else None
            for name, rows in bins.items()
        }

        model_slug = data["model"].replace("/", "--")
        (output_root / f"{model_slug}_depth_summary.json").write_text(
            json.dumps(
                {
                    "model": data["model"],
                    "depth_bins": depth_summary,
                    "layer_summaries": layers,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(indices, scores, marker="o")
        ax.set_title(f"Redundancy vs depth: {data['model']}")
        ax.set_xlabel("Transformer layer index")
        ax.set_ylabel("Mean head redundancy")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_root / f"{model_slug}_redundancy_vs_depth.png", dpi=200)
        plt.close(fig)


def main():
    args = parse_args()
    experiments_root = Path(args.experiments_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records = load_pruning_results(experiments_root)
    summary = aggregate(records)

    csv_path = output_root / "pruning_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "model",
            "method",
            "actual_ratio",
            "runs",
            "perplexity_mean",
            "perplexity_std",
            "loss_mean",
            "loss_std",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    plot_curves(summary, output_root)
    analyze_depth(experiments_root, output_root)

    print(f"Saved aggregated summary: {csv_path}")
    print(f"Saved plots and depth analysis under: {output_root}")


if __name__ == "__main__":
    main()
