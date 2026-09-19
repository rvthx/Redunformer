from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

from redundancy.experiment import SCHEMA_VERSION
from redundancy.measurement import load_gradient_importance, load_head_measurement


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate pruning experiments and create report-ready outputs")
    parser.add_argument("--experiments-root", default="configs/experiments")
    parser.add_argument("--output-dir", default="configs/experiments/analysis")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_results(root: Path, verbose: bool = False):
    records = []
    for path in root.rglob("*.json"):
        if "analysis" in path.parts or "measurements" in path.parts:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            if verbose:
                print(f"Skipping unreadable result: {path}")
            continue

        if data.get("schema_version") != SCHEMA_VERSION:
            if verbose:
                print(f"Skipping legacy result: {path}")
            continue

        required = {
            "model",
            "pruning_method",
            "requested_pruning_ratio",
            "actual_pruning_ratio",
            "metrics",
        }
        if not required.issubset(data):
            if verbose:
                print(f"Skipping incompatible result: {path}")
            continue

        data["_path"] = str(path)
        records.append(data)
    return records


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def baseline_by_model(records):
    baselines = {}
    for row in records:
        if row["pruning_method"] == "baseline":
            baselines[row["model"]] = row
    return baselines


def create_pruning_summary(records, baselines, output_dir: Path):
    rows = []
    for row in records:
        if row["pruning_method"] == "baseline":
            continue
        baseline = baselines.get(row["model"])
        if baseline is None:
            continue
        ppl = row["metrics"].get("perplexity")
        baseline_ppl = baseline["metrics"].get("perplexity")
        if ppl is None or baseline_ppl is None:
            continue
        delta = ppl - baseline_ppl
        relative = delta / baseline_ppl if baseline_ppl else None
        rows.append(
            {
                "model": row["model"],
                "method": row["pruning_method"],
                "seed": row.get("seed"),
                "requested_ratio": row["requested_pruning_ratio"],
                "actual_ratio": row["actual_pruning_ratio"],
                "perplexity": ppl,
                "baseline_perplexity": baseline_ppl,
                "delta_perplexity": delta,
                "relative_perplexity_change": relative,
                "harness_enabled": row.get("harness_enabled", False),
            }
        )
    fieldnames = [
        "model",
        "method",
        "seed",
        "requested_ratio",
        "actual_ratio",
        "perplexity",
        "baseline_perplexity",
        "delta_perplexity",
        "relative_perplexity_change",
        "harness_enabled",
    ]
    write_csv(output_dir / "pruning_summary.csv", rows, fieldnames)
    return rows


def create_similarity_vs_random(records, baselines, output_dir: Path):
    index = {}
    for row in records:
        method = row["pruning_method"]
        if method not in {"similarity", "random"}:
            continue
        key = (row["model"], row.get("seed"), row["requested_pruning_ratio"])
        index.setdefault(key, {})[method] = row

    rows = []
    for (model, seed, requested_ratio), pair in sorted(index.items()):
        if set(pair) != {"similarity", "random"}:
            continue
        baseline = baselines.get(model)
        if baseline is None:
            continue
        similarity_ppl = pair["similarity"]["metrics"].get("perplexity")
        random_ppl = pair["random"]["metrics"].get("perplexity")
        baseline_ppl = baseline["metrics"].get("perplexity")
        if None in {similarity_ppl, random_ppl, baseline_ppl}:
            continue
        rows.append(
            {
                "model": model,
                "seed": seed,
                "requested_ratio": requested_ratio,
                "similarity_actual_ratio": pair["similarity"]["actual_pruning_ratio"],
                "random_actual_ratio": pair["random"]["actual_pruning_ratio"],
                "baseline_perplexity": baseline_ppl,
                "similarity_perplexity": similarity_ppl,
                "random_perplexity": random_ppl,
                "similarity_minus_random_perplexity": similarity_ppl - random_ppl,
                "similarity_delta_perplexity": similarity_ppl - baseline_ppl,
                "random_delta_perplexity": random_ppl - baseline_ppl,
                "similarity_relative_ppl_change": (similarity_ppl - baseline_ppl) / baseline_ppl,
                "random_relative_ppl_change": (random_ppl - baseline_ppl) / baseline_ppl,
            }
        )

    fieldnames = [
        "model",
        "seed",
        "requested_ratio",
        "similarity_actual_ratio",
        "random_actual_ratio",
        "baseline_perplexity",
        "similarity_perplexity",
        "random_perplexity",
        "similarity_minus_random_perplexity",
        "similarity_delta_perplexity",
        "random_delta_perplexity",
        "similarity_relative_ppl_change",
        "random_relative_ppl_change",
    ]
    write_csv(output_dir / "similarity_vs_random.csv", rows, fieldnames)
    return rows


def metric_value(task_payload):
    if not isinstance(task_payload, dict):
        return None
    preferred = ["acc_norm,none", "acc,none", "exact_match,none", "word_perplexity,none"]
    for key in preferred:
        value = task_payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    for key, value in task_payload.items():
        if key.endswith("_stderr"):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def create_task_comparison(records, output_dir: Path):
    index = {}
    tasks = set()
    for row in records:
        if row["pruning_method"] not in {"similarity", "random"}:
            continue
        harness = row["metrics"].get("lm_harness") or {}
        key = (row["model"], row.get("seed"), row["requested_pruning_ratio"])
        index.setdefault(key, {})[row["pruning_method"]] = row
        tasks.update(harness)

    rows = []
    for key, pair in sorted(index.items()):
        if set(pair) != {"similarity", "random"}:
            continue
        model, seed, ratio = key
        sim_h = pair["similarity"]["metrics"].get("lm_harness") or {}
        rnd_h = pair["random"]["metrics"].get("lm_harness") or {}
        for task in sorted(tasks):
            sim_value = metric_value(sim_h.get(task))
            rnd_value = metric_value(rnd_h.get(task))
            if sim_value is None or rnd_value is None:
                continue
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "requested_ratio": ratio,
                    "task": task,
                    "similarity_metric": sim_value,
                    "random_metric": rnd_value,
                    "similarity_minus_random": sim_value - rnd_value,
                }
            )

    write_csv(
        output_dir / "similarity_vs_random_tasks.csv",
        rows,
        [
            "model",
            "seed",
            "requested_ratio",
            "task",
            "similarity_metric",
            "random_metric",
            "similarity_minus_random",
        ],
    )
    return rows


def plot_performance(summary_rows, output_dir: Path):
    by_model = defaultdict(list)
    for row in summary_rows:
        by_model[row["model"]].append(row)

    for model, rows in by_model.items():
        for field, ylabel, suffix in [
            ("perplexity", "Perplexity", "perplexity_vs_removal"),
            ("relative_perplexity_change", "Relative perplexity change", "relative_degradation"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for method in sorted({row["method"] for row in rows}):
                method_rows = [row for row in rows if row["method"] == method]
                grouped = defaultdict(list)
                for row in method_rows:
                    grouped[row["actual_ratio"]].append(row[field])
                xs = sorted(grouped)
                ys = [mean(grouped[x]) for x in xs]
                ax.plot(xs, ys, marker="o", label=method)
            ax.set_xlabel("Actual removed-head ratio")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{model}: {ylabel} vs head removal")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            safe_model = model.replace("/", "--")
            fig.savefig(output_dir / f"{safe_model}_{suffix}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)


def create_similarity_importance_analysis(root: Path, output_dir: Path):
    rows = []
    for measurement_path in root.rglob("measurements/head_redundancy_seed*.json"):
        seed_text = measurement_path.stem.split("seed")[-1]
        try:
            seed = int(seed_text)
        except ValueError:
            continue
        importance_path = measurement_path.parent / f"gradient_importance_seed{seed}.json"
        if not importance_path.exists():
            continue

        measurement = load_head_measurement(measurement_path)
        importance = load_gradient_importance(importance_path)
        model = measurement.get("model", measurement_path.parent.parent.name)

        x_values = []
        y_values = []
        for layer_key, layer_data in measurement["layers"].items():
            layer = int(layer_key)
            if layer not in importance:
                continue
            for head, similarity in enumerate(layer_data["head_scores"]):
                importance_value = float(importance[layer][head])
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "layer": layer,
                        "head": head,
                        "redundancy_score": similarity,
                        "gradient_importance": importance_value,
                    }
                )
                x_values.append(float(similarity))
                y_values.append(importance_value)

        if len(x_values) >= 2:
            pearson = float(np.corrcoef(x_values, y_values)[0, 1])
            ranks_x = np.argsort(np.argsort(x_values))
            ranks_y = np.argsort(np.argsort(y_values))
            spearman = float(np.corrcoef(ranks_x, ranks_y)[0, 1])

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(x_values, y_values, alpha=0.7)
            ax.set_xlabel("Head redundancy score")
            ax.set_ylabel("Gradient importance")
            ax.set_title(f"{model}, seed {seed}: similarity vs importance\n"
                         f"Pearson={pearson:.3f}, Spearman={spearman:.3f}")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            safe_model = str(model).replace("/", "--")
            fig.savefig(
                output_dir / f"{safe_model}_seed{seed}_similarity_vs_importance.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)

    write_csv(
        output_dir / "similarity_vs_importance.csv",
        rows,
        ["model", "seed", "layer", "head", "redundancy_score", "gradient_importance"],
    )


def main():
    args = parse_args()
    root = Path(args.experiments_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_results(root, verbose=args.verbose)
    baselines = baseline_by_model(records)
    summary_rows = create_pruning_summary(records, baselines, output_dir)
    create_similarity_vs_random(records, baselines, output_dir)
    create_task_comparison(records, output_dir)
    plot_performance(summary_rows, output_dir)
    create_similarity_importance_analysis(root, output_dir)

    print(f"Loaded {len(records)} schema-v{SCHEMA_VERSION} results")
    print(f"Analysis outputs: {output_dir}")


if __name__ == "__main__":
    main()
