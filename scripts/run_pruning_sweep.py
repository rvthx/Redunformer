from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from redundancy.data import get_wikitext_dataset
from redundancy.eval import evaluate_lm_harness, evaluate_perplexity
from redundancy.experiment import (
    SCHEMA_VERSION,
    current_command,
    current_git_commit,
    model_slug,
    parse_float_values,
    ratio_seed_suffix,
    ratio_selected,
    resolve_harness_tasks,
    resolve_seeds,
    write_json,
)
from redundancy.measurement import (
    collect_head_measurements,
    compute_and_save_gradient_importance,
    load_gradient_importance,
    load_head_measurement,
    save_head_measurement,
    similarities_from_measurement,
)
from redundancy.models import RedundancyModel
from redundancy.plotting import (
    plot_depth_profile,
    plot_redundancy_heatmap,
    plot_similarity_matrix,
)
from redundancy.pruning import apply_pruning_plan, select_pruning_plan
from redundancy.utils import get_model_metadata


METHODS = ["similarity", "gradient", "random"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run attention-head pruning ratio sweeps")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset", default="wikitext-103-raw-v1")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument(
        "--ratios",
        nargs="+",
        default=["0.05", "0.10", "0.20", "0.30", "0.40", "0.50"],
        help="Requested pruning ratios; commas are also accepted",
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=None)
    seed_group.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--harness-ratios",
        nargs="+",
        default=["0.20", "0.40"],
        help="Only these requested ratios run lm-eval; commas are also accepted",
    )
    parser.add_argument(
        "--harness-tasks",
        nargs="+",
        default=["hellaswag", "piqa", "arc_easy"],
        help="Any subset of supported tasks, or 'all'",
    )
    parser.add_argument("--disable-lm-harness", action="store_true")
    parser.add_argument("--measurement-batches", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--signed-similarity", action="store_true")
    parser.add_argument("--quantization", choices=["auto", "4bit", "none"], default="auto")
    parser.add_argument("--output-root", default="configs/experiments")
    parser.add_argument("--force-remeasure", action="store_true")
    return parser.parse_args()


def evaluate(model_wrapper, dataset, run_harness: bool, harness_tasks: list[str]):
    loss, perplexity, tokens = evaluate_perplexity(
        model=model_wrapper.model,
        tokenizer=model_wrapper.tokenizer,
        dataset=dataset,
        device=model_wrapper.device,
    )
    harness = {}
    if run_harness:
        harness = evaluate_lm_harness(
            model=model_wrapper.model,
            tokenizer=model_wrapper.tokenizer,
            device=model_wrapper.device,
            tasks=harness_tasks,
        )
    return {
        "loss": loss,
        "perplexity": perplexity,
        "tokens": tokens,
        "lm_harness": harness,
    }


def main():
    args = parse_args()
    ratios = parse_float_values(args.ratios, [])
    harness_ratios = parse_float_values(args.harness_ratios, [0.20, 0.40])
    seeds = resolve_seeds(args.seed, args.seeds)
    harness_tasks = resolve_harness_tasks(args.harness_tasks)

    if any(not 0 <= ratio <= 1 for ratio in ratios):
        raise ValueError("All pruning ratios must be between 0 and 1")

    wrapper = RedundancyModel(
        args.model,
        quantization=args.quantization,
        revision=args.model_revision,
    )
    wrapper.model.eval()
    model_meta = get_model_metadata(wrapper.model)
    evaluation_dataset = get_wikitext_dataset(subset=args.dataset, split="test")
    calibration_dataset = get_wikitext_dataset(subset=args.dataset, split="validation")

    root = Path(args.output_root) / model_slug(args.model)
    root.mkdir(parents=True, exist_ok=True)
    measurement_dir = root / "measurements"
    measurement_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = evaluate(
        wrapper,
        evaluation_dataset,
        run_harness=not args.disable_lm_harness,
        harness_tasks=harness_tasks,
    )
    baseline_payload = {
        "schema_version": SCHEMA_VERSION,
        "result_type": "baseline",
        "model": args.model,
        "model_revision": args.model_revision,
        "quantization": "4bit" if wrapper.is_quantized else "none",
        "dataset": args.dataset,
        "evaluation_split": "test",
        "calibration_split": "validation",
        "pruning_method": "baseline",
        "requested_pruning_ratio": 0.0,
        "actual_pruning_ratio": 0.0,
        "seed": None,
        "eligible_layers": list(model_meta.eligible_layers),
        "selected_heads": {},
        "metrics": baseline_metrics,
        "harness_enabled": not args.disable_lm_harness,
        "harness_tasks": harness_tasks if not args.disable_lm_harness else [],
        "command": current_command(),
        "git_commit": current_git_commit(),
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    }
    write_json(root / "baseline_schema_v2.json", baseline_payload)

    for seed in seeds:
        similarity_matrices = None
        importance_scores = None

        if "similarity" in args.methods:
            similarity_path = measurement_dir / f"head_redundancy_seed{seed}.json"
            similarity_csv = measurement_dir / f"head_redundancy_seed{seed}.csv"
            if similarity_path.exists() and not args.force_remeasure:
                measurement = load_head_measurement(similarity_path)
            else:
                measurement = collect_head_measurements(
                    model=wrapper.model,
                    tokenizer=wrapper.tokenizer,
                    dataset=calibration_dataset,
                    device=wrapper.device,
                    num_batches=args.measurement_batches,
                    max_length=args.max_length,
                    seed=seed,
                    absolute=not args.signed_similarity,
                )
                measurement.update(
                    {
                        "model": args.model,
                        "model_revision": args.model_revision,
                        "dataset": args.dataset,
                        "split": "validation",
                        "quantization": "4bit" if wrapper.is_quantized else "none",
                    }
                )
                save_head_measurement(measurement, similarity_path, similarity_csv)
                plot_redundancy_heatmap(
                    measurement,
                    measurement_dir / f"head_redundancy_seed{seed}_heatmap.png",
                )
                plot_similarity_matrix(
                    measurement,
                    measurement_dir / f"head_redundancy_seed{seed}_similarity_matrix.png",
                )
                plot_depth_profile(
                    measurement,
                    measurement_dir / f"head_redundancy_seed{seed}_depth_profile.png",
                )
            similarity_matrices = similarities_from_measurement(measurement)

        if "gradient" in args.methods:
            importance_path = measurement_dir / f"gradient_importance_seed{seed}.json"
            if importance_path.exists() and not args.force_remeasure:
                importance_scores = load_gradient_importance(importance_path)
            else:
                importance_scores = compute_and_save_gradient_importance(
                    model=wrapper.model,
                    tokenizer=wrapper.tokenizer,
                    dataset=calibration_dataset,
                    device=wrapper.device,
                    path=importance_path,
                    num_batches=args.measurement_batches,
                    max_length=args.max_length,
                    seed=seed,
                )

        for method in args.methods:
            for ratio in ratios:
                kwargs = {}
                if method == "similarity":
                    kwargs["similarity_matrices"] = similarity_matrices
                    kwargs["absolute"] = not args.signed_similarity
                elif method == "gradient":
                    kwargs["importance_scores"] = importance_scores

                plan = select_pruning_plan(
                    method=method,
                    model=wrapper.model,
                    model_name=args.model,
                    model_revision=args.model_revision,
                    ratio=ratio,
                    seed=seed,
                    tokenizer=wrapper.tokenizer,
                    dataset=calibration_dataset,
                    device=wrapper.device,
                    num_batches=args.measurement_batches,
                    max_length=args.max_length,
                    **kwargs,
                )
                handles = apply_pruning_plan(wrapper.model, plan)
                run_harness = (
                    not args.disable_lm_harness
                    and ratio_selected(ratio, harness_ratios)
                )
                try:
                    metrics = evaluate(
                        wrapper,
                        evaluation_dataset,
                        run_harness=run_harness,
                        harness_tasks=harness_tasks,
                    )
                finally:
                    for handle in handles:
                        handle.remove()

                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "result_type": "pruning",
                    "model": args.model,
                    "model_revision": args.model_revision,
                    "quantization": "4bit" if wrapper.is_quantized else "none",
                    "dataset": args.dataset,
                    "evaluation_split": "test",
                    "calibration_split": "validation",
                    "pruning_method": method,
                    "requested_pruning_ratio": ratio,
                    "actual_pruning_ratio": plan.actual_ratio,
                    "seed": seed,
                    "eligible_layers": plan.eligible_layers,
                    "selected_heads": {
                        str(layer): heads
                        for layer, heads in plan.selected_heads.items()
                    },
                    "metrics": metrics,
                    "harness_enabled": run_harness,
                    "harness_tasks": harness_tasks if run_harness else [],
                    "measurement_batches": args.measurement_batches,
                    "measurement_max_length": args.max_length,
                    "command": current_command(),
                    "git_commit": current_git_commit(),
                    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
                }

                suffix = ratio_seed_suffix(ratio, seed)
                output_file = root / f"{method}_{model_slug(args.model)}_{suffix}.json"
                plan_file = root / f"{method}_plan_{model_slug(args.model)}_{suffix}.json"
                write_json(output_file, payload)
                plan.save(plan_file)
                print(
                    f"{method:10s} requested={ratio:.4f} "
                    f"actual={plan.actual_ratio:.4f} seed={seed} -> {output_file}"
                )


if __name__ == "__main__":
    main()
