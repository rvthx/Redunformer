import argparse
import time
from pathlib import Path

from redundancy.data import get_wikitext_dataset
from redundancy.eval import evaluate_lm_harness, evaluate_perplexity
from redundancy.experiment import (
    SCHEMA_VERSION,
    current_command,
    current_git_commit,
    model_slug,
    ratio_seed_suffix,
    resolve_harness_tasks,
    write_json,
)
from redundancy.measurement import load_head_measurement, similarities_from_measurement
from redundancy.models import RedundancyModel
from redundancy.pruning.plan import apply_pruning_plan
from redundancy.pruning.similarity_pruning import select_similarity_pruning_plan


def parse_args():
    parser = argparse.ArgumentParser(description="Run similarity-guided attention-head pruning")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset", default="wikitext-103-raw-v1")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--similarity-batches", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--measurement-file", default=None)
    parser.add_argument("--signed-similarity", action="store_true")
    parser.add_argument("--quantization", choices=["auto", "4bit", "none"], default="auto")
    parser.add_argument("--disable-lm-harness", action="store_true")
    parser.add_argument(
        "--harness-tasks",
        nargs="+",
        default=["hellaswag", "piqa", "arc_easy"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    harness_tasks = resolve_harness_tasks(args.harness_tasks)
    wrapper = RedundancyModel(
        args.model,
        quantization=args.quantization,
        revision=args.model_revision,
    )
    evaluation_dataset = get_wikitext_dataset(subset=args.dataset, split="test")
    calibration_dataset = get_wikitext_dataset(subset=args.dataset, split="validation")
    cached = None
    if args.measurement_file:
        cached = similarities_from_measurement(
            load_head_measurement(args.measurement_file)
        )

    plan = select_similarity_pruning_plan(
        model=wrapper.model,
        ratio=args.ratio,
        tokenizer=wrapper.tokenizer,
        dataset=calibration_dataset,
        device=wrapper.device,
        num_batches=args.similarity_batches,
        max_length=args.max_length,
        seed=args.seed,
        absolute=not args.signed_similarity,
        model_name=args.model,
        model_revision=args.model_revision,
        similarity_matrices=cached,
    )
    handles = apply_pruning_plan(wrapper.model, plan)
    try:
        loss, perplexity, tokens = evaluate_perplexity(
            wrapper.model, wrapper.tokenizer, evaluation_dataset, wrapper.device
        )
        harness = {}
        if not args.disable_lm_harness:
            harness = evaluate_lm_harness(
                wrapper.model, wrapper.tokenizer, wrapper.device, harness_tasks
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
        "pruning_method": "similarity",
        "requested_pruning_ratio": args.ratio,
        "actual_pruning_ratio": plan.actual_ratio,
        "seed": args.seed,
        "eligible_layers": plan.eligible_layers,
        "selected_heads": {str(k): v for k, v in plan.selected_heads.items()},
        "metrics": {
            "loss": loss,
            "perplexity": perplexity,
            "tokens": tokens,
            "lm_harness": harness,
        },
        "harness_enabled": not args.disable_lm_harness,
        "harness_tasks": harness_tasks if not args.disable_lm_harness else [],
        "measurement_batches": args.similarity_batches,
        "measurement_max_length": args.max_length,
        "measurement_file": args.measurement_file,
        "absolute_similarity": not args.signed_similarity,
        "command": current_command(),
        "git_commit": current_git_commit(),
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    }

    output_dir = Path("configs/experiments") / model_slug(args.model)
    suffix = ratio_seed_suffix(args.ratio, args.seed)
    output = output_dir / f"similarity_{model_slug(args.model)}_{suffix}.json"
    plan.save(output_dir / f"similarity_plan_{model_slug(args.model)}_{suffix}.json")
    write_json(output, payload)
    print(f"Similarity pruning completed: {output}")


if __name__ == "__main__":
    main()
