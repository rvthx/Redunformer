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
from redundancy.models import RedundancyModel
from redundancy.pruning.plan import apply_pruning_plan
from redundancy.pruning.random_pruning import select_random_pruning_plan


def parse_args():
    parser = argparse.ArgumentParser(description="Run random attention-head pruning")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset", default="wikitext-103-raw-v1")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
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
    dataset = get_wikitext_dataset(subset=args.dataset, split="test")
    plan = select_random_pruning_plan(
        model=wrapper.model,
        ratio=args.ratio,
        seed=args.seed,
        model_name=args.model,
        model_revision=args.model_revision,
    )
    handles = apply_pruning_plan(wrapper.model, plan)
    try:
        loss, perplexity, tokens = evaluate_perplexity(
            wrapper.model, wrapper.tokenizer, dataset, wrapper.device
        )
        harness = {}
        if not args.disable_lm_harness:
            harness = evaluate_lm_harness(
                wrapper.model,
                wrapper.tokenizer,
                wrapper.device,
                harness_tasks,
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
        "calibration_split": None,
        "pruning_method": "random",
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
        "command": current_command(),
        "git_commit": current_git_commit(),
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    }

    output_dir = Path("configs/experiments") / model_slug(args.model)
    suffix = ratio_seed_suffix(args.ratio, args.seed)
    output = output_dir / f"random_{model_slug(args.model)}_{suffix}.json"
    plan.save(output_dir / f"random_plan_{model_slug(args.model)}_{suffix}.json")
    write_json(output, payload)
    print(f"Random pruning completed: {output}")


if __name__ == "__main__":
    main()
