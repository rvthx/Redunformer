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
    resolve_harness_tasks,
    write_json,
)
from redundancy.models import RedundancyModel
from redundancy.utils import get_model_metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline evaluation")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset", default="wikitext-103-raw-v1")
    parser.add_argument("--quantization", choices=["auto", "4bit", "none"], default="auto")
    parser.add_argument("--disable-lm-harness", action="store_true")
    parser.add_argument(
        "--harness-tasks",
        nargs="+",
        default=["hellaswag", "piqa", "arc_easy"],
        help="Any supported subset, or 'all'",
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
    wrapper.model.eval()
    dataset = get_wikitext_dataset(subset=args.dataset, split="test")

    loss, perplexity, tokens = evaluate_perplexity(
        model=wrapper.model,
        tokenizer=wrapper.tokenizer,
        dataset=dataset,
        device=wrapper.device,
    )
    harness = {}
    if not args.disable_lm_harness:
        harness = evaluate_lm_harness(
            model=wrapper.model,
            tokenizer=wrapper.tokenizer,
            device=wrapper.device,
            tasks=harness_tasks,
        )

    metadata = get_model_metadata(wrapper.model)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "result_type": "baseline",
        "model": args.model,
        "model_revision": args.model_revision,
        "quantization": "4bit" if wrapper.is_quantized else "none",
        "dataset": args.dataset,
        "evaluation_split": "test",
        "calibration_split": None,
        "pruning_method": "baseline",
        "requested_pruning_ratio": 0.0,
        "actual_pruning_ratio": 0.0,
        "seed": None,
        "eligible_layers": list(metadata.eligible_layers),
        "selected_heads": {},
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
    output = output_dir / "baseline_schema_v2.json"
    write_json(output, payload)
    print(f"Baseline evaluation completed: {output}")


if __name__ == "__main__":
    main()
