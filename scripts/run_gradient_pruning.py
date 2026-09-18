import argparse
import json
import time
from pathlib import Path

from redundancy.data import get_wikitext_dataset
from redundancy.eval import evaluate_lm_harness, evaluate_perplexity
from redundancy.models import RedundancyModel
from redundancy.pruning import apply_pruning_plan
from redundancy.pruning.gradient_pruning import select_gradient_pruning_plan

HARNESS_TASKS = [
    "hellaswag",
    "lambada",
    "piqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Gradient Importance Head Pruning Evaluation")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--dataset", type=str, default="wikitext-103-raw-v1", help="Dataset name")
    parser.add_argument(
        "--ratio", type=float, default=0.2, help="Requested fraction of heads to prune per layer"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for calibration sampling")
    parser.add_argument("--quantization", choices=["auto", "4bit", "none"], default="auto")
    parser.add_argument(
        "--importance-batches",
        type=int,
        default=16,
        help="Number of calibration windows used for gradient importance",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Calibration sequence length")
    parser.add_argument(
        "--disable-lm-harness", action="store_true", help="Disable LM Harness evaluation"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(
        f"Initializing gradient head pruning for {args.model}: "
        f"requested ratio={args.ratio:.4f}"
    )

    redundancy_model = RedundancyModel(args.model, quantization=args.quantization)
    redundancy_model.model.eval()

    dataset = get_wikitext_dataset(subset=args.dataset)
    importance_dataset = get_wikitext_dataset(subset=args.dataset, split="validation")

    plan = select_gradient_pruning_plan(
        model=redundancy_model.model,
        model_name=args.model,
        ratio=args.ratio,
        tokenizer=redundancy_model.tokenizer,
        dataset=importance_dataset,
        device=redundancy_model.device,
        num_batches=args.importance_batches,
        max_length=args.max_length,
        seed=args.seed,
    )
    active_hooks = apply_pruning_plan(redundancy_model.model, plan)

    avg_nll, perplexity, n_tokens = evaluate_perplexity(
        model=redundancy_model.model,
        tokenizer=redundancy_model.tokenizer,
        dataset=dataset,
        device=redundancy_model.device,
    )

    harness_res = {}
    if args.disable_lm_harness:
        print("LM Harness evaluation is disabled.")
    else:
        harness_res = evaluate_lm_harness(
            model=redundancy_model.model,
            tokenizer=redundancy_model.tokenizer,
            device=redundancy_model.device,
            tasks=HARNESS_TASKS,
        )

    results = {
        "model": args.model,
        "pruning_method": "gradient_head_pruning",
        "requested_pruning_ratio": args.ratio,
        "actual_pruning_ratio": plan.actual_ratio,
        "seed": args.seed,
        "importance_batches": args.importance_batches,
        "importance_max_length": args.max_length,
        "dataset": args.dataset,
        "eligible_layers": plan.eligible_layers,
        "selected_heads": {
            str(layer): heads for layer, heads in plan.selected_heads.items()
        },
        "pruned_loss": round(avg_nll, 4),
        "pruned_perplexity": round(perplexity, 4),
        "total_tokens_evaluated": n_tokens,
        "hardware_device": str(redundancy_model.device),
        "lm_harness_metrics": harness_res,
        "head_importance": plan.metadata.get("head_importance"),
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    }

    for hook in active_hooks:
        hook.remove()

    model_slug = args.model.replace("/", "--")
    output_directory = Path("configs/experiments") / model_slug
    output_directory.mkdir(parents=True, exist_ok=True)

    ratio_slug = f"{args.ratio:.4f}"
    output_file = output_directory / (
        f"gradient_pruned_results_{model_slug}_{args.dataset}_{ratio_slug}_seed{args.seed}.json"
    )
    plan_file = output_directory / (
        f"gradient_pruning_plan_{model_slug}_{args.dataset}_{ratio_slug}_seed{args.seed}.json"
    )

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    plan.save(plan_file)

    print(
        "Gradient pruning evaluation completed. "
        f"actual ratio={plan.actual_ratio:.4f}; results={output_file}"
    )


if __name__ == "__main__":
    main()
