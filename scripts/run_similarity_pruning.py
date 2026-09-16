import argparse
import json
import time
from pathlib import Path

from redundancy.data import get_wikitext_dataset
from redundancy.eval import evaluate_lm_harness, evaluate_perplexity
from redundancy.models import RedundancyModel
from redundancy.pruning.similarity_pruning import similarity_prune_model


HARNESS_TASKS = [
    "hellaswag",
    "lambada",
    "piqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Similarity-Guided Attention Head Pruning Evaluation"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext-103-raw-v1",
        help="Dataset name",
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="Percentage of heads to prune (0.0 to 1.0)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for calibration-window sampling",
    )

    parser.add_argument(
        "--quantization",
        choices=["auto", "4bit", "none"],
        default="auto",
    )

    parser.add_argument(
        "--similarity-batches",
        type=int,
        default=16,
        help="Number of batches used to estimate head-output similarity",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Sequence length used for similarity estimation",
    )

    parser.add_argument(
        "--signed-similarity",
        action="store_true",
        help=(
            "Use signed centered-cosine similarity instead of absolute similarity"
        ),
    )

    parser.add_argument(
        "--disable-lm-harness",
        action="store_true",
        help="Disable LM Harness evaluation",
    )

    return parser.parse_args()


def main():
    print("Starting similarity-guided pruning evaluation script...")

    args = parse_args()

    print(
        f"Initializing similarity-guided pruning evaluation for: "
        f"{args.model} at {args.ratio * 100}% sparsity"
    )

    redundancy_model = RedundancyModel(
        args.model,
        quantization=args.quantization,
    )

    redundancy_model.model.eval()

    # Evaluation dataset
    dataset = get_wikitext_dataset(
        subset=args.dataset,
    )

    # Separate calibration dataset for measuring head similarity
    similarity_dataset = get_wikitext_dataset(
        subset=args.dataset,
        split="validation",
    )

    active_hooks, plan = similarity_prune_model(
        model=redundancy_model.model,
        tokenizer=redundancy_model.tokenizer,
        dataset=similarity_dataset,
        device=redundancy_model.device,
        sparsity=args.ratio,
        num_batches=args.similarity_batches,
        max_length=args.max_length,
        seed=args.seed,
        absolute=not args.signed_similarity,
    )

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
        "pruning_method": "similarity_head_pruning",
        "pruning_ratio": args.ratio,
        "actual_pruning_ratio": plan.actual_ratio,
        "seed": args.seed,
        "similarity_batches": args.similarity_batches,
        "similarity_max_length": args.max_length,
        "absolute_similarity": not args.signed_similarity,
        "selection_strategy": "greedy_pairwise",
        "dataset": args.dataset,
        "selected_heads": {
            str(layer): heads
            for layer, heads in plan.selected_heads.items()
        },
        "pruned_loss": round(avg_nll, 4),
        "pruned_perplexity": round(perplexity, 4),
        "total_tokens_evaluated": n_tokens,
        "hardware_device": str(redundancy_model.device),
        "lm_harness_metrics": harness_res,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    }

    for hook in active_hooks:
        hook.remove()

    model_slug = args.model.replace("/", "--")

    output_directory = (
        Path("configs/experiments")
        / model_slug
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_directory / (
        f"similarity_pruned_results_"
        f"{model_slug}_"
        f"{args.dataset}_"
        f"{args.ratio:.2f}.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    # Save the complete pruning plan separately.
    plan_file = output_directory / (
        f"similarity_pruning_plan_"
        f"{model_slug}_"
        f"{args.dataset}_"
        f"{args.ratio:.2f}.json"
    )

    plan.save(plan_file)

    print(
        f"Similarity-guided pruning evaluation completed. "
        f"Results saved to {output_file}"
    )

    print(
        f"Pruning plan saved to {plan_file}"
    )


if __name__ == "__main__":
    main()