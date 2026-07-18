import argparse
import json
import time

from redundancy.data import get_wikitext_dataset
from redundancy.eval import evaluate_lm_harness, evaluate_perplexity
from redundancy.models import RedundancyModel
from redundancy.pruning.gradient_pruning import gradient_prune_model


def main():
    print("Starting gradient-based pruning evaluation script...")
    parser = argparse.ArgumentParser(description="Run Gradient Importance Pruning Evaluation")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--dataset", type=str, default="wikitext-103-raw-v1", help="Dataset name")
    parser.add_argument(
        "--ratio", type=float, default=0.2, help="Percentage of heads to prune (0.0 to 1.0)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for batch sampling")
    parser.add_argument(
        "--importance-batches",
        type=int,
        default=16,
        help="Number of batches used to estimate gradient-based head importance",
    )
    parser.add_argument(
        "--disable-lm-harness", action="store_true", help="Disable LM Harness evaluation"
    )
    args = parser.parse_args()

    print(f"Initializing gradient pruning evaluation for: {args.model} at {args.ratio*100}% sparsity")
    redundancy_model = RedundancyModel(args.model)
    redundancy_model.model.eval()

    dataset = get_wikitext_dataset(subset=args.dataset)

    importance_dataset = get_wikitext_dataset(subset=args.dataset, split="validation")

    active_hooks, importance = gradient_prune_model(
        model=redundancy_model.model,
        tokenizer=redundancy_model.tokenizer,
        dataset=importance_dataset,
        device=redundancy_model.device,
        sparsity=args.ratio,
        num_batches=args.importance_batches,
        seed=args.seed,
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
            tasks=["hellaswag", "lambada", "piqa", "winogrande", "arc_easy", "arc_challenge"],
        )

    results = {
        "model": args.model,
        "pruning_method": "gradient_head_pruning",
        "pruning_ratio": args.ratio,
        "seed": args.seed,
        "importance_batches": args.importance_batches,
        "dataset": args.dataset,
        "pruned_loss": round(avg_nll, 4),
        "pruned_perplexity": round(perplexity, 4),
        "total_tokens_evaluated": n_tokens,
        "hardware_device": str(redundancy_model.device),
        "lm_harness_metrics": harness_res,
        "head_importance": importance.tolist() if importance is not None else None,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    }

    for hook in active_hooks:
        hook.remove()

    output_file = (
        f"configs/experiments/gradient_pruned_results_{args.model}_{args.dataset}_{args.ratio:.2f}.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, f)

    print(f"Gradient pruning evaluation completed. Results saved to {output_file}")


if __name__ == "__main__":
    main()