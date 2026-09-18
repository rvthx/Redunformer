import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from redundancy.data import get_wikitext_dataset
from redundancy.models import RedundancyModel
from redundancy.pruning.similarity_pruning import compute_head_similarities


def parse_args():
    parser = argparse.ArgumentParser(description="Measure attention-head output redundancy")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--dataset", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--similarity-batches", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--signed-similarity", action="store_true")
    parser.add_argument("--quantization", choices=["auto", "4bit", "none"], default="auto")
    parser.add_argument("--output-root", default="configs/experiments/redundancy")
    return parser.parse_args()


def main():
    args = parse_args()
    redundancy_model = RedundancyModel(args.model, quantization=args.quantization)
    dataset = get_wikitext_dataset(subset=args.dataset, split=args.split)

    matrices = compute_head_similarities(
        model=redundancy_model.model,
        tokenizer=redundancy_model.tokenizer,
        dataset=dataset,
        device=redundancy_model.device,
        num_batches=args.similarity_batches,
        max_length=args.max_length,
        seed=args.seed,
        absolute=not args.signed_similarity,
    )

    head_rows = []
    layer_summaries = []
    score_rows = []

    for layer in sorted(matrices):
        matrix = matrices[layer]
        scores = matrix.max(dim=1).values
        score_rows.append(scores.tolist())
        for head, score in enumerate(scores.tolist()):
            head_rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "redundancy_score": score,
                }
            )
        layer_summaries.append(
            {
                "layer": layer,
                "mean_head_redundancy": float(scores.mean()),
                "max_head_redundancy": float(scores.max()),
            }
        )

    model_slug = args.model.replace("/", "--")
    out_dir = Path(args.output_root) / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_slug}_{args.dataset}_seed{args.seed}"

    payload = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "similarity_batches": args.similarity_batches,
        "max_length": args.max_length,
        "absolute_similarity": not args.signed_similarity,
        "definition": (
            "Per-head redundancy score is the maximum centered-cosine similarity "
            "to another head in the same eligible attention layer."
        ),
        "eligible_layers": sorted(matrices),
        "head_scores": head_rows,
        "layer_summaries": layer_summaries,
        "similarity_matrices": {
            str(layer): matrix.tolist() for layer, matrix in matrices.items()
        },
    }

    json_path = out_dir / f"{stem}_head_redundancy.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = out_dir / f"{stem}_head_redundancy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "head", "redundancy_score"])
        writer.writeheader()
        writer.writerows(head_rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(score_rows, aspect="auto")
    ax.set_title(f"Head redundancy scores: {args.model}")
    ax.set_xlabel("Head index")
    ax.set_ylabel("Eligible layer order")
    fig.colorbar(image, ax=ax, label="max centered-cosine similarity")
    fig.tight_layout()
    heatmap_path = out_dir / f"{stem}_head_redundancy_heatmap.png"
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)

    most_redundant_layer = max(
        layer_summaries, key=lambda item: item["mean_head_redundancy"]
    )["layer"]
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrices[most_redundant_layer].numpy(), vmin=0, vmax=1)
    ax.set_title(f"Similarity matrix, layer {most_redundant_layer}")
    ax.set_xlabel("Head")
    ax.set_ylabel("Head")
    fig.colorbar(image, ax=ax, label="centered-cosine similarity")
    fig.tight_layout()
    matrix_path = out_dir / f"{stem}_most_redundant_layer_matrix.png"
    fig.savefig(matrix_path, dpi=200)
    plt.close(fig)

    print(f"Saved measurement JSON: {json_path}")
    print(f"Saved sortable CSV: {csv_path}")
    print(f"Saved heatmap: {heatmap_path}")
    print(f"Saved layer matrix: {matrix_path}")


if __name__ == "__main__":
    main()
