from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from redundancy.data import get_c4_recovery_dataset, get_wikitext_dataset
from redundancy.eval import evaluate_lm_harness, evaluate_perplexity
from redundancy.models import RedundancyModel
from redundancy.pruning import PRUNING_METHODS, apply_pruning_plan, select_pruning_plan
from redundancy.recovery import RecoveryConfig, recover_with_lora, save_recovery_adapter

HARNESS_TASKS = [
    "hellaswag",
    "lambada",
    "piqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
]


def _evaluate_loss(redundancy_model, dataset):
    loss, perplexity, tokens = evaluate_perplexity(
        model=redundancy_model.model,
        tokenizer=redundancy_model.tokenizer,
        dataset=dataset,
        device=redundancy_model.device,
    )
    return {"loss": loss, "perplexity": perplexity, "tokens": tokens}


def _model_slug(model_name):
    return model_name.replace("/", "--")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Recover causal-LM loss after head pruning")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--pruning-method", choices=sorted(PRUNING_METHODS), default="gradient")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--importance-batches", type=int, default=16)
    parser.add_argument("--max-train-tokens", type=int, default=1_000_000)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--c4-revision", default="main")
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    parser.add_argument("--quantization", choices=["auto", "4bit", "none"], default="auto")
    parser.add_argument("--run-lm-harness", action="store_true")
    parser.add_argument("--output-root", default="outputs/recovery")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_slug = _model_slug(args.model)
    run_id = f"{args.pruning_method}_{args.ratio:.2f}_{timestamp}"
    output_directory = Path(args.output_root) / model_slug / run_id
    output_directory.mkdir(parents=True, exist_ok=False)

    print(f"Loading {args.model} with quantization={args.quantization}")
    redundancy_model = RedundancyModel(
        args.model,
        quantization=args.quantization,
        revision=args.model_revision,
    )
    wiki_test = get_wikitext_dataset(split="test")
    baseline_metrics = _evaluate_loss(redundancy_model, wiki_test)

    calibration_dataset = None
    if args.pruning_method in {"gradient", "similarity"}:
        calibration_dataset = get_wikitext_dataset(
        split="validation"
        )

    plan = select_pruning_plan(
        method=args.pruning_method,
        model=redundancy_model.model,
        model_name=args.model,
        model_revision=args.model_revision,
        ratio=args.ratio,
        seed=args.seed,
        tokenizer=redundancy_model.tokenizer,
        dataset=calibration_dataset,
        device=redundancy_model.device,
        num_batches=args.importance_batches,
        max_length=args.sequence_length,
    )
    plan.save(output_directory / "pruning_plan.json")

    pruned_handles = apply_pruning_plan(redundancy_model.model, plan)
    pruned_metrics = _evaluate_loss(redundancy_model, wiki_test)
    pruned_harness = {}
    if args.run_lm_harness:
        pruned_harness = evaluate_lm_harness(
            redundancy_model.model,
            redundancy_model.tokenizer,
            redundancy_model.device,
            HARNESS_TASKS,
        )
    for handle in pruned_handles:
        handle.remove()

    c4_dataset = get_c4_recovery_dataset(
        tokenizer=redundancy_model.tokenizer,
        max_train_tokens=args.max_train_tokens,
        sequence_length=args.sequence_length,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        revision=args.c4_revision,
    )
    recovery_config = RecoveryConfig()
    recovery_result = recover_with_lora(
        model=redundancy_model.model,
        tokenizer=redundancy_model.tokenizer,
        dataset=c4_dataset,
        plan=plan,
        config=recovery_config,
        device=redundancy_model.device,
    )
    redundancy_model.model = recovery_result.model
    recovered_metrics = _evaluate_loss(redundancy_model, wiki_test)
    recovered_harness = {}
    if args.run_lm_harness:
        recovered_harness = evaluate_lm_harness(
            redundancy_model.model,
            redundancy_model.tokenizer,
            redundancy_model.device,
            HARNESS_TASKS,
        )

    denominator = pruned_metrics["loss"] - baseline_metrics["loss"]
    if denominator <= 0:
        print("Warning: pruned loss did not exceed baseline loss; recovery percentage is undefined")
        recovered_percent = None
    else:
        recovered_percent = 100 * (pruned_metrics["loss"] - recovered_metrics["loss"]) / denominator
        if not math.isfinite(recovered_percent):
            recovered_percent = None

    metrics = {
        "model": args.model,
        "model_revision": args.model_revision,
        "pruning_method": args.pruning_method,
        "requested_pruning_ratio": args.ratio,
        "actual_pruning_ratio": plan.actual_ratio,
        "seed": args.seed,
        "quantization": "4bit" if redundancy_model.is_quantized else "none",
        "baseline": baseline_metrics,
        "pruned": {**pruned_metrics, "lm_harness": pruned_harness},
        "recovered": {**recovered_metrics, "lm_harness": recovered_harness},
        "loss_recovered_percent": recovered_percent,
        "c4": c4_dataset.metadata.to_dict(),
        "training": {
            **recovery_config.to_dict(),
            "optimizer_steps": recovery_result.optimizer_steps,
            "tokens_seen": recovery_result.tokens_seen,
            "trainable_parameters": recovery_result.trainable_parameters,
            "total_parameters": recovery_result.total_parameters,
        },
        "hardware_device": str(redundancy_model.device),
        "timestamp": timestamp,
    }
    run_config = vars(args)
    save_recovery_adapter(recovery_result, output_directory)
    _write_json(output_directory / "metrics.json", metrics)
    _write_json(output_directory / "training_history.json", recovery_result.training_history)
    _write_json(output_directory / "run_config.json", run_config)

    mirror_path = Path("configs/experiments") / model_slug / f"recovery_{run_id}.json"
    _write_json(mirror_path, metrics)

    print(f"Recovery completed. Artifacts: {output_directory}")
    print(f"Recovered loss: {recovered_metrics['loss']:.4f} ({recovered_percent}%)")


if __name__ == "__main__":
    main()
