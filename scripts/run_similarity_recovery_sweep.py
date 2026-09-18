from __future__ import annotations

import argparse
import subprocess
import sys

MODELS = [
    "gpt2",
    "Qwen/Qwen3.5-4B",
    "meta-llama/Llama-3.2-3B",
]


def parse_float_list(values):
    return [float(value) for value in values.split(",") if value.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LoRA/QLoRA recovery after similarity-guided head pruning"
    )
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument(
        "--ratios",
        default="0.20,0.40",
        help="Comma-separated representative pruning ratios for recovery",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-tokens", type=int, default=1_000_000)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--run-lm-harness", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for model in args.models:
        for ratio in parse_float_list(args.ratios):
            command = [
                sys.executable,
                "scripts/run_recovery.py",
                "--model",
                model,
                "--pruning-method",
                "similarity",
                "--ratio",
                str(ratio),
                "--seed",
                str(args.seed),
                "--max-train-tokens",
                str(args.max_train_tokens),
                "--sequence-length",
                str(args.sequence_length),
            ]
            if args.run_lm_harness:
                command.append("--run-lm-harness")
            print("\n$", " ".join(command), flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
