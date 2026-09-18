from __future__ import annotations

import argparse
import subprocess
import sys

MODELS = [
    "gpt2",
    "Qwen/Qwen3.5-4B",
    "meta-llama/Llama-3.2-3B",
]

METHOD_TO_SCRIPT = {
    "similarity": "scripts/run_similarity_pruning.py",
    "gradient": "scripts/run_gradient_pruning.py",
    "random": "scripts/run_random_pruning.py",
}


def parse_float_list(values):
    return [float(value) for value in values.split(",") if value.strip()]


def parse_int_list(values):
    return [int(value) for value in values.split(",") if value.strip()]


def run(command):
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run head-pruning sweeps across GPT-2, Qwen, and Llama"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="Hugging Face model IDs",
    )
    parser.add_argument(
        "--ratios",
        default="0.10,0.20,0.30,0.40,0.50",
        help="Comma-separated requested per-layer pruning ratios",
    )
    parser.add_argument(
        "--random-seeds",
        default="1,2,3,4,5",
        help="Comma-separated seeds used for random baseline repetitions",
    )
    parser.add_argument(
        "--calibration-seed",
        type=int,
        default=42,
        help="Calibration-window seed for similarity and gradient methods",
    )
    parser.add_argument(
        "--harness-ratios",
        default="0.20,0.40",
        help="Ratios on which to run downstream lm-eval-harness tasks",
    )
    parser.add_argument("--similarity-batches", type=int, default=16)
    parser.add_argument("--importance-batches", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--skip-measurement", action="store_true")
    parser.add_argument("--skip-gradient", action="store_true")
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--skip-similarity", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ratios = parse_float_list(args.ratios)
    random_seeds = parse_int_list(args.random_seeds)
    harness_ratios = parse_float_list(args.harness_ratios)

    for model in args.models:
        if not args.skip_measurement:
            run(
                [
                    sys.executable,
                    "scripts/measure_redundancy.py",
                    "--model",
                    model,
                    "--seed",
                    str(args.calibration_seed),
                    "--similarity-batches",
                    str(args.similarity_batches),
                    "--max-length",
                    str(args.max_length),
                ]
            )

        for ratio in ratios:
            run_harness = any(abs(ratio - target) < 1e-12 for target in harness_ratios)

            if not args.skip_similarity:
                command = [
                    sys.executable,
                    METHOD_TO_SCRIPT["similarity"],
                    "--model",
                    model,
                    "--ratio",
                    str(ratio),
                    "--seed",
                    str(args.calibration_seed),
                    "--similarity-batches",
                    str(args.similarity_batches),
                    "--max-length",
                    str(args.max_length),
                ]
                if not run_harness:
                    command.append("--disable-lm-harness")
                run(command)

            if not args.skip_gradient:
                command = [
                    sys.executable,
                    METHOD_TO_SCRIPT["gradient"],
                    "--model",
                    model,
                    "--ratio",
                    str(ratio),
                    "--seed",
                    str(args.calibration_seed),
                    "--importance-batches",
                    str(args.importance_batches),
                    "--max-length",
                    str(args.max_length),
                ]
                if not run_harness:
                    command.append("--disable-lm-harness")
                run(command)

            if not args.skip_random:
                for seed in random_seeds:
                    command = [
                        sys.executable,
                        METHOD_TO_SCRIPT["random"],
                        "--model",
                        model,
                        "--ratio",
                        str(ratio),
                        "--seed",
                        str(seed),
                    ]
                    if not run_harness:
                        command.append("--disable-lm-harness")
                    run(command)


if __name__ == "__main__":
    main()
