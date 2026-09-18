# Redunformer

Head-level redundancy experiments for causal language models.

## Head-wise redundancy measurement

The measurement captures each eligible attention head immediately before the
layer output projection and computes an absolute centered-cosine similarity
matrix between head activation traces. A head's redundancy score is the maximum
similarity to any other head in the same layer.

```bash
uv run python scripts/measure_redundancy.py --model gpt2
uv run python scripts/measure_redundancy.py --model Qwen/Qwen3.5-4B
uv run python scripts/measure_redundancy.py --model meta-llama/Llama-3.2-3B
```

Outputs are written under
`configs/experiments/redundancy/<model>/`:

- `*_head_redundancy.json`: similarity matrices, per-head scores, and layer summaries
- `*_head_redundancy.csv`: sortable layer/head redundancy table
- `*_head_redundancy_heatmap.png`: layer-by-head redundancy heatmap
- `*_most_redundant_layer_matrix.png`: similarity matrix for the layer with the
  highest mean head redundancy

Use `--similarity-batches`, `--max-length`, and `--seed` to control the
calibration measurement.

Qwen3.5 hybrid models interleave full attention with gated-delta linear
attention. The current implementation measures and prunes standard full-attention
layers only; linear-attention value heads are deliberately excluded because they
have different semantics and dimensions.

## Full pruning sweep

The experiment sweep evaluates similarity-guided, gradient-importance, and
random head pruning on GPT-2, Qwen3.5-4B, and Llama-3.2-3B.

By default it evaluates requested ratios 0.10, 0.20, 0.30, 0.40, and 0.50.
Because a discrete number of heads is removed from each eligible layer, every
result also records `actual_pruning_ratio`. Random pruning is repeated with five
seeds by default.

```bash
uv run python scripts/run_experiment_sweep.py
```

The default sweep runs lm-evaluation-harness at requested ratios 0.20 and 0.40.
Other ratios still receive WikiText loss/perplexity evaluation. Customize the
experiment with, for example:

```bash
uv run python scripts/run_experiment_sweep.py \
  --ratios 0.10,0.20,0.30,0.40,0.50 \
  --random-seeds 1,2,3,4,5 \
  --harness-ratios 0.20,0.40
```

Each pruning run stores the selected heads, seed, requested ratio, actual ratio,
dataset, calibration settings, performance metrics, and a reusable pruning plan.

## Analysis

After the sweep, aggregate results and generate the main performance and depth
figures with:

```bash
uv run python scripts/analyze_experiments.py
```

This produces:

- `pruning_summary.csv`: method-wise performance aggregated by actual pruning ratio
- `similarity_vs_random.csv`: direct similarity-guided versus random comparison
- per-model perplexity-versus-pruning-ratio figures
- per-model redundancy-versus-depth figures
- shallow/middle/deep redundancy summaries

These outputs support the central questions of whether high head-output
similarity identifies removable heads, whether it outperforms a random baseline,
and how measured redundancy varies with model depth and model family.

## C4 loss recovery

Recover loss after pruning by training a LoRA adapter on a deterministic packed
subset of C4. WikiText is used for unpruned, pruned, and recovered evaluation.

A single recovery run:

```bash
uv run python scripts/run_recovery.py \
  --model gpt2 \
  --pruning-method similarity \
  --ratio 0.2
```

Run similarity-guided recovery on representative ratios across all three model
families:

```bash
uv run python scripts/run_similarity_recovery_sweep.py
```

The recovery sweep defaults to requested ratios 0.20 and 0.40. GPT-2 uses LoRA;
Qwen3.5-4B and Llama-3.2-3B use 4-bit NF4 QLoRA when
`--quantization auto` is selected by the underlying recovery script. Add
`--run-lm-harness` to the recovery sweep when downstream evaluation before and
after recovery is required.

Run artifacts are written under `outputs/recovery/`, with compact metrics copies
under `configs/experiments/<model>/`.
