# Redunformer

Redunformer is an experimental framework for studying **attention-head redundancy in causal language models**.  
This branch focuses on measuring where redundant attention heads appear, pruning heads with different strategies, comparing redundancy-guided pruning against baselines, evaluating performance across pruning ratios, and testing post-pruning recovery.

The current experiment pipeline supports:

- **GPT-2**
- **Qwen3.5**
- **Llama 3.2**
- WikiText perplexity evaluation
- selected `lm-evaluation-harness` downstream tasks
- similarity-guided head pruning
- gradient-importance head pruning
- random head pruning
- LoRA / QLoRA recovery
- redundancy heatmaps and similarity matrices
- performance-vs-removal plots
- report-ready CSV summaries

---

## 1. Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/rvthx/Redunformer.git
cd Redunformer
git checkout experimental_last

uv sync
```

For gated Hugging Face models, provide an access token:

```bash
export HF_TOKEN=YOUR_TOKEN
```

The main experiment outputs are written under:

```text
configs/experiments/
```

Recovery artifacts are written under:

```text
outputs/recovery/
```

---

## 2. What is measured?

For each eligible attention layer, the project captures the concatenated attention-head outputs **immediately before the output projection**.

For every pair of heads within the same layer, it computes centered cosine similarity.

By default the absolute similarity is used:

```text
0.0  -> highly distinct head outputs
1.0  -> highly similar head outputs
```

For each head, its redundancy score is defined as the maximum similarity to any other head in the same layer.

The measurement pipeline also records:

- most similar head
- activation mean
- activation standard deviation
- activation RMS
- per-layer mean redundancy
- per-layer maximum redundancy
- per-layer redundancy standard deviation
- complete head-to-head similarity matrices

These measurements are used to answer questions such as:

- Where does redundancy appear across model depth?
- Which heads are most similar to other heads?
- Are highly similar heads more safely removable?
- Does similarity-guided pruning outperform random pruning?
- How does redundancy differ across model families?

---

# 3. Standalone redundancy measurement

Run a measurement independently of pruning:

```bash
uv run python scripts/measure_redundancy.py \
  --model gpt2 \
  --seed 42 \
  --batches 16 \
  --max-length 512
```

For another model:

```bash
uv run python scripts/measure_redundancy.py \
  --model Qwen/Qwen3.5-4B \
  --seed 42 \
  --batches 16 \
  --max-length 512
```

The measurement is computed on the WikiText validation split by default.

Outputs are written to:

```text
configs/experiments/<model>/measurements/
```

Example:

```text
configs/experiments/gpt2/measurements/
├── head_redundancy_seed42.json
├── head_redundancy_seed42.csv
├── head_redundancy_seed42_heatmap.png
├── head_redundancy_seed42_similarity_matrix.png
└── head_redundancy_seed42_depth_profile.png
```

---

## 4. Measurement plots

The standalone measurement automatically produces three plots.

### 4.1 Layer × head redundancy heatmap

Each cell corresponds to one attention head.

```text
x-axis: attention head
y-axis: transformer layer
value: maximum within-layer head similarity
```

This plot is intended to answer:

> Where does redundancy appear in the model?

It also provides a descriptive view of redundancy across model depth.

---

### 4.2 Head × head similarity matrix

A complete pairwise similarity matrix is generated for the layer with the highest mean redundancy.

```text
head_i × head_j -> centered-cosine similarity
```

This makes clusters or highly similar head pairs directly visible.

---

### 4.3 Redundancy across depth

The depth-profile plot shows:

```text
x-axis: transformer layer
y-axis: mean head redundancy
```

This can be used to compare shallow, middle, and deeper parts of the network descriptively.

---

# 5. Pruning methods

Three pruning strategies are implemented.

## Similarity-guided pruning

Heads are selected using the measured head-output similarity matrices.

The algorithm repeatedly:

1. finds the most similar active pair,
2. compares how redundant each head is relative to the remaining active heads,
3. removes the more redundant head,
4. repeats until the requested number of heads has been selected.

This avoids independently ranking all heads and pruning multiple members of the same redundancy group without updating the active set.

---

## Gradient-importance pruning

A differentiable mask is applied to each head and gradients are used to estimate head importance.

Heads with the lowest estimated importance are pruned first.

This provides a functional-importance baseline against the representational similarity measurement.

---

## Random pruning

Random pruning removes the same number of heads per eligible layer.

It is the primary simple baseline for testing whether similarity-guided redundancy is actually useful for identifying removable heads.

---

# 6. Main pruning sweep

The recommended experiment interface is:

```text
scripts/run_pruning_sweep.py
```

Example:

```bash
uv run python scripts/run_pruning_sweep.py \
  --model gpt2 \
  --methods similarity gradient random \
  --ratios 0.05 0.10 0.20 0.30 0.40 0.50 \
  --seed 42 \
  --harness-ratios 0.20 0.40 \
  --harness-tasks hellaswag piqa arc_easy
```

The sweep performs:

```text
baseline
   ↓
similarity measurement
   ↓
gradient-importance measurement
   ↓
pruning at every requested ratio
   ↓
WikiText evaluation
   ↓
LM-Harness only at selected ratios
   ↓
structured result files
```

---

## 6.1 Requested vs actual pruning ratio

The requested pruning ratio is not always exactly achievable.

For example, GPT-2 has 12 heads per layer:

```text
requested ratio = 0.20
heads removed   = int(12 × 0.20) = 2
actual ratio    = 2 / 12 = 0.1667
```

Every new result therefore stores both:

```json
{
  "requested_pruning_ratio": 0.20,
  "actual_pruning_ratio": 0.1666666667
}
```

Experiment scheduling uses the **requested ratio**.

Performance plots use the **actual ratio**.

---

# 7. Measurement caching

Similarity matrices and gradient-importance scores are expensive to compute but do not need to be recomputed for every pruning ratio.

The sweep therefore measures once per seed and caches the results under:

```text
configs/experiments/<model>/measurements/
```

Example:

```text
head_redundancy_seed42.json
gradient_importance_seed42.json
```

All pruning ratios for the same seed reuse these measurements.

To force recomputation:

```bash
uv run python scripts/run_pruning_sweep.py \
  --model gpt2 \
  --ratios 0.20 0.40 \
  --seed 42 \
  --force-remeasure
```

Single-run pruning scripts can also use a measurement cache explicitly with:

```text
--measurement-file PATH
```

---

# 8. Optional multi-seed experiments

The default workflow uses one seed:

```bash
--seed 42
```

Multiple seeds are also supported:

```bash
uv run python scripts/run_pruning_sweep.py \
  --model gpt2 \
  --methods similarity gradient random \
  --ratios 0.20 0.40 \
  --seeds 42 123 2026
```

`--seed` and `--seeds` are mutually exclusive.

For similarity and gradient pruning, the seed controls calibration-window sampling.

For random pruning, it also controls random head selection.

---

# 9. Selective LM-Harness evaluation

Running downstream evaluation at every pruning ratio can be expensive.

The sweep therefore supports:

```text
--harness-ratios
```

Default:

```text
0.20 0.40
```

Example:

```bash
uv run python scripts/run_pruning_sweep.py \
  --model gpt2 \
  --ratios 0.05 0.10 0.20 0.30 0.40 0.50 \
  --harness-ratios 0.20 0.40
```

In this example:

```text
0.05 -> perplexity only
0.10 -> perplexity only
0.20 -> perplexity + LM-Harness
0.30 -> perplexity only
0.40 -> perplexity + LM-Harness
0.50 -> perplexity only
```

The unpruned baseline is evaluated once with the selected LM-Harness tasks so that the downstream pruning results have a reference point.

To disable downstream evaluation completely:

```bash
--disable-lm-harness
```

---

# 10. Selecting LM-Harness tasks

The currently supported task list is:

```text
hellaswag
lambada
piqa
winogrande
arc_easy
arc_challenge
```

Run one task:

```bash
--harness-tasks hellaswag
```

Run multiple tasks:

```bash
--harness-tasks hellaswag piqa arc_easy
```

Run all configured tasks:

```bash
--harness-tasks all
```

Example full experiment:

```bash
uv run python scripts/run_pruning_sweep.py \
  --model gpt2 \
  --methods similarity gradient random \
  --ratios 0.05 0.10 0.20 0.30 0.40 0.50 \
  --seed 42 \
  --harness-ratios 0.20 0.40 \
  --harness-tasks hellaswag piqa arc_easy
```

---

# 11. Result format

New pruning results use schema version 2.

A result contains information such as:

```json
{
  "schema_version": 2,
  "result_type": "pruning",
  "model": "gpt2",
  "model_revision": null,
  "quantization": "none",
  "dataset": "wikitext-103-raw-v1",
  "evaluation_split": "test",
  "calibration_split": "validation",
  "pruning_method": "similarity",
  "requested_pruning_ratio": 0.20,
  "actual_pruning_ratio": 0.1666666667,
  "seed": 42,
  "eligible_layers": [],
  "selected_heads": {},
  "metrics": {
    "loss": 0.0,
    "perplexity": 0.0,
    "tokens": 0,
    "lm_harness": {}
  },
  "harness_enabled": true,
  "harness_tasks": [
    "hellaswag",
    "piqa",
    "arc_easy"
  ],
  "command": "...",
  "git_commit": "...",
  "timestamp": "..."
}
```

The exact command and Git commit are stored to improve reproducibility.

---

# 12. Result file naming

New result files include:

- requested ratio with four decimals
- seed

Example:

```text
similarity_gpt2_r0.2000_seed42.json
gradient_gpt2_r0.2000_seed42.json
random_gpt2_r0.2000_seed42.json
```

Pruning plans use the same convention:

```text
similarity_plan_gpt2_r0.2000_seed42.json
```

This avoids collisions between nearby ratios and different random seeds.

---

# 13. Single-run experiments

The original method-specific scripts remain available for debugging and focused experiments.

## Similarity

```bash
uv run python scripts/run_similarity_pruning.py \
  --model gpt2 \
  --ratio 0.20 \
  --seed 42 \
  --harness-tasks hellaswag piqa arc_easy
```

Using a cached measurement:

```bash
uv run python scripts/run_similarity_pruning.py \
  --model gpt2 \
  --ratio 0.20 \
  --seed 42 \
  --measurement-file configs/experiments/gpt2/measurements/head_redundancy_seed42.json \
  --disable-lm-harness
```

## Gradient importance

```bash
uv run python scripts/run_gradient_pruning.py \
  --model gpt2 \
  --ratio 0.20 \
  --seed 42 \
  --harness-tasks hellaswag piqa arc_easy
```

## Random

```bash
uv run python scripts/run_random_pruning.py \
  --model gpt2 \
  --ratio 0.20 \
  --seed 42 \
  --harness-tasks hellaswag piqa arc_easy
```

---

# 14. Experiment analysis

After running experiments:

```bash
uv run python scripts/analyze_experiments.py
```

Analysis outputs are written to:

```text
configs/experiments/analysis/
```

Main outputs include:

```text
pruning_summary.csv
similarity_vs_random.csv
similarity_vs_random_tasks.csv
recovery_summary.csv
similarity_vs_importance.csv
similarity_vs_importance_correlations.csv
```

and several plots.

---

## 14.1 Performance vs removal ratio

For every model, the analysis script generates:

```text
<model>_perplexity_vs_removal.png
<model>_relative_degradation.png
```

The x-axis uses the actual fraction of attention heads removed.

The main comparison is:

```text
similarity-guided
vs
gradient-importance
vs
random
```

---

## 14.2 Relative degradation

In addition to raw perplexity, the analysis computes:

```text
(pruned_perplexity - baseline_perplexity) / baseline_perplexity
```

This is useful when comparing models whose baseline perplexities differ substantially.

---

## 14.3 Similarity vs random

A dedicated report-ready file is created:

```text
similarity_vs_random.csv
```

Its central metric is:

```text
similarity_minus_random_perplexity
    = similarity_perplexity - random_perplexity
```

Interpretation:

```text
< 0  -> similarity-guided pruning has lower perplexity
= 0  -> equal perplexity
> 0  -> random pruning has lower perplexity
```

The CSV also contains:

- baseline perplexity
- similarity perplexity
- random perplexity
- absolute degradation from baseline
- relative degradation from baseline
- requested ratio
- actual ratio
- seed

This file is intended to provide a direct quantitative answer to:

> Does the measured head redundancy identify heads that are more safely removable than random heads?

---

## 14.4 Downstream task analysis

LM-Harness results are also aggregated.

The analysis creates:

```text
similarity_vs_random_tasks.csv
```

and per-task plots such as:

```text
gpt2_hellaswag_task_vs_removal.png
gpt2_piqa_task_vs_removal.png
gpt2_arc_easy_task_vs_removal.png
```

Only pruning ratios at which LM-Harness was actually run appear in these plots.

---

# 15. Similarity vs gradient importance

When both measurement caches are present, the analysis compares:

```text
head redundancy score
vs
gradient-based head importance
```

Outputs:

```text
similarity_vs_importance.csv
similarity_vs_importance_correlations.csv
<model>_seed<seed>_similarity_vs_importance.png
```

Both correlations are reported:

```text
Pearson r
Spearman rho
```

This provides an additional analysis question:

> Are representationally redundant heads also functionally unimportant?

---

# 16. Legacy result files

The repository may contain experiment files generated before schema version 2.

By default:

```bash
uv run python scripts/analyze_experiments.py
```

silently ignores incompatible legacy files.

To see which files are skipped:

```bash
uv run python scripts/analyze_experiments.py --verbose
```

---

# 17. Recovery after pruning

Recovery uses LoRA / QLoRA fine-tuning on a deterministic packed C4 subset.

WikiText is used for:

```text
unpruned evaluation
pruned evaluation
recovered evaluation
```

Example similarity recovery:

```bash
uv run python scripts/run_recovery.py \
  --model gpt2 \
  --pruning-method similarity \
  --ratio 0.20 \
  --seed 42 \
  --run-lm-harness \
  --harness-tasks hellaswag piqa arc_easy
```

Using a previously generated similarity measurement:

```bash
uv run python scripts/run_recovery.py \
  --model gpt2 \
  --pruning-method similarity \
  --ratio 0.20 \
  --seed 42 \
  --measurement-file configs/experiments/gpt2/measurements/head_redundancy_seed42.json
```

A second recovery point can be run at a more aggressive pruning ratio:

```bash
uv run python scripts/run_recovery.py \
  --model gpt2 \
  --pruning-method similarity \
  --ratio 0.40 \
  --seed 42 \
  --run-lm-harness \
  --harness-tasks hellaswag piqa arc_easy
```

Recovery outputs include:

- baseline loss and perplexity
- pruned loss and perplexity
- recovered loss and perplexity
- percentage of lost performance recovered
- optional LM-Harness results
- training history
- LoRA adapter
- pruning plan
- training metadata

The analysis script summarizes schema-v2 recovery runs into:

```text
recovery_summary.csv
```

---

# 18. Quantization

With:

```bash
--quantization auto
```

the current loader uses:

- GPT-2: normal model loading
- Qwen: 4-bit NF4
- Llama: 4-bit NF4

This keeps larger-model experiments feasible on limited GPU memory.

You can override the behavior with:

```text
--quantization none
--quantization 4bit
```

---

# 19. Qwen3.5 hybrid attention

Qwen3.5 hybrid models contain both standard full-attention layers and gated-delta linear-attention layers.

The current implementation measures and prunes **standard full-attention heads only**.

Linear-attention value heads are intentionally excluded because their semantics and dimensions differ from normal multi-head self-attention.

Therefore, for Qwen hybrid models:

> the reported pruning ratio is the fraction of heads removed among eligible standard full-attention heads.

---

# 20. Suggested end-to-end experiment workflow

A typical workflow is:

### Step 1 — Smoke test

```bash
uv run python scripts/run_pruning_sweep.py \
  --model gpt2 \
  --methods similarity gradient random \
  --ratios 0.20 \
  --seed 42 \
  --harness-ratios 0.20 \
  --harness-tasks piqa \
  --measurement-batches 2
```

### Step 2 — Full GPT-2 sweep

```bash
uv run python scripts/run_pruning_sweep.py \
  --model gpt2 \
  --methods similarity gradient random \
  --ratios 0.05 0.10 0.20 0.30 0.40 0.50 \
  --seed 42 \
  --harness-ratios 0.20 0.40 \
  --harness-tasks hellaswag piqa arc_easy
```

### Step 3 — Repeat for other model families

```bash
uv run python scripts/run_pruning_sweep.py \
  --model Qwen/Qwen3.5-4B \
  --methods similarity gradient random \
  --ratios 0.05 0.10 0.20 0.30 0.40 0.50 \
  --seed 42 \
  --harness-ratios 0.20 0.40 \
  --harness-tasks hellaswag piqa arc_easy
```

```bash
uv run python scripts/run_pruning_sweep.py \
  --model meta-llama/Llama-3.2-3B \
  --methods similarity gradient random \
  --ratios 0.05 0.10 0.20 0.30 0.40 0.50 \
  --seed 42 \
  --harness-ratios 0.20 0.40 \
  --harness-tasks hellaswag piqa arc_easy
```

### Step 4 — Analyze all experiment results

```bash
uv run python scripts/analyze_experiments.py
```

### Step 5 — Recovery experiments

For selected pruning ratios, for example 0.20 and 0.40:

```bash
uv run python scripts/run_recovery.py \
  --model gpt2 \
  --pruning-method similarity \
  --ratio 0.20 \
  --seed 42 \
  --run-lm-harness \
  --harness-tasks hellaswag piqa arc_easy
```

---

# 21. Repository structure

Relevant files in this branch:

```text
src/redundancy/
├── data.py
├── eval.py
├── experiment.py
├── hooks.py
├── measurement.py
├── models.py
├── plotting.py
├── recovery.py
├── utils.py
└── pruning/
    ├── gradient_pruning.py
    ├── plan.py
    ├── random_pruning.py
    └── similarity_pruning.py

scripts/
├── analyze_experiments.py
├── measure_redundancy.py
├── run_baseline.py
├── run_gradient_pruning.py
├── run_pruning_sweep.py
├── run_random_pruning.py
├── run_recovery.py
└── run_similarity_pruning.py
```

---

# 22. Reproducibility

New experiment results record:

- model identifier
- optional model revision
- quantization mode
- dataset
- calibration split
- evaluation split
- requested pruning ratio
- actual pruning ratio
- pruning method
- random seed
- selected heads
- eligible layers
- selected LM-Harness tasks
- full invocation command
- Git commit
- timestamp

For final experiments, fixing the model revision explicitly is recommended when possible:

```bash
--model-revision <revision-or-commit>
```

This makes later reproduction less dependent on changes to upstream Hugging Face repositories.

---

## Summary

The intended experimental logic of this branch is:

```text
Measure head redundancy
        ↓
Locate redundancy across heads and layers
        ↓
Prune heads at increasing removal ratios
        ↓
Compare similarity-guided pruning
against gradient importance and random pruning
        ↓
Evaluate language-modeling and downstream performance
        ↓
Quantify similarity-vs-random differences
        ↓
Test selected recovery scenarios
        ↓
Analyze model, task, depth, and recovery dimensions
```
