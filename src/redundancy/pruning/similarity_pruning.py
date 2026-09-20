"""Attention-head output similarity measurement and pruning."""

from __future__ import annotations

import torch

from redundancy.hooks import PerChannelHeadOutputStatsHook
from redundancy.pruning.plan import PruningPlan, apply_pruning_plan
from redundancy.utils import get_model_metadata, get_output_projection


SIMILARITY_DEFINITION = "per_channel_standardized_correlation"


def compute_per_channel_standardized_similarity(
    collector: PerChannelHeadOutputStatsHook,
    absolute: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute pairwise head similarity after per-channel standardization.

    For each head h and channel d, activations are centered/scaled over tokens.
    Pairwise Pearson correlations are then computed independently for each
    channel and averaged with equal weight across valid channels.

    This prevents a small number of high-variance channels from dominating the
    similarity score, which can happen when [tokens, head_dim] is flattened and
    normalized with a single global scalar mean/variance.
    """

    if collector.count == 0:
        raise ValueError("No head-output statistics were collected")

    count = float(collector.count)

    # [heads, heads, head_dim]
    centered_cross = (
        collector.cross
        - (
            collector.sum[:, None, :]
            * collector.sum[None, :, :]
        ) / count
    )

    # [heads, head_dim]
    centered_sq_norms = (
        collector.sum_sq
        - (collector.sum ** 2) / count
    ).clamp_min(0.0)

    # Pairwise denominator for each channel.
    denominator = torch.sqrt(
        centered_sq_norms[:, None, :]
        * centered_sq_norms[None, :, :]
    )

    valid = denominator > eps
    channel_correlation = torch.zeros_like(centered_cross)
    channel_correlation[valid] = (
        centered_cross[valid] / denominator[valid]
    )
    channel_correlation = channel_correlation.clamp(-1.0, 1.0)

    # Equal weighting across channels. Constant/near-constant channels are
    # excluded pairwise rather than being allowed to create numerical noise.
    valid_counts = valid.sum(dim=-1)
    similarity = torch.zeros(
        collector.num_heads,
        collector.num_heads,
        dtype=torch.float64,
    )
    nonempty = valid_counts > 0
    similarity[nonempty] = (
        (channel_correlation * valid).sum(dim=-1)[nonempty]
        / valid_counts[nonempty]
    )

    # Preserve the previous absolute/signed semantics: first aggregate the
    # standardized channel correlations, then optionally discard the sign.
    if absolute:
        similarity = similarity.abs().clamp(0.0, 1.0)
    else:
        similarity = similarity.clamp(-1.0, 1.0)

    similarity.fill_diagonal_(0.0)
    return similarity


# Backward-compatible public name used elsewhere in the codebase. On this
# branch it intentionally implements the per-channel standardized definition.
def compute_centered_cosine_similarity(
    collector: PerChannelHeadOutputStatsHook,
    absolute: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    return compute_per_channel_standardized_similarity(
        collector=collector,
        absolute=absolute,
        eps=eps,
    )


def compute_head_similarities(
    model,
    tokenizer,
    dataset,
    device,
    num_batches: int = 16,
    max_length: int = 512,
    seed: int = 42,
    absolute: bool = True,
):
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")

    model_metadata = get_model_metadata(model)
    collectors = {
        layer: PerChannelHeadOutputStatsHook(
            model_metadata.num_heads,
            model_metadata.head_dim,
        )
        for layer in model_metadata.eligible_layers
    }
    handles = [
        get_output_projection(
            model,
            layer,
            model_metadata.model_type,
        ).register_forward_pre_hook(collectors[layer])
        for layer in model_metadata.eligible_layers
    ]

    was_training = model.training
    model.eval()
    texts = [text for text in dataset["text"] if text and text.strip()]
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
    sequence_length = encodings.input_ids.size(1)

    if sequence_length < max_length:
        for handle in handles:
            handle.remove()
        model.train(was_training)
        raise ValueError(
            f"Dataset is too short ({sequence_length} tokens) "
            f"for max_length={max_length}"
        )

    max_start = sequence_length - max_length
    starts = torch.randint(
        0,
        max_start + 1,
        (num_batches,),
        generator=torch.Generator().manual_seed(seed),
    )

    try:
        with torch.no_grad():
            for start in starts.tolist():
                input_ids = encodings.input_ids[
                    :, start : start + max_length
                ].to(device)
                model(input_ids=input_ids)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return {
        layer: compute_per_channel_standardized_similarity(
            collector,
            absolute=absolute,
        )
        for layer, collector in collectors.items()
    }


def select_redundant_heads(
    similarity: torch.Tensor,
    num_to_prune: int,
) -> list[int]:
    if similarity.ndim != 2:
        raise ValueError("similarity must be a 2D matrix")
    num_heads = similarity.shape[0]
    if similarity.shape[1] != num_heads:
        raise ValueError("similarity must be a square matrix")
    if not 0 <= num_to_prune <= num_heads:
        raise ValueError("num_to_prune must be between 0 and num_heads")
    if num_to_prune == 0:
        return []

    active_heads = list(range(num_heads))
    selected: list[int] = []

    while len(selected) < num_to_prune and len(active_heads) > 1:
        active_similarity = similarity[active_heads][:, active_heads].clone()
        active_similarity.fill_diagonal_(-1.0)

        flat_index = torch.argmax(active_similarity).item()
        row = flat_index // len(active_heads)
        col = flat_index % len(active_heads)
        head_a = active_heads[row]
        head_b = active_heads[col]

        others_a = [head for head in active_heads if head != head_a]
        others_b = [head for head in active_heads if head != head_b]
        mean_similarity_a = similarity[head_a, others_a].mean()
        mean_similarity_b = similarity[head_b, others_b].mean()

        if mean_similarity_a > mean_similarity_b:
            prune_head = head_a
        elif mean_similarity_b > mean_similarity_a:
            prune_head = head_b
        else:
            prune_head = max(head_a, head_b)

        selected.append(prune_head)
        active_heads.remove(prune_head)

    while len(selected) < num_to_prune:
        selected.append(active_heads.pop())

    return sorted(selected)


def select_similarity_pruning_plan(
    model,
    ratio: float,
    tokenizer=None,
    dataset=None,
    device=None,
    num_batches: int = 16,
    max_length: int = 512,
    seed: int = 42,
    absolute: bool = True,
    model_name: str | None = None,
    model_revision: str | None = None,
    similarity_matrices: dict[int, torch.Tensor] | None = None,
    **_,
) -> PruningPlan:
    if not 0 <= ratio <= 1:
        raise ValueError("ratio must be between 0 and 1")

    model_metadata = get_model_metadata(model)
    heads_per_layer = (
        max(1, int(model_metadata.num_heads * ratio))
        if ratio > 0
        else 0
    )

    similarities = similarity_matrices
    if heads_per_layer and similarities is None:
        if tokenizer is None or dataset is None or device is None:
            raise ValueError(
                "tokenizer, dataset, and device are required "
                "when no similarity cache is supplied"
            )
        similarities = compute_head_similarities(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            device=device,
            num_batches=num_batches,
            max_length=max_length,
            seed=seed,
            absolute=absolute,
        )

    if heads_per_layer:
        selected_heads = {
            layer: select_redundant_heads(
                similarity,
                heads_per_layer,
            )
            for layer, similarity in similarities.items()
        }
    else:
        selected_heads = {
            layer: [] for layer in model_metadata.eligible_layers
        }

    return PruningPlan(
        method="similarity",
        model_name=model_name
        or getattr(model.config, "_name_or_path", type(model).__name__),
        model_revision=model_revision,
        requested_ratio=ratio,
        actual_ratio=heads_per_layer / model_metadata.num_heads,
        seed=seed,
        eligible_layers=list(model_metadata.eligible_layers),
        selected_heads=selected_heads,
        num_heads=model_metadata.num_heads,
        head_dim=model_metadata.head_dim,
        metadata={
            "similarity_batches": num_batches,
            "similarity_max_length": max_length,
            "absolute_similarity": absolute,
            "similarity_definition": SIMILARITY_DEFINITION,
            "channel_normalization": "per_head_per_channel_over_tokens",
            "channel_aggregation": "equal_weight_mean_correlation",
            "selection_strategy": "greedy_pairwise",
            "used_cached_similarity": similarity_matrices is not None,
            "similarity_matrices": (
                {
                    str(layer): matrix.tolist()
                    for layer, matrix in similarities.items()
                }
                if similarities is not None
                else None
            ),
        },
    )


def similarity_prune_model(
    model,
    tokenizer,
    dataset,
    device,
    sparsity: float,
    num_batches: int = 16,
    max_length: int = 512,
    seed: int = 42,
    absolute: bool = True,
    similarity_matrices: dict[int, torch.Tensor] | None = None,
):
    plan = select_similarity_pruning_plan(
        model=model,
        ratio=sparsity,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        num_batches=num_batches,
        max_length=max_length,
        seed=seed,
        absolute=absolute,
        similarity_matrices=similarity_matrices,
    )
    return apply_pruning_plan(model, plan), plan


def prune_model(
    model,
    tokenizer,
    dataset,
    device,
    sparsity,
    num_batches=16,
    seed=42,
):
    return similarity_prune_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        sparsity=sparsity,
        num_batches=num_batches,
        seed=seed,
    )
