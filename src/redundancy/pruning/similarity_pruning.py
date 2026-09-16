"""Attention-head output similarity measurement and pruning."""

from __future__ import annotations

import torch

from redundancy.hooks import HeadOutputStatsHook
from redundancy.pruning.plan import PruningPlan, apply_pruning_plan
from redundancy.utils import get_model_metadata, get_output_projection


def compute_centered_cosine_similarity(
    collector: HeadOutputStatsHook,
    absolute: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute pairwise centered-cosine similarity between attention-head outputs.

    The calculation uses sufficient statistics collected by HeadOutputStatsHook,
    so the full activation traces do not need to be stored in memory.
    """

    if collector.count == 0:
        raise ValueError("No head-output statistics were collected")

    count = float(collector.count)

    # sum((x_i - mean_i) * (x_j - mean_j))
    centered_cross = (
        collector.cross
        - torch.outer(collector.sum, collector.sum) / count
    )

    # sum((x_i - mean_i)^2)
    centered_sq_norms = (
        collector.sum_sq
        - (collector.sum ** 2) / count
    ).clamp_min(0.0)

    denominator = torch.sqrt(
        torch.outer(
            centered_sq_norms,
            centered_sq_norms,
        )
    ).clamp_min(eps)

    similarity = centered_cross / denominator

    if absolute:
        similarity = similarity.abs()
        similarity = similarity.clamp(0.0, 1.0)
    else:
        similarity = similarity.clamp(-1.0, 1.0)

    # Do not allow self-similarity to influence pruning.
    similarity.fill_diagonal_(0.0)

    return similarity


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
    """
    Collect head-output activations and compute one pairwise similarity
    matrix per eligible transformer layer.

    Returns:
        dict[int, torch.Tensor]:
            layer -> [num_heads, num_heads] similarity matrix
    """

    if num_batches <= 0:
        raise ValueError("num_batches must be positive")

    model_metadata = get_model_metadata(model)

    collectors = {
        layer: HeadOutputStatsHook(
            num_heads=model_metadata.num_heads,
            head_dim=model_metadata.head_dim,
        )
        for layer in model_metadata.eligible_layers
    }

    handles = [
        get_output_projection(
            model,
            layer,
            model_metadata.model_type,
        ).register_forward_pre_hook(
            collectors[layer]
        )
        for layer in model_metadata.eligible_layers
    ]

    was_training = model.training
    model.eval()

    texts = [
        text
        for text in dataset["text"]
        if text and text.strip()
    ]

    encodings = tokenizer(
        "\n\n".join(texts),
        return_tensors="pt",
    )

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

    generator = torch.Generator().manual_seed(seed)

    starts = torch.randint(
        0,
        max_start + 1,
        (num_batches,),
        generator=generator,
    )

    try:
        with torch.no_grad():
            for start in starts.tolist():
                input_ids = encodings.input_ids[
                    :,
                    start : start + max_length,
                ].to(device)

                model(input_ids=input_ids)

    finally:
        for handle in handles:
            handle.remove()

        model.train(was_training)

    similarities = {
        layer: compute_centered_cosine_similarity(
            collector,
            absolute=absolute,
        )
        for layer, collector in collectors.items()
    }

    return similarities


def select_redundant_heads(
    similarity: torch.Tensor,
    num_to_prune: int,
) -> list[int]:
    """
    Greedily select redundant heads from a pairwise similarity matrix.

    At each step:
    1. Find the most similar pair of currently active heads.
    2. Compare the two heads' average similarities to the remaining heads.
    3. Remove the head that is, on average, more similar to the rest.
    4. Repeat until the requested number of heads has been selected.

    This avoids simply ranking heads independently, which could cause
    multiple heads from the same redundancy group to be removed at once.
    """

    if similarity.ndim != 2:
        raise ValueError("similarity must be a 2D matrix")

    num_heads = similarity.shape[0]

    if similarity.shape[1] != num_heads:
        raise ValueError("similarity must be a square matrix")

    if not 0 <= num_to_prune <= num_heads:
        raise ValueError(
            "num_to_prune must be between 0 and num_heads"
        )

    if num_to_prune == 0:
        return []

    active_heads = list(range(num_heads))
    selected = []

    while (
        len(selected) < num_to_prune
        and len(active_heads) > 1
    ):
        active_similarity = similarity[
            active_heads
        ][:, active_heads].clone()

        # Prevent a head from being paired with itself.
        active_similarity.fill_diagonal_(-1.0)

        flat_index = torch.argmax(
            active_similarity
        ).item()

        row = flat_index // len(active_heads)
        col = flat_index % len(active_heads)

        head_a = active_heads[row]
        head_b = active_heads[col]

        others_a = [
            head
            for head in active_heads
            if head != head_a
        ]

        others_b = [
            head
            for head in active_heads
            if head != head_b
        ]

        mean_similarity_a = similarity[
            head_a,
            others_a,
        ].mean()

        mean_similarity_b = similarity[
            head_b,
            others_b,
        ].mean()

        if mean_similarity_a > mean_similarity_b:
            prune_head = head_a

        elif mean_similarity_b > mean_similarity_a:
            prune_head = head_b

        else:
            # Deterministic tie-breaking.
            prune_head = max(
                head_a,
                head_b,
            )

        selected.append(prune_head)
        active_heads.remove(prune_head)

    # Only relevant when pruning 100% of the heads.
    while len(selected) < num_to_prune:
        selected.append(
            active_heads.pop()
        )

    return sorted(selected)


def select_similarity_pruning_plan(
    model,
    ratio: float,
    tokenizer,
    dataset,
    device,
    num_batches: int = 16,
    max_length: int = 512,
    seed: int = 42,
    absolute: bool = True,
    model_name: str | None = None,
    model_revision: str | None = None,
    **_,
) -> PruningPlan:
    """
    Create a layer-wise similarity-guided attention-head pruning plan.

    The same number of heads is pruned from every eligible layer,
    matching the current random and gradient pruning implementations.
    """

    if not 0 <= ratio <= 1:
        raise ValueError(
            "ratio must be between 0 and 1"
        )

    model_metadata = get_model_metadata(model)

    heads_per_layer = (
        max(
            1,
            int(
                model_metadata.num_heads
                * ratio
            ),
        )
        if ratio > 0
        else 0
    )

    similarities = None

    if heads_per_layer:
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

        selected_heads = {
            layer: select_redundant_heads(
                similarity=similarity,
                num_to_prune=heads_per_layer,
            )
            for layer, similarity in similarities.items()
        }

    else:
        selected_heads = {
            layer: []
            for layer in model_metadata.eligible_layers
        }

    return PruningPlan(
        method="similarity",
        model_name=(
            model_name
            or getattr(
                model.config,
                "_name_or_path",
                type(model).__name__,
            )
        ),
        model_revision=model_revision,
        requested_ratio=ratio,
        actual_ratio=(
            heads_per_layer
            / model_metadata.num_heads
        ),
        seed=seed,
        eligible_layers=list(
            model_metadata.eligible_layers
        ),
        selected_heads=selected_heads,
        num_heads=model_metadata.num_heads,
        head_dim=model_metadata.head_dim,
        metadata={
            "similarity_batches": num_batches,
            "similarity_max_length": max_length,
            "absolute_similarity": absolute,
            "selection_strategy": "greedy_pairwise",
            "similarity_matrices": (
                {
                    str(layer): matrix.tolist()
                    for layer, matrix
                    in similarities.items()
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
):
    """
    Backward-compatible helper that creates and immediately applies
    a similarity-guided pruning plan.
    """

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
    )

    handles = apply_pruning_plan(
        model,
        plan,
    )

    return handles, plan


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