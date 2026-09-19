"""First-order Taylor attention-head importance and pruning."""

from __future__ import annotations

import torch

from redundancy.hooks import HeadImportanceMaskHook
from redundancy.pruning.plan import PruningPlan, apply_pruning_plan
from redundancy.utils import get_model_metadata, get_output_projection


def compute_head_importance(
    model,
    tokenizer,
    dataset,
    device,
    num_batches: int = 16,
    max_length: int = 512,
    seed: int = 42,
):
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    model_metadata = get_model_metadata(model)
    masks = {
        layer: torch.ones(model_metadata.num_heads, device=device, requires_grad=True)
        for layer in model_metadata.eligible_layers
    }
    handles = [
        get_output_projection(model, layer, model_metadata.model_type).register_forward_pre_hook(
            HeadImportanceMaskHook(masks[layer], model_metadata.head_dim)
        )
        for layer in model_metadata.eligible_layers
    ]

    parameter_flags = [parameter.requires_grad for parameter in model.parameters()]
    was_training = model.training
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    texts = [text for text in dataset["text"] if text and text.strip()]
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
    sequence_length = encodings.input_ids.size(1)
    if sequence_length < max_length:
        for handle in handles:
            handle.remove()
        for parameter, flag in zip(model.parameters(), parameter_flags):
            parameter.requires_grad_(flag)
        model.train(was_training)
        raise ValueError(
            f"Dataset is too short ({sequence_length} tokens) for max_length={max_length}"
        )

    max_start = sequence_length - max_length
    starts = torch.randint(
        0,
        max_start + 1,
        (num_batches,),
        generator=torch.Generator().manual_seed(seed),
    )
    importance = {
        layer: torch.zeros(model_metadata.num_heads, device=device)
        for layer in model_metadata.eligible_layers
    }

    try:
        for start in starts.tolist():
            input_ids = encodings.input_ids[:, start : start + max_length].to(device)
            for mask in masks.values():
                if mask.grad is not None:
                    mask.grad.zero_()
            model(input_ids=input_ids, labels=input_ids).loss.backward()
            for layer, mask in masks.items():
                if mask.grad is None:
                    raise RuntimeError(f"No importance gradient was produced for layer {layer}")
                importance[layer] += mask.grad.detach().abs()
        for layer in importance:
            importance[layer] /= num_batches
    finally:
        for handle in handles:
            handle.remove()
        for parameter, flag in zip(model.parameters(), parameter_flags):
            parameter.requires_grad_(flag)
        model.train(was_training)

    return {layer: scores.cpu() for layer, scores in importance.items()}


def select_gradient_pruning_plan(
    model,
    ratio: float,
    tokenizer=None,
    dataset=None,
    device=None,
    num_batches: int = 16,
    max_length: int = 512,
    seed: int = 42,
    model_name: str | None = None,
    model_revision: str | None = None,
    importance_scores: dict[int, torch.Tensor] | None = None,
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

    importance = importance_scores
    if heads_per_layer and importance is None:
        if tokenizer is None or dataset is None or device is None:
            raise ValueError(
                "tokenizer, dataset, and device are required when no importance cache is supplied"
            )
        importance = compute_head_importance(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            device=device,
            num_batches=num_batches,
            max_length=max_length,
            seed=seed,
        )

    if heads_per_layer:
        selected_heads = {
            layer: torch.argsort(scores)[:heads_per_layer].tolist()
            for layer, scores in importance.items()
        }
    else:
        selected_heads = {
            layer: [] for layer in model_metadata.eligible_layers
        }

    return PruningPlan(
        method="gradient",
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
            "importance_batches": num_batches,
            "importance_max_length": max_length,
            "used_cached_importance": importance_scores is not None,
            "head_importance": (
                {str(layer): scores.tolist() for layer, scores in importance.items()}
                if importance is not None
                else None
            ),
        },
    )


def gradient_prune_model(
    model,
    tokenizer,
    dataset,
    device,
    sparsity: float,
    num_batches: int = 16,
    max_length: int = 512,
    seed: int = 42,
    importance_scores: dict[int, torch.Tensor] | None = None,
):
    plan = select_gradient_pruning_plan(
        model=model,
        ratio=sparsity,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        num_batches=num_batches,
        max_length=max_length,
        seed=seed,
        importance_scores=importance_scores,
    )
    handles = apply_pruning_plan(model, plan)
    importance = plan.metadata.get("head_importance")
    importance_tensor = (
        torch.tensor([importance[str(layer)] for layer in plan.eligible_layers])
        if importance is not None
        else None
    )
    return handles, importance_tensor


def prune_model(model, tokenizer, dataset, device, sparsity, num_batches=16, seed=42):
    return gradient_prune_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        sparsity=sparsity,
        num_batches=num_batches,
        seed=seed,
    )
