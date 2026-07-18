"""
Gradient-based attention head importance & pruning.

Method: first-order Taylor expansion of the loss w.r.t. a per-head mask
variable (Michel et al., "Are Sixteen Heads Really Better than One?", 2019).

Heads with lowest importance are the most redundant. Pruning selects bottom-k
heads per layer.
"""

import torch

from redundancy.hooks import HeadPruningHook
from redundancy.pruning.random_pruning import get_model_meta, get_output_projection


class HeadImportanceMaskHook:
    """
    use only during importance scoring.
    """

    def __init__(self, mask: torch.Tensor, head_dim: int):
        self.mask = mask  # shape: (num_heads,), requires_grad=True
        self.head_dim = head_dim

    def __call__(self, module, args):
        hidden_states = args[0]
        expanded_mask = self.mask.repeat_interleave(self.head_dim)
        expanded_mask = expanded_mask.view(*([1] * (hidden_states.dim() - 1)), -1)
        return (hidden_states * expanded_mask,)


@torch.no_grad()
def _set_requires_grad(model, flag: bool):
    for p in model.parameters():
        p.requires_grad_(flag)


def compute_head_importance(
    model,
    tokenizer,
    dataset,
    device,
    num_batches: int = 16,
    max_length: int = 512,
    seed: int = 42,
):
    """
    Runs num_batches forward+backward passes and accumulates
    |d loss / d mask_h| per (layer, head).

    Returns a tensor of shape (num_layers, num_heads) 
    """
    num_layers, num_heads, head_dim, model_type = get_model_meta(model)

    masks = [
        torch.ones(num_heads, device=device, requires_grad=True) for _ in range(num_layers)
    ]

    hooks = []
    for layer_idx in range(num_layers):
        proj_layer = get_output_projection(model, layer_idx, model_type)
        hook = proj_layer.register_forward_pre_hook(
            HeadImportanceMaskHook(masks[layer_idx], head_dim)
        )
        hooks.append(hook)

    _set_requires_grad(model, False)
    was_training = model.training
    model.eval()

    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    if seq_len <= max_length:
        raise ValueError(
            f"Dataset too short ({seq_len} tokens) for max_length={max_length}."
        )

    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, seq_len - max_length, (num_batches,), generator=generator)

    importance = torch.zeros(num_layers, num_heads, device=device)
    used_batches = 0

    for start in starts.tolist():
        input_ids = encodings.input_ids[:, start:start + max_length].to(device)
        target_ids = input_ids.clone()

        for m in masks:
            if m.grad is not None:
                m.grad.zero_()

        outputs = model(input_ids, labels=target_ids)
        outputs.loss.backward()

        for layer_idx, m in enumerate(masks):
            importance[layer_idx] += m.grad.detach().abs()

        used_batches += 1

    importance /= max(used_batches, 1)

    for hook in hooks:
        hook.remove()

    _set_requires_grad(model, True)
    model.train(was_training)

    return importance.cpu()


def gradient_prune_model(
    model,
    tokenizer,
    dataset,
    device,
    sparsity: float,
    num_batches: int = 16,
    max_length: int = 512,
    seed: int = 42,
):
    num_layers, num_heads, head_dim, model_type = get_model_meta(model)

    k = max(1, int(num_heads * sparsity)) if sparsity > 0 else 0
    if k == 0:
        print(f"No heads to prune for sparsity {sparsity}. Returning without pruning.")
        return [], None

    importance = compute_head_importance(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        num_batches=num_batches,
        max_length=max_length,
        seed=seed,
    )

    hooks = []
    for layer_idx in range(num_layers):
        layer_scores = importance[layer_idx]
        # Lowest gradient importance == most redundant head.
        heads_to_mask = torch.argsort(layer_scores)[:k].tolist()

        proj_layer = get_output_projection(model, layer_idx, model_type)
        hook = proj_layer.register_forward_pre_hook(HeadPruningHook(heads_to_mask, head_dim))
        hooks.append(hook)

    actual_sparsity = (k / num_heads) * 100
    print(
        f"Successfully masked {k}/{num_heads} heads ({actual_sparsity:.1f}%) per layer "
        f"in {model_type} using gradient importance."
    )
    return hooks, importance


def prune_model(model, tokenizer, dataset, device, sparsity, num_batches=16, seed=42):
    hooks, _ = gradient_prune_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        sparsity=sparsity,
        num_batches=num_batches,
        seed=seed,
    )
    return hooks