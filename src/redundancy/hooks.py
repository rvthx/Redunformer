import torch


class HeadPruningHook:
    def __init__(self, head_indices, head_dim):
        self.head_indices = tuple(sorted(set(head_indices)))
        self.head_dim = head_dim

    def __call__(self, module, args):
        hidden_states = args[0]
        feature_mask = torch.ones(
            hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype
        )
        for head in self.head_indices:
            start = head * self.head_dim
            feature_mask[start : start + self.head_dim] = 0
        return (hidden_states * feature_mask, *args[1:])


class RandomHeadPruningHook(HeadPruningHook):
    pass


class HeadImportanceMaskHook:
    def __init__(self, mask: torch.Tensor, head_dim: int):
        self.mask = mask
        self.head_dim = head_dim

    def __call__(self, module, args):
        hidden_states = args[0]
        expanded_mask = self.mask.repeat_interleave(self.head_dim).to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        expanded_mask = expanded_mask.view(*([1] * (hidden_states.dim() - 1)), -1)
        return (hidden_states * expanded_mask, *args[1:])


class HeadOutputStatsHook:
    """Legacy global-flatten sufficient statistics for head-output similarity."""

    def __init__(self, num_heads: int, head_dim: int):
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.count = 0
        self.sum = torch.zeros(num_heads, dtype=torch.float64)
        self.sum_sq = torch.zeros(num_heads, dtype=torch.float64)
        self.cross = torch.zeros(num_heads, num_heads, dtype=torch.float64)

    def __call__(self, module, args):
        hidden_states = args[0].detach()

        head_outputs = hidden_states.reshape(
            -1, self.num_heads, self.head_dim
        )

        head_outputs = (
            head_outputs
            .permute(1, 0, 2)
            .reshape(self.num_heads, -1)
            .float()
        )

        self.count += head_outputs.shape[1]
        self.sum += head_outputs.sum(dim=1).cpu().double()
        self.sum_sq += (head_outputs ** 2).sum(dim=1).cpu().double()
        self.cross += (head_outputs @ head_outputs.T).cpu().double()


class PerChannelHeadOutputStatsHook:
    """
    Collect per-channel sufficient statistics for pairwise head similarity.

    Each attention head is kept as [tokens, head_dim]. Statistics are tracked
    independently for every (head, channel), so later correlation computation
    can standardize each channel over tokens before aggregating across channels.
    """

    def __init__(self, num_heads: int, head_dim: int):
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Number of token observations per channel.
        self.count = 0

        # [heads, head_dim]
        self.sum = torch.zeros(
            num_heads, head_dim, dtype=torch.float64
        )
        self.sum_sq = torch.zeros(
            num_heads, head_dim, dtype=torch.float64
        )

        # cross[h, j, d] = sum_t x[t, h, d] * x[t, j, d]
        self.cross = torch.zeros(
            num_heads, num_heads, head_dim, dtype=torch.float64
        )

    def __call__(self, module, args):
        hidden_states = args[0].detach()

        # [batch, seq, hidden]
        # -> [tokens, heads, head_dim]
        head_outputs = hidden_states.reshape(
            -1, self.num_heads, self.head_dim
        ).float()

        self.count += head_outputs.shape[0]

        self.sum += head_outputs.sum(dim=0).cpu().double()
        self.sum_sq += (head_outputs ** 2).sum(dim=0).cpu().double()

        # Keep channels separate. This has the same asymptotic arithmetic cost
        # as the previous global H x (tokens*D) matrix product, but preserves
        # the channel axis for per-channel standardization.
        self.cross += torch.einsum(
            "thd,tjd->hjd",
            head_outputs,
            head_outputs,
        ).cpu().double()
