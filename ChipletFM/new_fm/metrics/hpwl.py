from __future__ import annotations

import torch


def graph_hpwl(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros((), dtype=pos.dtype, device=pos.device)
    src, dst = _unique_undirected_edges(edge_index)
    delta = (pos[src] - pos[dst]).abs()
    return delta.sum()


def _unique_undirected_edges(edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    src = edge_index[0].long()
    dst = edge_index[1].long()
    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)
    pairs = torch.stack([lo, hi], dim=1)
    pairs = pairs[lo != hi]
    if pairs.numel() == 0:
        return src[:0], dst[:0]
    pairs = torch.unique(pairs, dim=0)
    return pairs[:, 0], pairs[:, 1]

