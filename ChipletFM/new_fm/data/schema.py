from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class Canvas:
    origin: torch.Tensor
    size: torch.Tensor


@dataclass
class LayoutSample:
    sample_id: str
    target_pos: torch.Tensor
    size: torch.Tensor
    node_feat: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    movable_mask: torch.Tensor
    canvas: Canvas


@dataclass
class LayoutBatch:
    sample_id: List[str]
    target_pos: torch.Tensor
    size: torch.Tensor
    node_feat: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    movable_mask: torch.Tensor
    batch: torch.Tensor
    ptr: torch.Tensor
    canvas_origin: torch.Tensor
    canvas_size: torch.Tensor

    def to(self, device: torch.device | str) -> "LayoutBatch":
        fields = {
            "target_pos": self.target_pos.to(device),
            "size": self.size.to(device),
            "node_feat": self.node_feat.to(device),
            "edge_index": self.edge_index.to(device),
            "edge_attr": self.edge_attr.to(device),
            "movable_mask": self.movable_mask.to(device),
            "batch": self.batch.to(device),
            "ptr": self.ptr.to(device),
            "canvas_origin": self.canvas_origin.to(device),
            "canvas_size": self.canvas_size.to(device),
        }
        return LayoutBatch(sample_id=self.sample_id, **fields)

    def repeat_single(self, repeats: int) -> "LayoutBatch":
        if len(self.sample_id) != 1:
            raise ValueError("repeat_single expects a batch with exactly one graph")
        target_pos = self.target_pos.unsqueeze(0).expand(repeats, -1, -1).reshape(-1, 2)
        size = self.size.unsqueeze(0).expand(repeats, -1, -1).reshape(-1, 2)
        node_feat = self.node_feat.unsqueeze(0).expand(repeats, -1, -1).reshape(
            -1, self.node_feat.shape[1]
        )
        movable_mask = self.movable_mask.unsqueeze(0).expand(repeats, -1).reshape(-1)
        n = self.target_pos.shape[0]
        e = self.edge_index.shape[1]
        offsets = torch.arange(repeats, device=self.edge_index.device).view(repeats, 1, 1) * n
        edge_index = (
            self.edge_index.view(1, 2, e).expand(repeats, -1, -1) + offsets
        ).permute(1, 0, 2).reshape(2, repeats * e)
        edge_attr = self.edge_attr.unsqueeze(0).expand(repeats, -1, -1).reshape(
            repeats * e, self.edge_attr.shape[1]
        )
        batch = torch.arange(repeats, device=self.target_pos.device).repeat_interleave(n)
        ptr = torch.arange(repeats + 1, device=self.target_pos.device, dtype=torch.long) * n
        return LayoutBatch(
            sample_id=[self.sample_id[0] for _ in range(repeats)],
            target_pos=target_pos,
            size=size,
            node_feat=node_feat,
            edge_index=edge_index.long(),
            edge_attr=edge_attr,
            movable_mask=movable_mask,
            batch=batch.long(),
            ptr=ptr,
            canvas_origin=self.canvas_origin.expand(repeats, -1),
            canvas_size=self.canvas_size.expand(repeats, -1),
        )


def collate_layout_samples(samples: List[LayoutSample]) -> LayoutBatch:
    node_counts = [s.target_pos.shape[0] for s in samples]
    offsets = torch.tensor([0] + node_counts[:-1], dtype=torch.long).cumsum(0)

    edge_parts = []
    for sample, offset in zip(samples, offsets):
        edge_parts.append(sample.edge_index.long() + int(offset))

    batch_index = [
        torch.full((count,), i, dtype=torch.long) for i, count in enumerate(node_counts)
    ]
    ptr = torch.tensor([0] + node_counts, dtype=torch.long).cumsum(0)

    return LayoutBatch(
        sample_id=[s.sample_id for s in samples],
        target_pos=torch.cat([s.target_pos for s in samples], dim=0).float(),
        size=torch.cat([s.size for s in samples], dim=0).float(),
        node_feat=torch.cat([s.node_feat for s in samples], dim=0).float(),
        edge_index=torch.cat(edge_parts, dim=1).long()
        if edge_parts
        else torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.cat([s.edge_attr for s in samples], dim=0).float()
        if samples
        else torch.empty((0, 0), dtype=torch.float32),
        movable_mask=torch.cat([s.movable_mask for s in samples], dim=0).bool(),
        batch=torch.cat(batch_index, dim=0),
        ptr=ptr,
        canvas_origin=torch.stack([s.canvas.origin for s in samples], dim=0).float(),
        canvas_size=torch.stack([s.canvas.size for s in samples], dim=0).float(),
    )


def batch_to_sample_slices(batch: LayoutBatch) -> Dict[str, slice]:
    return {
        sample_id: slice(int(batch.ptr[i]), int(batch.ptr[i + 1]))
        for i, sample_id in enumerate(batch.sample_id)
    }
