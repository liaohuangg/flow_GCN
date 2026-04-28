from __future__ import annotations

import torch


def legality_score(
    center_pos: torch.Tensor,
    size: torch.Tensor,
    canvas_origin: torch.Tensor,
    canvas_size: torch.Tensor,
) -> torch.Tensor:
    total_area = (size[:, 0] * size[:, 1]).sum().clamp_min(1e-6)
    overlap = overlap_area(center_pos, size)
    boundary = boundary_violation_area(center_pos, size, canvas_origin, canvas_size)
    penalty = (overlap + boundary) / total_area
    return (1.0 - penalty).clamp(0.0, 1.0)


def overlap_area(center_pos: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    if center_pos.shape[0] < 2:
        return torch.zeros((), dtype=center_pos.dtype, device=center_pos.device)
    lo = center_pos - size / 2.0
    hi = center_pos + size / 2.0
    total = torch.zeros((), dtype=center_pos.dtype, device=center_pos.device)
    for i in range(center_pos.shape[0]):
        inter_lo = torch.maximum(lo[i + 1 :], lo[i])
        inter_hi = torch.minimum(hi[i + 1 :], hi[i])
        wh = (inter_hi - inter_lo).clamp_min(0.0)
        total = total + (wh[:, 0] * wh[:, 1]).sum()
    return total


def boundary_violation_area(
    center_pos: torch.Tensor,
    size: torch.Tensor,
    canvas_origin: torch.Tensor,
    canvas_size: torch.Tensor,
) -> torch.Tensor:
    box_lo = center_pos - size / 2.0
    box_hi = center_pos + size / 2.0
    canvas_lo = canvas_origin
    canvas_hi = canvas_origin + canvas_size
    inside_lo = torch.maximum(box_lo, canvas_lo.view(1, 2))
    inside_hi = torch.minimum(box_hi, canvas_hi.view(1, 2))
    inside_wh = (inside_hi - inside_lo).clamp_min(0.0)
    box_area = (size[:, 0] * size[:, 1]).sum()
    inside_area = (inside_wh[:, 0] * inside_wh[:, 1]).sum()
    return (box_area - inside_area).clamp_min(0.0)

