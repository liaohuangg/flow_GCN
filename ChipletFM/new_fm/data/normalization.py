from __future__ import annotations

import torch


def normalize_pos(pos: torch.Tensor, origin: torch.Tensor, canvas_size: torch.Tensor) -> torch.Tensor:
    return (pos - origin.view(1, 2)) / canvas_size.clamp_min(1e-6).view(1, 2)


def denormalize_pos(pos: torch.Tensor, origin: torch.Tensor, canvas_size: torch.Tensor) -> torch.Tensor:
    return pos * canvas_size.clamp_min(1e-6).view(1, 2) + origin.view(1, 2)


def normalize_size(size: torch.Tensor, canvas_size: torch.Tensor) -> torch.Tensor:
    return size / canvas_size.clamp_min(1e-6).view(1, 2)

