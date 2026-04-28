from __future__ import annotations

import torch

from new_fm.metrics.legality import legality_score, overlap_area


def test_legality_perfect_simple_layout() -> None:
    pos = torch.tensor([[1.0, 1.0], [3.0, 1.0]])
    size = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    score = legality_score(pos, size, torch.tensor([0.0, 0.0]), torch.tensor([4.0, 4.0]))
    assert torch.isclose(score, torch.tensor(1.0))


def test_legality_detects_overlap() -> None:
    pos = torch.tensor([[1.0, 1.0], [1.5, 1.0]])
    size = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    assert torch.isclose(overlap_area(pos, size), torch.tensor(0.5))
    score = legality_score(pos, size, torch.tensor([0.0, 0.0]), torch.tensor([4.0, 4.0]))
    assert score < 1.0

