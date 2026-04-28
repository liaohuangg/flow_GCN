from __future__ import annotations

import torch

from new_fm.metrics.hpwl import graph_hpwl


def test_hpwl_uses_unique_undirected_edges() -> None:
    pos = torch.tensor([[0.0, 0.0], [3.0, 4.0], [5.0, 4.0]])
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]])
    assert torch.isclose(graph_hpwl(pos, edge_index), torch.tensor(9.0))

