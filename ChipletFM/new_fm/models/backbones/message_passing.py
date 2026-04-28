from __future__ import annotations

import torch
from torch import nn


class SimpleGraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, message_steps: int) -> None:
        super().__init__()
        self.message_steps = message_steps
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, node_state: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        h = node_state
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index
        for _ in range(self.message_steps):
            msg_in = torch.cat([h[src], h[dst], edge_attr], dim=1)
            msg = self.edge_mlp(msg_in)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, msg)
            deg = torch.zeros((h.shape[0], 1), device=h.device, dtype=h.dtype)
            deg.index_add_(0, dst, torch.ones((dst.numel(), 1), device=h.device, dtype=h.dtype))
            agg = agg / deg.clamp_min(1.0)
            h = h + self.node_mlp(torch.cat([h, agg], dim=1))
        return h


class ResidualGraphEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        message_steps: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResidualMessagePassingBlock(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dim,
                    dropout=dropout,
                )
                for _ in range(message_steps)
            ]
        )

    def forward(
        self, node_state: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        h = node_state
        for block in self.blocks:
            h = block(h, edge_index, edge_attr)
        return h


class ResidualMessagePassingBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.msg_norm = nn.LayerNorm(hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self, node_state: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return node_state
        h = self.node_norm(node_state)
        src, dst = edge_index
        msg = self.edge_mlp(torch.cat([h[src], h[dst], edge_attr], dim=1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros((h.shape[0], 1), device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones((dst.numel(), 1), device=h.device, dtype=h.dtype))
        agg = self.msg_norm(agg / deg.clamp_min(1.0))
        return node_state + self.node_mlp(torch.cat([h, agg], dim=1))
