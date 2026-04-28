from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from new_fm.data.schema import LayoutBatch
from new_fm.models.backbones.message_passing import ResidualGraphEncoder, SimpleGraphEncoder
from new_fm.models.time_embedding import TimeMLP


class FlowMatchingModel(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        time_dim: int = 64,
        message_steps: int = 2,
        backbone: str = "simple",
        dropout: float = 0.0,
        prior: str = "gaussian",
        use_global_feat: bool = False,
    ) -> None:
        super().__init__()
        self.prior = prior
        self.use_global_feat = use_global_feat
        global_dim = 4 if use_global_feat else 0
        self.time_encoder = TimeMLP(time_dim=time_dim, hidden_dim=hidden_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim + 2 + hidden_dim + global_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if backbone == "residual":
            self.graph_encoder = ResidualGraphEncoder(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                message_steps=message_steps,
                dropout=dropout,
            )
        elif backbone == "simple":
            self.graph_encoder = SimpleGraphEncoder(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                message_steps=message_steps,
            )
        else:
            raise ValueError(f"unknown backbone: {backbone}")
        self.flow_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, batch: LayoutBatch, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.view(1).expand(len(batch.sample_id))
        if t.numel() == len(batch.sample_id):
            t_node = t[batch.batch]
        elif t.numel() == x_t.shape[0]:
            t_node = t
        else:
            raise ValueError("t must be scalar, per-graph, or per-node")
        t_emb = self.time_encoder(t_node)
        parts = [batch.node_feat, x_t, t_emb]
        if self.use_global_feat:
            parts.append(_global_features(batch))
        node_in = torch.cat(parts, dim=1)
        h = self.node_encoder(node_in)
        h = self.graph_encoder(h, batch.edge_index, batch.edge_attr)
        flow = self.flow_head(h)
        return flow * batch.movable_mask.float().view(-1, 1)

    def loss(self, batch: LayoutBatch) -> torch.Tensor:
        batch_size = len(batch.sample_id)
        t_graph = torch.rand(batch_size, device=batch.target_pos.device)
        t_node = t_graph[batch.batch].view(-1, 1)
        x_data = batch.target_pos
        x_prior = self.sample_prior(batch)
        x_t = (1.0 - t_node) * x_prior + t_node * x_data
        target = (x_data - x_prior) * batch.movable_mask.float().view(-1, 1)
        pred = self.forward(batch, x_t, t_graph)
        mask = batch.movable_mask.float().view(-1, 1)
        denom = mask.sum().clamp_min(1.0) * x_data.shape[1]
        return F.mse_loss(pred * mask, target, reduction="sum") / denom

    @torch.no_grad()
    def sample(
        self,
        batch: LayoutBatch,
        steps: int = 50,
        x_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.sample_prior(batch) if x_init is None else x_init.clone()
        fixed = ~batch.movable_mask
        x[fixed] = batch.target_pos[fixed]
        dt = 1.0 / float(steps)
        for i in range(steps):
            t = torch.full((len(batch.sample_id),), i / float(steps), device=x.device)
            flow = self.forward(batch, x, t)
            x = x + dt * flow
            x[fixed] = batch.target_pos[fixed]
        return x.clamp(0.0, 1.0)

    def sample_prior(self, batch: LayoutBatch) -> torch.Tensor:
        if self.prior == "gaussian":
            return torch.randn_like(batch.target_pos)
        if self.prior == "uniform":
            return torch.rand_like(batch.target_pos)
        raise ValueError(f"unknown prior: {self.prior}")


def _global_features(batch: LayoutBatch) -> torch.Tensor:
    num_graphs = len(batch.sample_id)
    node_count = torch.zeros((num_graphs, 1), device=batch.target_pos.device)
    node_count.index_add_(
        0,
        batch.batch,
        torch.ones((batch.batch.numel(), 1), device=batch.target_pos.device),
    )
    area = (batch.size[:, 0] * batch.size[:, 1]).view(-1, 1)
    total_area = torch.zeros((num_graphs, 1), device=batch.target_pos.device)
    total_area.index_add_(0, batch.batch, area)
    canvas_ratio = batch.canvas_size[:, :1] / batch.canvas_size[:, 1:2].clamp_min(1e-6)
    graph_feat = torch.cat(
        [
            torch.log1p(node_count) / 5.0,
            total_area,
            canvas_ratio,
            1.0 / canvas_ratio.clamp_min(1e-6),
        ],
        dim=1,
    )
    return graph_feat[batch.batch]
