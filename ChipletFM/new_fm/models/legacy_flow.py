from __future__ import annotations

import torch
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import Data

from new_fm.data.schema import LayoutBatch
from new_fm.legacy import legalization
from new_fm.legacy.models import FlowMatchingModel as OldFlowMatchingModel


class LegacyFlowMatchingWrapper(nn.Module):
    """Adapter around the legacy diffusion FlowMatchingModel/backbones."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        backbone: str,
        backbone_params: dict,
        t_encoding_type: str = "sinusoid",
        t_encoding_dim: int = 32,
        max_diffusion_steps: int = 200,
        fm_t_min: float = 1e-4,
        mask_key: str | None = "is_ports",
        use_mask_as_input: bool = True,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        params = OmegaConf.create(backbone_params)
        self.old_model = OldFlowMatchingModel(
            backbone=backbone,
            backbone_params=params,
            input_shape=(None, 2),
            t_encoding_type=t_encoding_type,
            t_encoding_dim=t_encoding_dim,
            max_diffusion_steps=max_diffusion_steps,
            fm_t_min=fm_t_min,
            mask_key=mask_key,
            use_mask_as_input=use_mask_as_input,
            device=device,
            **kwargs,
        )

    def forward(self, batch: LayoutBatch, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i in range(len(batch.sample_id)):
            sl = _node_slice(batch, i)
            cond = _batch_item_to_old_cond(batch, i)
            x_old = _to_old_pos(x_t[sl]).unsqueeze(0)
            if t.numel() == len(batch.sample_id):
                t_i = t[i].view(1)
            elif t.numel() == x_t.shape[0]:
                t_i = t[sl][:1].view(1)
            else:
                t_i = t.view(1)
            out_old = self.old_model(x_old, cond, t_i)
            outputs.append(_from_old_velocity(out_old.squeeze(0)))
        return torch.cat(outputs, dim=0) * batch.movable_mask.float().view(-1, 1)

    def loss(self, batch: LayoutBatch) -> torch.Tensor:
        if _is_repeated_single_graph(batch):
            cond = _batch_item_to_old_cond(batch, 0)
            n = int(batch.ptr[1] - batch.ptr[0])
            b = len(batch.sample_id)
            x_old = _to_old_pos(batch.target_pos).view(b, n, 2)
            loss, _ = self.old_model.loss(x_old, cond, None)
            return loss
        losses = []
        for i in range(len(batch.sample_id)):
            sl = _node_slice(batch, i)
            cond = _batch_item_to_old_cond(batch, i)
            x_old = _to_old_pos(batch.target_pos[sl]).unsqueeze(0)
            loss, _ = self.old_model.loss(x_old, cond, None)
            losses.append(loss)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def sample(
        self,
        batch: LayoutBatch,
        steps: int = 50,
        x_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = []
        for i in range(len(batch.sample_id)):
            sl = _node_slice(batch, i)
            cond = _batch_item_to_old_cond(batch, i)
            x_in = _to_old_pos(batch.target_pos[sl]).unsqueeze(0)
            mask_override = None
            if x_init is not None:
                x_init_old = _to_old_pos(x_init[sl]).unsqueeze(0)
                x_in = x_init_old
            sample_old, _ = self.old_model.reverse_samples(
                B=1,
                x_in=x_in,
                cond=cond,
                num_timesteps=steps,
                mask_override=mask_override,
            )
            outputs.append(_from_old_pos(sample_old.squeeze(0)).clamp(0.0, 1.0))
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def legalize(
        self,
        batch: LayoutBatch,
        pos: torch.Tensor,
        mode: str = "scheduled",
        **kwargs,
    ) -> torch.Tensor:
        outputs = []
        for i in range(len(batch.sample_id)):
            sl = _node_slice(batch, i)
            cond = _batch_item_to_old_cond(batch, i)
            x_old = _to_old_pos(pos[sl]).unsqueeze(0)
            if mode == "scheduled":
                out_old, _, _ = legalization.legalize(x_old, cond, **kwargs)
            elif mode == "opt":
                out_old, _, _ = legalization.legalize_opt(x_old, cond, **kwargs)
            else:
                raise ValueError(f"unknown legalizer mode: {mode}")
            outputs.append(_from_old_pos(out_old.squeeze(0)).clamp(0.0, 1.0))
        return torch.cat(outputs, dim=0)


def _batch_item_to_old_cond(batch: LayoutBatch, sample_idx: int) -> Data:
    sl = _node_slice(batch, sample_idx)
    start = int(batch.ptr[sample_idx])
    end = int(batch.ptr[sample_idx + 1])
    edge_mask = (batch.edge_index[0] >= start) & (batch.edge_index[0] < end)
    edge_index = batch.edge_index[:, edge_mask] - start
    edge_attr = batch.edge_attr[edge_mask]
    node_feat = batch.node_feat[sl]
    is_ports = node_feat[:, 3] > 0.5 if node_feat.shape[1] >= 4 else ~batch.movable_mask[sl]
    is_macros = node_feat[:, 4] > 0.5 if node_feat.shape[1] >= 5 else batch.movable_mask[sl]
    return Data(
        x=(batch.size[sl] * 2.0).float(),
        edge_index=edge_index.long(),
        edge_attr=edge_attr.float(),
        is_ports=is_ports.bool(),
        is_macros=is_macros.bool(),
    ).to(batch.target_pos.device)


def _node_slice(batch: LayoutBatch, sample_idx: int) -> slice:
    return slice(int(batch.ptr[sample_idx]), int(batch.ptr[sample_idx + 1]))


def _to_old_pos(pos: torch.Tensor) -> torch.Tensor:
    return pos * 2.0 - 1.0


def _from_old_pos(pos: torch.Tensor) -> torch.Tensor:
    return (pos + 1.0) / 2.0


def _from_old_velocity(velocity: torch.Tensor) -> torch.Tensor:
    return velocity / 2.0


def _is_repeated_single_graph(batch: LayoutBatch) -> bool:
    if len(batch.sample_id) <= 1:
        return False
    n = int(batch.ptr[1] - batch.ptr[0])
    if not torch.all((batch.ptr[1:] - batch.ptr[:-1]) == n):
        return False
    first_id = batch.sample_id[0]
    if any(sample_id != first_id for sample_id in batch.sample_id):
        return False
    return True
