import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow running `python guidance_model.py ...` from within thermalmodel/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from thermalmodel.dataLoader import ThermalDataset
from thermalmodel.draw_thermal_fig import plot_thermal_grid_overlay


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_coord_maps(h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return x/y coordinate maps in [-1,1], shape (1,1,H,W)."""
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(1, 1, h, w)
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(1, 1, h, w)
    return xs, ys


def _group_norm(ch: int) -> nn.GroupNorm:
    # Choose a divisor of ch for stable GroupNorm
    for g in (16, 8, 4, 2, 1):
        if ch % g == 0:
            return nn.GroupNorm(num_groups=g, num_channels=ch)
    return nn.GroupNorm(num_groups=1, num_channels=ch)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.gn = _group_norm(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class LiteInvertedResidual(nn.Module):
    """Lite inverted residual block (MobileNetV3-style) using SiLU + GroupNorm.

    QAT friendliness:
      - no BatchNorm
      - depthwise + pointwise conv
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int = 2):
        super().__init__()
        assert stride in (1, 2)
        mid = int(in_ch * expand_ratio)

        self.use_res = (stride == 1 and in_ch == out_ch)

        layers = []
        if mid != in_ch:
            layers.append(ConvGNAct(in_ch, mid, k=1, s=1, p=0))

        layers.append(nn.Conv2d(mid, mid, kernel_size=3, stride=stride, padding=1, groups=mid, bias=False))
        layers.append(_group_norm(mid))
        layers.append(nn.SiLU(inplace=True))

        layers.append(nn.Conv2d(mid, out_ch, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(_group_norm(out_ch))

        self.net = nn.Sequential(*layers)
        self.out_act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.use_res:
            y = y + x
        return self.out_act(y)


class FiLM(nn.Module):
    """FiLM modulation for bottleneck features."""

    def __init__(self, cond_dim: int, feat_ch: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, feat_ch * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.mlp(cond)  # (B,2C)
        gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class ThermalGuidanceNet(nn.Module):
    """ThermalGuidanceNet: 64x64 grid predictor for diffusion guidance.

    Required input channels (CoordConv): [power_grid, layout_mask, x_coord_map, y_coord_map].

    Note:
      The current dataset loader derives a hard 0/1 mask from .flp (not differentiable). For diffusion,
      keep the model differentiable and supply a differentiable layout representation at inference time.
    """

    def __init__(self, base: int = 32):
        super().__init__()

        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

        # Default QAT config (actual QAT is enabled in train via FX prepare_qat_fx).
        self.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")

        in_ch = 4

        self.stem = ConvGNAct(in_ch, base, k=3, s=1)

        # encoder
        self.e1 = nn.Sequential(
            LiteInvertedResidual(base, base, stride=1, expand_ratio=2),
            LiteInvertedResidual(base, base, stride=1, expand_ratio=2),
        )
        self.d1 = LiteInvertedResidual(base, base * 2, stride=2, expand_ratio=2)  # 128->64

        self.e2 = nn.Sequential(
            LiteInvertedResidual(base * 2, base * 2, stride=1, expand_ratio=2),
            LiteInvertedResidual(base * 2, base * 2, stride=1, expand_ratio=2),
        )
        self.d2 = LiteInvertedResidual(base * 2, base * 4, stride=2, expand_ratio=2)  # 64->32

        self.e3 = nn.Sequential(
            LiteInvertedResidual(base * 4, base * 4, stride=1, expand_ratio=2),
            LiteInvertedResidual(base * 4, base * 4, stride=1, expand_ratio=2),
        )
        self.d3 = LiteInvertedResidual(base * 4, base * 8, stride=2, expand_ratio=2)  # 32->16

        # bottleneck
        self.bottleneck = nn.Sequential(
            LiteInvertedResidual(base * 8, base * 8, stride=1, expand_ratio=2),
            LiteInvertedResidual(base * 8, base * 8, stride=1, expand_ratio=2),
        )

        # decoder: upsample + concat skip + conv
        self.up3 = ConvGNAct(base * 8, base * 4, k=1, s=1, p=0)
        self.fuse3 = ConvGNAct(base * 8, base * 4, k=3, s=1)

        self.up2 = ConvGNAct(base * 4, base * 2, k=1, s=1, p=0)
        self.fuse2 = ConvGNAct(base * 4, base * 2, k=3, s=1)

        # Head directly at 64x64 (no 128->64 avg_pool downsample)
        self.head = nn.Conv2d(base * 2, 1, kernel_size=1)

        # cache coord maps
        self._coord_hw: Optional[Tuple[int, int]] = None
        self._coord_xy: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _coords(self, b: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self._coord_hw != (h, w)
            or self._coord_xy is None
            or self._coord_xy[0].device != device
            or self._coord_xy[0].dtype != dtype
        ):
            self._coord_hw = (h, w)
            self._coord_xy = _make_coord_maps(h, w, device=device, dtype=dtype)
        x, y = self._coord_xy
        return x.expand(b, -1, -1, -1), y.expand(b, -1, -1, -1)

    def forward(self, power_grid: torch.Tensor, layout_mask: torch.Tensor) -> torch.Tensor:
        b, _, h, w = power_grid.shape
        xmap, ymap = self._coords(b, h, w, device=power_grid.device, dtype=power_grid.dtype)

        x = torch.cat([power_grid, layout_mask, xmap, ymap], dim=1)
        x = self.quant(x)

        x0 = self.stem(x)
        s1 = self.e1(x0)
        x1 = self.d1(s1)
        s2 = self.e2(x1)
        x2 = self.d2(s2)
        s3 = self.e3(x2)
        x3 = self.d3(s3)

        z = self.bottleneck(x3)

        u3 = F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False)
        u3 = self.up3(u3)
        u3 = torch.cat([u3, s3], dim=1)
        u3 = self.fuse3(u3)

        u2 = F.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = self.up2(u2)
        u2 = torch.cat([u2, s2], dim=1)
        u2 = self.fuse2(u2)

        out = self.head(u2)
        out = self.dequant(out)
        return out


def _sobel_filters(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    return kx, ky


def spatial_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kx, ky = _sobel_filters(pred.device, pred.dtype)
    gx_p = F.conv2d(pred, kx, padding=1)
    gy_p = F.conv2d(pred, ky, padding=1)
    gx_t = F.conv2d(target, kx, padding=1)
    gy_t = F.conv2d(target, ky, padding=1)
    return F.mse_loss(gx_p, gx_t) + F.mse_loss(gy_p, gy_t)


def guidance_loss(pred: torch.Tensor, target: torch.Tensor, grad_w: float = 0.01) -> Tuple[torch.Tensor, Dict[str, float]]:
    # 1) base per-pixel MSE
    base_mse = F.mse_loss(pred, target, reduction="none")

    # 2) hotspot-aware weighting (target is normalized to [0,1])
    tmin = target.amin(dim=(2, 3), keepdim=True)
    tmax = target.amax(dim=(2, 3), keepdim=True)
    weight = 1.0 + 3.0 * (target - tmin) / (tmax - tmin + 1e-8)

    weighted_mse = (base_mse * weight).mean()

    # 3) gradient loss
    grad = spatial_gradient_loss(pred, target)

    loss = weighted_mse + grad_w * grad
    return loss, {
        "mse": float(weighted_mse.detach().cpu()),
        "grad": float(grad.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }


@dataclass
class _Stats:
    temp_min: float
    temp_max: float


def _maybe_temp_stats(ckpt: dict) -> Optional[_Stats]:
    s = ckpt.get("stats")
    if not isinstance(s, dict):
        return None
    try:
        return _Stats(temp_min=float(s["temp_min"]), temp_max=float(s["temp_max"]))
    except Exception:
        return None


def _denorm_temp_k(x01: torch.Tensor, st: _Stats) -> torch.Tensor:
    return x01 * (st.temp_max - st.temp_min) + st.temp_min


def main_train(args) -> None:
    torch.manual_seed(0)
    device = _device()

    dataset = ThermalDataset(
        thermal_map_rel="Dataset/dataset/output/thermal/thermal_map",
        hotspot_cfg_rel="Dataset/dataset/output/thermal/hotspot_config",
        power_grid_size=128,
        temp_grid_size=64,
    )

    n_total = len(dataset)
    n_train = int(n_total * 0.9)
    n_val = n_total - n_train
    gen = torch.Generator().manual_seed(0)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=gen)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ThermalGuidanceNet(base=args.base).to(device)
    if args.qat:
        # NOTE: FX graph-mode QAT APIs have been evolving; if you see deprecation warnings from
        # your local IDE/linter, the runtime torch version is the source of truth.
        from torch.ao.quantization import get_default_qat_qconfig_mapping
        from torch.ao.quantization.quantize_fx import prepare_qat_fx

        model.train()
        # Per-tensor weight observers to keep int8 conversion on CPU backend simple.
        # (Avoid per_channel_affine which can be unsupported by some convert paths.)
        from torch.ao.quantization.qconfig import QConfig
        from torch.ao.quantization.observer import MovingAverageMinMaxObserver

        act = MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
        wt = MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        qconfig_mapping = get_default_qat_qconfig_mapping("fbgemm").set_global(QConfig(activation=act, weight=wt))
        example_inputs = (
            torch.zeros(args.batch_size, 1, 128, 128, device=device),
            torch.zeros(args.batch_size, 1, 128, 128, device=device),
        )
        model = prepare_qat_fx(model, qconfig_mapping, example_inputs)
        print("[QAT] enabled: prepare_qat_fx(model, qconfig_mapping, example_inputs)")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    print("==== Thermal Guidance Training Config ====")
    print(f"device: {device} (cuda_available={torch.cuda.is_available()})")
    if torch.cuda.is_available():
        try:
            print(f"cuda_device: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    print("power_grid: 128 -> temp_grid: 64")
    print(f"dataset_total: {n_total} | train: {n_train} | val: {n_val}")
    print(f"batch_size: {args.batch_size} | train_batches: {len(train_loader)} | val_batches: {len(val_loader)}")
    print("========================================")

    for epoch in range(1, args.epochs + 1):
        model.train()
        for it, batch in enumerate(train_loader):
            power = batch["power"].to(device)
            layout = batch["layout"].to(device)
            temp = batch["temp"].to(device)

            pred = model(power, layout)
            loss, m = guidance_loss(pred, temp, grad_w=args.grad_w)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # Print the first-iteration (one batch) output once for quick sanity-check.
            if epoch == 1 and it == 0:
                with torch.no_grad():
                    b0_pred = pred[0].detach().cpu()
                    b0_gt = temp[0].detach().cpu()
                    b0_totalp = None
                    print(
                        f"[one-batch] ep {epoch:04d} it {it:04d} "
                        f"loss {m['loss']:.6f} mse {m['mse']:.6f} grad {m['grad']:.6f} "
                        f"pred(min/max/avg) {float(b0_pred.min()):.4f}/{float(b0_pred.max()):.4f}/{float(b0_pred.mean()):.4f} "
                        f"gt(min/max/avg) {float(b0_gt.min()):.4f}/{float(b0_gt.max()):.4f}/{float(b0_gt.mean()):.4f}"
                    )

            if it % args.print_every == 0:
                with torch.no_grad():
                    pmax = float(pred.max().item())
                    pavg = float(pred.mean().item())
                    tmax = float(temp.max().item())
                    tavg = float(temp.mean().item())
                print(
                    f"[train] ep {epoch:04d} it {it:04d} loss {m['loss']:.6f} mse {m['mse']:.6f} grad {m['grad']:.6f} "
                    f"pred(max/avg) {pmax:.4f}/{pavg:.4f} gt(max/avg) {tmax:.4f}/{tavg:.4f}"
                )

        model.eval()
        vm = {"loss": 0.0, "mse": 0.0, "grad": 0.0}
        steps = 0
        with torch.no_grad():
            for batch in val_loader:
                power = batch["power"].to(device)
                layout = batch["layout"].to(device)
                temp = batch["temp"].to(device)

                pred = model(power, layout)
                _, m = guidance_loss(pred, temp, grad_w=args.grad_w)
                for k in vm:
                    vm[k] += m[k]
                steps += 1
        for k in vm:
            vm[k] /= max(steps, 1)
        print(f"Epoch {epoch:04d} | val loss {vm['loss']:.6f} mse {vm['mse']:.6f} grad {vm['grad']:.6f}")

        if epoch % args.ckpt_every == 0:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "stats": dataset.stats.to_dict(),
                "grid_size": 128,
                "base": args.base,
            }
            path = os.path.join(out_dir, f"guidance_net_epoch{epoch}.pth")
            torch.save(ckpt, path)
            print(f"[ckpt] saved: {path}")

            if args.qat:
                # Export an int8-converted model checkpoint by default when --qat is enabled.
                try:
                    from torch.ao.quantization.quantize_fx import convert_fx

                    model.eval()
                    int8_model = convert_fx(model)
                    int8_path = os.path.join(out_dir, f"guidance_net_epoch{epoch}_int8.pth")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model": int8_model.state_dict(),
                            "stats": dataset.stats.to_dict(),
                            "grid_size": 128,
                            "base": args.base,
                            "int8": True,
                        },
                        int8_path,
                    )
                    print(f"[ckpt] saved: {int8_path}")
                except Exception as e:
                    print(f"[QAT] convert_fx export failed: {e}")


def main_test(args) -> None:
    device = _device()

    dataset = ThermalDataset(
        thermal_map_rel="Dataset/dataset/output/thermal/thermal_map",
        hotspot_cfg_rel="Dataset/dataset/output/thermal/hotspot_config",
        power_grid_size=128,
        temp_grid_size=64,
    )

    n_total = len(dataset)
    n_train = int(n_total * 0.9)
    n_test = n_total - n_train
    gen = torch.Generator().manual_seed(0)
    _, test_set = torch.utils.data.random_split(dataset, [n_train, n_test], generator=gen)

    loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = ThermalGuidanceNet(base=int(ckpt.get("base", 32))).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    st = _maybe_temp_stats(ckpt)

    out_dir = os.path.join(os.path.dirname(__file__), "test_result")
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "test_guidance_fig")
    os.makedirs(fig_dir, exist_ok=True)

    if hasattr(args, "out_fig_dir") and args.out_fig_dir:
        fig_dir = args.out_fig_dir
        os.makedirs(fig_dir, exist_ok=True)

    mse_sum = 0.0
    rmse_sum = 0.0
    grad_sum = 0.0
    n = 0

    block_size = 100
    in_block = 0
    block_idx = 0
    worst = None

    def flush() -> None:
        nonlocal worst, block_idx
        if worst is None:
            return
        i = int(worst["i"])
        j = int(worst["j"])
        score = float(worst["rmse"])

        flp = os.path.join(
            _PROJECT_ROOT,
            "Dataset/dataset/output/thermal/hotspot_config",
            f"system_{i}_config",
            "system.flp",
        )

        pred = worst["pred"]
        gt = worst["gt"]

        if st is not None:
            pred = _denorm_temp_k(pred, st)
            gt = _denorm_temp_k(gt, st)
            units = "K"
        else:
            units = "norm"

        vmin = float(min(pred.min().item(), gt.min().item()))
        vmax = float(max(pred.max().item(), gt.max().item()))

        plot_thermal_grid_overlay(
            flp,
            pred,
            os.path.join(fig_dir, f"guidance_model_block{block_idx:03d}_i{i}_j{j}_pred_rmse{score:.6f}.png"),
            title=f"Pred block {block_idx} i={i} j={j} RMSE={score:.6f}",
            vmin=vmin,
            vmax=vmax,
            units=units,
        )
        plot_thermal_grid_overlay(
            flp,
            gt,
            os.path.join(fig_dir, f"guidance_model_block{block_idx:03d}_i{i}_j{j}_gt.png"),
            title=f"GT block {block_idx} i={i} j={j}",
            vmin=vmin,
            vmax=vmax,
            units=units,
        )

        worst = None

    with torch.no_grad():
        for batch in loader:
            power = batch["power"].to(device)
            layout = batch["layout"].to(device)
            temp = batch["temp"].to(device)

            pred = model(power, layout)

            mse = F.mse_loss(pred, temp, reduction="none").mean(dim=(1, 2, 3)).detach().cpu()
            rmse = torch.sqrt(mse)
            grad = spatial_gradient_loss(pred, temp).detach().cpu()

            for b in range(mse.shape[0]):
                mse_sum += float(mse[b].item())
                rmse_sum += float(rmse[b].item())
                grad_sum += float(grad.item())
                n += 1

                if worst is None or float(rmse[b].item()) > float(worst["rmse"]):
                    worst = {
                        "rmse": float(rmse[b].item()),
                        "i": int(batch["i"][b]),
                        "j": int(batch["j"][b]),
                        "pred": pred[b].detach().cpu(),
                        "gt": temp[b].detach().cpu(),
                    }

                in_block += 1
                if in_block >= block_size:
                    flush()
                    block_idx += 1
                    in_block = 0

    if worst is not None:
        flush()

    print("==== Guidance Test Metrics (normalized) ====")
    print(f"mse:  {mse_sum / max(n, 1):.8f}")
    print(f"rmse: {rmse_sum / max(n, 1):.8f}")
    print(f"grad: {grad_sum / max(n, 1):.8f}")
    print(f"fig_dir: {fig_dir}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch_size", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--base", type=int, default=32)
    t.add_argument("--grad_w", type=float, default=0.1)
    t.add_argument("--ckpt_every", type=int, default=1)
    t.add_argument("--print_every", type=int, default=10)
    t.add_argument("--qat", action="store_true", help="enable quantization-aware training")

    te = sub.add_parser("test")
    te.add_argument("--ckpt", type=str, required=True)
    te.add_argument("--batch_size", type=int, default=8)
    te.add_argument("--out_fig_dir", type=str, default="")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.cmd == "train":
        main_train(args)
    elif args.cmd == "test":
        main_test(args)

# python guidance_model.py train --epochs 20 --batch_size 16 --ckpt_every 50 --print_every 10
'''
train
python /root/placement/flow_GCN/thermalmodel/guidance_model.py train --epochs 20 --batch_size 16 --lr 5e-4 --base 32 --grad_w 0.01 --ckpt_every 1 --print_every 10

test
python /root/placement/flow_GCN/thermalmodel/guidance_model.py test --ckpt /root/placement/flow_GCN/thermalmodel/checkpoints/guidance_net_epoch20.pth  --batch_size 8 --out_fig_dir /root/placement/flow_GCN/thermalmodel/test_result/test_guidance_fig
'''