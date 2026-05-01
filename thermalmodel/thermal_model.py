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
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None, d: int = 1):
        super().__init__()
        if p is None:
            p = (k // 2) * d
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.gn = _group_norm(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class LargeKernelBlock(nn.Module):
    """Large-kernel residual block to better capture global heat diffusion."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.norm = _group_norm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return res + x


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale/global context."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvGNAct(in_ch, out_ch, k=1, p=0)
        self.conv2 = ConvGNAct(in_ch, out_ch, k=3, d=2)
        self.conv3 = ConvGNAct(in_ch, out_ch, k=3, d=4)
        self.conv4 = ConvGNAct(in_ch, out_ch, k=3, d=6)
        self.out_conv = ConvGNAct(out_ch * 4, out_ch, k=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return self.out_conv(out)



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

    def __init__(self, base: int = 64):
        super().__init__()

        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

        # Default QAT config (actual QAT is enabled in train via FX prepare_qat_fx).
        self.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")

        in_ch = 4

        # Stem: 128x128 -> 64x64 (align to target temperature grid)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=4, stride=2, padding=1, bias=False),
            _group_norm(base),
            nn.GELU(),
        )

        # Encoder (64 -> 32 -> 16 -> 8)
        self.e1 = nn.Sequential(LargeKernelBlock(base), LargeKernelBlock(base))
        self.d1 = ConvGNAct(base, base * 2, k=3, s=2)  # 64->32

        self.e2 = nn.Sequential(LargeKernelBlock(base * 2), LargeKernelBlock(base * 2))
        self.d2 = ConvGNAct(base * 2, base * 4, k=3, s=2)  # 32->16

        self.e3 = nn.Sequential(LargeKernelBlock(base * 4), LargeKernelBlock(base * 4))
        self.d3 = ConvGNAct(base * 4, base * 8, k=3, s=2)  # 16->8

        # Bottleneck with ASPP
        self.bottleneck = nn.Sequential(
            LargeKernelBlock(base * 8),
            ASPP(base * 8, base * 8),
            LargeKernelBlock(base * 8),
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.fuse3 = nn.Sequential(ConvGNAct(base * 8, base * 4, k=3, s=1), LargeKernelBlock(base * 4))

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.fuse2 = nn.Sequential(ConvGNAct(base * 4, base * 2, k=3, s=1), LargeKernelBlock(base * 2))

        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.fuse1 = nn.Sequential(ConvGNAct(base * 2, base, k=3, s=1), LargeKernelBlock(base))

        # Head at 64x64
        self.head = nn.Conv2d(base, 1, kernel_size=3, padding=1)

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

        # 128 -> 64
        x0 = self.stem(x)

        s1 = self.e1(x0)  # 64
        x1 = self.d1(s1)  # 32

        s2 = self.e2(x1)  # 32
        x2 = self.d2(s2)  # 16

        s3 = self.e3(x2)  # 16
        x3 = self.d3(s3)  # 8

        z = self.bottleneck(x3)  # 8

        u3 = self.up3(z)  # 16
        u3 = torch.cat([u3, s3], dim=1)
        u3 = self.fuse3(u3)

        u2 = self.up2(u3)  # 32
        u2 = torch.cat([u2, s2], dim=1)
        u2 = self.fuse2(u2)

        u1 = self.up1(u2)  # 64
        u1 = torch.cat([u1, s1], dim=1)
        u1 = self.fuse1(u1)

        out = self.head(u1)
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
    return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


def laplacian_filter(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    return kernel


def physics_informed_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    lap_kernel = laplacian_filter(pred.device, pred.dtype)
    lap_p = F.conv2d(pred, lap_kernel, padding=1)
    lap_t = F.conv2d(target, lap_kernel, padding=1)
    return F.l1_loss(lap_p, lap_t)


def guidance_loss(pred: torch.Tensor, target: torch.Tensor, grad_w: float = 0.1) -> Tuple[torch.Tensor, Dict[str, float]]:
    # 1) base per-pixel L1
    base_l1 = F.l1_loss(pred, target, reduction="none")

    # 2) hotspot-aware weighting
    tmin = target.amin(dim=(2, 3), keepdim=True)
    tmax = target.amax(dim=(2, 3), keepdim=True)
    weight = 1.0 + 5.0 * (target - tmin) / (tmax - tmin + 1e-8)
    weighted_l1 = (base_l1 * weight).mean()

    # 3) gradient loss (1st order)
    grad = spatial_gradient_loss(pred, target)

    # 4) Laplacian loss (2nd order)
    lap = physics_informed_loss(pred, target)

    loss = weighted_l1 + grad_w * grad + 0.05 * lap

    with torch.no_grad():
        mse_metric = F.mse_loss(pred, target).item()

    return loss, {
        "l1": float(weighted_l1.detach().cpu()),
        "mse": float(mse_metric),
        "grad": float(grad.detach().cpu()),
        "lap": float(lap.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }


@dataclass
class _Stats:
    temp_min: float
    temp_max: float


def _maybe_temp_stats(ckpt: dict) -> Optional[_Stats]:
    return None


def _denorm_temp_k(x01: torch.Tensor, st: _Stats) -> torch.Tensor:
    # x01 is min-max normalized to [0,1] using dataset stats.
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

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    start_epoch = 1
    ckpt = None
    if getattr(args, "resume_ckpt", ""):
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[resume] ckpt={args.resume_ckpt} | start_epoch={start_epoch} | target_epochs={args.epochs}")

    # Scheduler uses total epochs; when resuming we restore scheduler state so LR continues smoothly.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    if getattr(args, "resume_ckpt", ""):
        if ckpt is None or "scheduler" not in ckpt:
            raise ValueError("Resume checkpoint is missing 'scheduler' state; please resume from a newer checkpoint.")
        scheduler.load_state_dict(ckpt["scheduler"])
        print(f"[resume] scheduler restored (last_epoch={scheduler.last_epoch})")

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

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        for it, batch in enumerate(train_loader):
            totalp = batch.get("total_power")

            power = batch["power"].to(device)
            layout = batch["layout"].to(device)
            temp = batch["temp"].to(device)
            totalp = totalp.to(device) if totalp is not None else None

            pred, _pred_avg = model(power, layout, totalp)
            loss, m = guidance_loss(pred, temp, grad_w=args.grad_w)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # Print the first-iteration (one batch) output once for quick sanity-check.
            if epoch == start_epoch and it == 0:
                with torch.no_grad():
                    b0_pred = pred[0].detach().cpu()
                    b0_gt = temp[0].detach().cpu()
                    print(
                        f"[one-batch] ep {epoch:04d} it {it:04d} "
                        f"loss {m['loss']:.6f} mse {m['mse']:.6f} l1 {m['l1']:.6f} grad {m['grad']:.6f} lap {m['lap']:.6f} "
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
                    f"[train] ep {epoch:04d} it {it:04d} loss {m['loss']:.6f} mse {m['mse']:.6f} l1 {m['l1']:.6f} "
                    f"grad {m['grad']:.6f} lap {m['lap']:.6f} "
                    f"pred(max/avg) {pmax:.4f}/{pavg:.4f} gt(max/avg) {tmax:.4f}/{tavg:.4f}"
                )

        model.eval()
        vm = {"loss": 0.0, "mse": 0.0, "l1": 0.0, "grad": 0.0, "lap": 0.0}
        steps = 0
        with torch.no_grad():
            for batch in val_loader:
                totalp = batch.get("total_power")

                power = batch["power"].to(device)
                layout = batch["layout"].to(device)
                temp = batch["temp"].to(device)
                totalp = totalp.to(device) if totalp is not None else None

                pred, _pred_avg = model(power, layout, totalp)
                _, m = guidance_loss(pred, temp, grad_w=args.grad_w)
                for k in vm:
                    vm[k] += m[k]
                steps += 1
        for k in vm:
            vm[k] /= max(steps, 1)
        print(
            f"Epoch {epoch:04d} | val loss {vm['loss']:.6f} mse {vm['mse']:.6f} "
            f"l1 {vm['l1']:.6f} grad {vm['grad']:.6f} lap {vm['lap']:.6f}"
        )

        scheduler.step()

        if epoch % args.ckpt_every == 0:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "stats": dataset.stats.to_dict(),
                "grid_size": 128,
                "base": args.base,
                "train_args": {
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "base": int(args.base),
                    "grad_w": float(args.grad_w),
                    "resume_ckpt": str(getattr(args, "resume_ckpt", "")),
                },
            }
            tag = f"bs{int(args.batch_size)}_lr{float(args.lr):.0e}_base{int(args.base)}_gradw{float(args.grad_w):g}"
            path = os.path.join(out_dir, f"guidance_net_epoch{epoch}_{tag}.pth")
            torch.save(ckpt, path)
            print(f"[ckpt] saved: {path}")

            if args.qat:
                # Export an int8-converted model checkpoint by default when --qat is enabled.
                try:
                    from torch.ao.quantization.quantize_fx import convert_fx

                    model.eval()
                    int8_model = convert_fx(model)
                    tag = f"bs{int(args.batch_size)}_lr{float(args.lr):.0e}_base{int(args.base)}_gradw{float(args.grad_w):g}"
                    int8_path = os.path.join(out_dir, f"guidance_net_epoch{epoch}_{tag}_int8.pth")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model": int8_model.state_dict(),
                            "stats": dataset.stats.to_dict(),
                            "grid_size": 128,
                            "base": args.base,
                            "int8": True,
                            "train_args": {
                                "batch_size": int(args.batch_size),
                                "lr": float(args.lr),
                                "base": int(args.base),
                                "grad_w": float(args.grad_w),
                                "resume_ckpt": str(getattr(args, "resume_ckpt", "")),
                            },
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

    # Use dataset stats for denormalization (matches ThermalDataset min-max scaling)
    st = _Stats(temp_min=float(dataset.stats.temp_min), temp_max=float(dataset.stats.temp_max))

    # Sanity-check: dataset temp should already be normalized [0,1]
    ds0 = dataset[0]["temp"]
    ds0_min = float(ds0.min().item())
    ds0_max = float(ds0.max().item())
    if ds0_min < -0.05 or ds0_max > 1.05:
        print(f"[warn] dataset temp seems not normalized: min={ds0_min:.4f} max={ds0_max:.4f}")

    print(
        f"[test] denorm stats: temp_min={st.temp_min:.6f} temp_max={st.temp_max:.6f} "
        f"(dataset.stats; range={st.temp_max - st.temp_min:.6f})"
    )

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

    rmse_min = None
    rmse_max = None

    max_ae = 0.0

    rmspe_sum = 0.0
    rmspe_n = 0

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

        units = "K"

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
            totalp = batch.get("total_power")

            power = batch["power"].to(device)
            layout = batch["layout"].to(device)
            temp = batch["temp"].to(device)
            totalp = totalp.to(device) if totalp is not None else None

            pred, _pred_avg = model(power, layout, totalp)

            pred_eval_norm = pred.detach().cpu()
            temp_eval_norm = temp.detach().cpu()

            # debug: print a few values in both norm and K scales
            if n == 0:
                with torch.no_grad():
                    pn_min = float(pred_eval_norm.min().item())
                    pn_max = float(pred_eval_norm.max().item())
                    tn_min = float(temp_eval_norm.min().item())
                    tn_max = float(temp_eval_norm.max().item())
                    print(f"[debug] pred_norm(min/max) {pn_min:.6f}/{pn_max:.6f} | gt_norm(min/max) {tn_min:.6f}/{tn_max:.6f}")

            pred_eval = _denorm_temp_k(pred_eval_norm, st)
            temp_eval = _denorm_temp_k(temp_eval_norm, st)

            if n == 0:
                with torch.no_grad():
                    pk_min = float(pred_eval.min().item())
                    pk_max = float(pred_eval.max().item())
                    tk_min = float(temp_eval.min().item())
                    tk_max = float(temp_eval.max().item())
                    # print a 2x3 patch from the first sample
                    p_patch = pred_eval[0, 0, :2, :3]
                    t_patch = temp_eval[0, 0, :2, :3]
                    print(f"[debug] pred_K(min/max) {pk_min:.3f}/{pk_max:.3f} | gt_K(min/max) {tk_min:.3f}/{tk_max:.3f}")
                    print(f"[debug] pred_K patch\n{p_patch}")
                    print(f"[debug] gt_K patch\n{t_patch}")

            # per-sample rmse
            mse = F.mse_loss(pred_eval, temp_eval, reduction="none").mean(dim=(1, 2, 3))
            rmse = torch.sqrt(mse)
            grad = spatial_gradient_loss(pred_eval, temp_eval)

            # absolute error metrics
            abs_err = (pred_eval - temp_eval).abs()
            batch_max_ae = float(abs_err.max().item())
            if batch_max_ae > max_ae:
                max_ae = batch_max_ae

            # percentage error (use K-scale denominator)
            denom = temp_eval.abs().clamp_min(1e-8)
            pe2 = ((pred_eval - temp_eval) / denom) ** 2
            rmspe_sum += float(pe2.mean().sqrt().item())
            rmspe_n += 1

            for b in range(mse.shape[0]):
                r = float(rmse[b].item())

                mse_sum += float(mse[b].item())
                rmse_sum += r
                grad_sum += float(grad.item())
                n += 1

                rmse_min = r if rmse_min is None else min(rmse_min, r)
                rmse_max = r if rmse_max is None else max(rmse_max, r)

                if worst is None or r > float(worst["rmse"]):
                    worst = {
                        "rmse": r,
                        "i": int(batch["i"][b]),
                        "j": int(batch["j"][b]),
                        "pred": pred_eval[b],
                        "gt": temp_eval[b],
                    }

                in_block += 1
                if in_block >= block_size:
                    flush()
                    block_idx += 1
                    in_block = 0

    if worst is not None:
        flush()

    print("==== Guidance Test Metrics (K) ====")
    print(f"max_rmse:  {float(rmse_max) if rmse_max is not None else float('nan'):.8f}")
    print(f"min_rmse:  {float(rmse_min) if rmse_min is not None else float('nan'):.8f}")
    print(f"mean_rmse: {rmse_sum / max(n, 1):.8f}")
    print(f"max_ae:    {max_ae:.8f}")
    print(f"mean_rmspe:{(rmspe_sum / max(rmspe_n, 1)) * 100.0:.6f}%")
    print(f"mse:       {mse_sum / max(n, 1):.8f}")
    print(f"grad:      {grad_sum / max(n, 1):.8f}")
    print(f"fig_dir:   {fig_dir}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--resume_ckpt", type=str, default="", help="resume from checkpoint (.pth)")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch_size", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--base", type=int, default=64)
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
python /root/placement/flow_GCN/thermalmodel/guidance_model.py train --epochs 200 --batch_size 16 --lr 5e-4 --base 32 --grad_w 0.01 --ckpt_every 5 --print_every 10

test
python /root/placement/flow_GCN/thermalmodel/guidance_model.py test --ckpt /root/placement/flow_GCN/thermalmodel/checkpoints/guidance_net_epoch20.pth  --batch_size 8 --out_fig_dir /root/placement/flow_GCN/thermalmodel/test_result/test_guidance_fig
'''