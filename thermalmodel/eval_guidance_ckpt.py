import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Allow running `python eval_guidance_ckpt.py ...` from within thermalmodel/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from thermalmodel.dataLoader import ThermalDataset
from thermalmodel.draw_thermal_fig import plot_thermal_grid_overlay
from thermalmodel.guidance_model import ThermalGuidanceNet


def _device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _maybe_temp_stats(ckpt: Dict) -> Optional[Dict[str, float]]:
    st = ckpt.get("stats")
    if isinstance(st, dict) and ("temp_min" in st) and ("temp_max" in st):
        try:
            return {"temp_min": float(st["temp_min"]), "temp_max": float(st["temp_max"])}
        except Exception:
            return None
    return None


def _denorm_temp_k(t01: torch.Tensor, st: Dict[str, float]) -> torch.Tensor:
    scale = st["temp_max"] - st["temp_min"]
    return t01 * scale + st["temp_min"]


def _spatial_gradient_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    # pred/gt: (B,1,H,W)
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

    px = F.conv2d(pred, sobel_x, padding=1)
    py = F.conv2d(pred, sobel_y, padding=1)
    gx = F.conv2d(gt, sobel_x, padding=1)
    gy = F.conv2d(gt, sobel_y, padding=1)
    return torch.mean(torch.abs(px - gx) + torch.abs(py - gy))


@dataclass
class Metrics:
    units: str
    mean_rmse: float
    min_rmse: float
    max_rmse: float
    mean_grad: float
    max_ae: float
    mean_rmspe_pct: float


@torch.no_grad()
def eval_ckpt_with_extremes(
    ckpt_path: str,
    *,
    batch_size: int = 8,
    seed: int = 0,
    prefer_device: str = "cuda",
    out_fig_dir: str = "",
) -> Tuple[Metrics, Dict]:
    device = _device(prefer=prefer_device)

    dataset = ThermalDataset(
        thermal_map_rel="Dataset/dataset/output/thermal/thermal_map",
        hotspot_cfg_rel="Dataset/dataset/output/thermal/hotspot_config",
        power_grid_size=128,
        temp_grid_size=64,
    )

    n_total = len(dataset)
    n_train = int(n_total * 0.9)
    n_test = n_total - n_train
    gen = torch.Generator().manual_seed(seed)
    _, test_set = torch.utils.data.random_split(dataset, [n_train, n_test], generator=gen)

    loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ThermalGuidanceNet(base=int(ckpt.get("base", 32))).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    st = _maybe_temp_stats(ckpt)

    # Accumulators
    rmse_sum = 0.0
    rmse_min = float("inf")
    rmse_max = 0.0
    grad_sum = 0.0
    max_ae = 0.0
    rmspe_sum = 0.0
    n = 0

    worst = None  # {rmse, i, j, pred, gt, units}
    best = None   # {rmse, i, j, pred, gt, units}

    for batch in loader:
        totalp = batch.get("total_power")

        power = batch["power"].to(device)
        layout = batch["layout"].to(device)
        temp = batch["temp"].to(device)
        totalp = totalp.to(device) if totalp is not None else None

        pred, _pred_avg = model(power, layout, totalp)

        # Evaluate metrics in CPU tensors to avoid mixing devices when saving figures
        pred_eval = pred.detach().cpu()
        temp_eval = temp.detach().cpu()

        units = "norm"
        if st is not None:
            pred_eval = _denorm_temp_k(pred_eval, st)
            temp_eval = _denorm_temp_k(temp_eval, st)
            units = "K"

        # Per-sample RMSE over H,W
        diff = pred_eval - temp_eval
        mse = torch.mean(diff * diff, dim=(1, 2, 3))
        rmse = torch.sqrt(mse)

        # Gradient metric (batch-level)
        grad = _spatial_gradient_loss(pred_eval, temp_eval)

        # Max absolute error over pixels in this batch
        batch_max_ae = float(diff.abs().max().item())
        max_ae = max(max_ae, batch_max_ae)

        # RMSPE (%), per-sample then average
        # Avoid div-by-zero; temps should be safely >0 in Kelvin, but keep epsilon.
        eps = 1e-6
        pe2 = torch.mean(((diff / (temp_eval.abs() + eps)) ** 2), dim=(1, 2, 3))
        rmspe = torch.sqrt(pe2) * 100.0

        for b in range(rmse.shape[0]):
            r = float(rmse[b].item())
            rmse_sum += r
            rmse_min = min(rmse_min, r)
            if r > rmse_max:
                rmse_max = r
                worst = {
                    "rmse": r,
                    "i": int(batch["i"][b]),
                    "j": int(batch["j"][b]),
                    "pred": pred_eval[b],
                    "gt": temp_eval[b],
                    "units": units,
                }
            if best is None or r < float(best["rmse"]):
                best = {
                    "rmse": r,
                    "i": int(batch["i"][b]),
                    "j": int(batch["j"][b]),
                    "pred": pred_eval[b],
                    "gt": temp_eval[b],
                    "units": units,
                }
            n += 1

        rmspe_sum += float(rmspe.mean().item())
        grad_sum += float(grad.item())

    metrics = Metrics(
        units="K" if st is not None else "norm",
        mean_rmse=rmse_sum / max(n, 1),
        min_rmse=rmse_min if n > 0 else 0.0,
        max_rmse=rmse_max,
        mean_grad=grad_sum / max(len(loader), 1),
        max_ae=max_ae,
        mean_rmspe_pct=rmspe_sum / max(len(loader), 1),
    )

    # Save extreme-case images
    if out_fig_dir:
        os.makedirs(out_fig_dir, exist_ok=True)

        def _save_one(tag: str, meta: Dict) -> Dict:
            i = int(meta["i"])
            j = int(meta["j"])
            score = float(meta["rmse"])

            flp = os.path.join(
                _PROJECT_ROOT,
                "Dataset/dataset/output/thermal/hotspot_config",
                f"system_{i}_config",
                "system.flp",
            )

            pred_w = meta["pred"]
            gt_w = meta["gt"]

            vmin = float(min(pred_w.min().item(), gt_w.min().item()))
            vmax = float(max(pred_w.max().item(), gt_w.max().item()))

            pred_path = os.path.join(out_fig_dir, f"{tag}_i{i}_j{j}_pred_rmse{score:.6f}.png")
            gt_path = os.path.join(out_fig_dir, f"{tag}_i{i}_j{j}_gt.png")

            plot_thermal_grid_overlay(
                flp,
                pred_w,
                pred_path,
                title=f"Pred {tag} i={i} j={j} RMSE={score:.6f}",
                vmin=vmin,
                vmax=vmax,
                units=meta["units"],
            )
            plot_thermal_grid_overlay(
                flp,
                gt_w,
                gt_path,
                title=f"GT {tag} i={i} j={j}",
                vmin=vmin,
                vmax=vmax,
                units=meta["units"],
            )

            meta_out = dict(meta)
            meta_out.pop("pred", None)
            meta_out.pop("gt", None)
            return meta_out

        out_meta: Dict[str, Dict] = {}
        if worst is not None:
            out_meta["worst"] = _save_one("worst", worst)
        if best is not None:
            out_meta["best"] = _save_one("best", best)
    else:
        out_meta = {}
        if worst is not None:
            out_meta["worst"] = {k: v for k, v in worst.items() if k not in ("pred", "gt")}
        if best is not None:
            out_meta["best"] = {k: v for k, v in best.items() if k not in ("pred", "gt")}

    return metrics, out_meta


@torch.no_grad()
def eval_ckpt_topk(
    ckpt_path: str,
    *,
    k: int = 10,
    batch_size: int = 8,
    seed: int = 0,
    prefer_device: str = "cuda",
    out_fig_dir: str = "",
) -> Tuple[Metrics, Dict]:
    device = _device(prefer=prefer_device)

    dataset = ThermalDataset(
        thermal_map_rel="Dataset/dataset/output/thermal/thermal_map",
        hotspot_cfg_rel="Dataset/dataset/output/thermal/hotspot_config",
        power_grid_size=128,
        temp_grid_size=64,
    )

    n_total = len(dataset)
    n_train = int(n_total * 0.9)
    n_test = n_total - n_train
    gen = torch.Generator().manual_seed(seed)
    _, test_set = torch.utils.data.random_split(dataset, [n_train, n_test], generator=gen)

    loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ThermalGuidanceNet(base=int(ckpt.get("base", 32))).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    st = _maybe_temp_stats(ckpt)

    # Accumulators
    rmse_sum = 0.0
    rmse_min = float("inf")
    rmse_max = 0.0
    grad_sum = 0.0
    max_ae = 0.0
    rmspe_sum = 0.0
    n = 0

    entries: List[Dict] = []  # {rmse, i, j, pred, gt, units}

    for batch in loader:
        totalp = batch.get("total_power")

        power = batch["power"].to(device)
        layout = batch["layout"].to(device)
        temp = batch["temp"].to(device)
        totalp = totalp.to(device) if totalp is not None else None

        pred, _pred_avg = model(power, layout, totalp)

        # Evaluate metrics in CPU tensors to avoid mixing devices when saving figures
        pred_eval = pred.detach().cpu()
        temp_eval = temp.detach().cpu()

        units = "norm"
        if st is not None:
            pred_eval = _denorm_temp_k(pred_eval, st)
            temp_eval = _denorm_temp_k(temp_eval, st)
            units = "K"

        # Per-sample RMSE over H,W
        diff = pred_eval - temp_eval
        mse = torch.mean(diff * diff, dim=(1, 2, 3))
        rmse = torch.sqrt(mse)

        # Gradient metric (batch-level)
        grad = _spatial_gradient_loss(pred_eval, temp_eval)

        # Max absolute error over pixels in this batch
        batch_max_ae = float(diff.abs().max().item())
        max_ae = max(max_ae, batch_max_ae)

        # RMSPE (%), per-sample then average
        eps = 1e-6
        pe2 = torch.mean(((diff / (temp_eval.abs() + eps)) ** 2), dim=(1, 2, 3))
        rmspe = torch.sqrt(pe2) * 100.0

        for b in range(rmse.shape[0]):
            r = float(rmse[b].item())
            rmse_sum += r
            rmse_min = min(rmse_min, r)
            rmse_max = max(rmse_max, r)

            entries.append(
                {
                    "rmse": r,
                    "i": int(batch["i"][b]),
                    "j": int(batch["j"][b]),
                    "pred": pred_eval[b],
                    "gt": temp_eval[b],
                    "units": units,
                }
            )
            n += 1

        rmspe_sum += float(rmspe.mean().item())
        grad_sum += float(grad.item())

    metrics = Metrics(
        units="K" if st is not None else "norm",
        mean_rmse=rmse_sum / max(n, 1),
        min_rmse=rmse_min if n > 0 else 0.0,
        max_rmse=rmse_max,
        mean_grad=grad_sum / max(len(loader), 1),
        max_ae=max_ae,
        mean_rmspe_pct=rmspe_sum / max(len(loader), 1),
    )

    entries.sort(key=lambda d: float(d["rmse"]))
    k = max(0, min(int(k), len(entries)))
    best_k = entries[:k]
    worst_k = list(reversed(entries[-k:]))

    def _save_many(tag: str, metas: List[Dict]) -> List[Dict]:
        out: List[Dict] = []
        for rank, meta in enumerate(metas, start=1):
            i = int(meta["i"])
            j = int(meta["j"])
            score = float(meta["rmse"])

            flp = os.path.join(
                _PROJECT_ROOT,
                "Dataset/dataset/output/thermal/hotspot_config",
                f"system_{i}_config",
                "system.flp",
            )

            pred_w = meta["pred"]
            gt_w = meta["gt"]

            vmin = float(min(pred_w.min().item(), gt_w.min().item()))
            vmax = float(max(pred_w.max().item(), gt_w.max().item()))

            pred_path = os.path.join(out_fig_dir, f"{tag}_r{rank:02d}_i{i}_j{j}_pred_rmse{score:.6f}.png")
            gt_path = os.path.join(out_fig_dir, f"{tag}_r{rank:02d}_i{i}_j{j}_sim.png")

            plot_thermal_grid_overlay(
                flp,
                pred_w,
                pred_path,
                title=f"Pred {tag} r={rank:02d} i={i} j={j} RMSE={score:.6f}",
                vmin=vmin,
                vmax=vmax,
                units=meta["units"],
            )
            plot_thermal_grid_overlay(
                flp,
                gt_w,
                gt_path,
                title=f"Sim {tag} r={rank:02d} i={i} j={j} RMSE={score:.6f}",
                vmin=vmin,
                vmax=vmax,
                units=meta["units"],
            )

            meta_out = dict(meta)
            meta_out.pop("pred", None)
            meta_out.pop("gt", None)
            out.append(meta_out)
        return out

    out_meta: Dict[str, object] = {
        "best": [{k: v for k, v in m.items() if k not in ("pred", "gt")} for m in best_k],
        "worst": [{k: v for k, v in m.items() if k not in ("pred", "gt")} for m in worst_k],
    }

    if out_fig_dir and k > 0:
        os.makedirs(out_fig_dir, exist_ok=True)
        out_meta["best"] = _save_many("best", best_k)
        out_meta["worst"] = _save_many("worst", worst_k)

    return metrics, out_meta


@torch.no_grad()
def benchmark_model(
    ckpt_path: str,
    *,
    batch_size: int = 8,
    warmup_batches: int = 10,
    timed_batches: int = 50,
    prefer_device: str = "cuda",
) -> Tuple[float, float, str]:
    device = _device(prefer=prefer_device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ThermalGuidanceNet(base=int(ckpt.get("base", 32))).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    power = torch.zeros(batch_size, 1, 128, 128, device=device)
    layout = torch.zeros(batch_size, 1, 128, 128, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    for _ in range(warmup_batches):
        _ = model(power, layout, None)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(timed_batches):
        _ = model(power, layout, None)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_batch_time_s = (t1 - t0) / max(timed_batches, 1)
    avg_case_time_s = avg_batch_time_s / max(batch_size, 1)
    return avg_case_time_s, avg_batch_time_s, str(device)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_bs", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out_fig_dir", type=str, default="")
    ap.add_argument("--topk", type=int, default=0, help="If >0, save top-k best/worst cases by per-sample RMSE")

    ap.add_argument("--bench", action="store_true", help="benchmark pure model latency")
    ap.add_argument("--bench_bs", type=int, default=8)
    ap.add_argument("--warmup_batches", type=int, default=10)
    ap.add_argument("--timed_batches", type=int, default=50)
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    if args.topk and args.topk > 0:
        metrics, meta = eval_ckpt_topk(
            args.ckpt,
            k=args.topk,
            batch_size=args.eval_bs,
            seed=args.seed,
            prefer_device=args.device,
            out_fig_dir=args.out_fig_dir,
        )
    else:
        metrics, meta = eval_ckpt_with_extremes(
            args.ckpt,
            batch_size=args.eval_bs,
            seed=args.seed,
            prefer_device=args.device,
            out_fig_dir=args.out_fig_dir,
        )

    bench_txt = ""
    if args.bench:
        avg_case_s, avg_batch_s, dev = benchmark_model(
            args.ckpt,
            batch_size=args.bench_bs,
            warmup_batches=args.warmup_batches,
            timed_batches=args.timed_batches,
            prefer_device=args.device,
        )
        bench_txt = f" | speed({dev}) avg_case_ms={avg_case_s*1e3:.3f} avg_batch_ms={avg_batch_s*1e3:.3f}"

    def _pick_one(d: object, which: str) -> Dict:
        if isinstance(d, dict):
            v = d.get(which)
            if isinstance(v, dict):
                return v
            return {}
        return {}

    def _pick_first_list(d: object, which: str) -> Dict:
        if isinstance(d, dict):
            v = d.get(which)
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                return v[0]
        return {}

    worst_d = _pick_one(meta, "worst")
    best_d = _pick_one(meta, "best")

    # topk mode returns lists; fall back to first element for summary printing
    if not worst_d:
        worst_d = _pick_first_list(meta, "worst")
    if not best_d:
        best_d = _pick_first_list(meta, "best")

    wi = worst_d.get("i", "")
    wj = worst_d.get("j", "")
    wr = worst_d.get("rmse", "")

    bi = best_d.get("i", "")
    bj = best_d.get("j", "")
    br = best_d.get("rmse", "")

    # Single-line summary for logs
    print(
        "metrics"
        f" units={metrics.units}"
        f" mean_rmse={metrics.mean_rmse:.6f}"
        f" min_rmse={metrics.min_rmse:.6f}"
        f" max_rmse={metrics.max_rmse:.6f}"
        f" mean_grad={metrics.mean_grad:.6f}"
        f" max_ae={metrics.max_ae:.6f}"
        f" mean_rmspe_pct={metrics.mean_rmspe_pct:.6f}"
        f" worst_i={wi} worst_j={wj} worst_rmse={wr}"
        f" best_i={bi} best_j={bj} best_rmse={br}"
        f" ckpt={args.ckpt}"
        + bench_txt
    )


if __name__ == "__main__":
    main()
