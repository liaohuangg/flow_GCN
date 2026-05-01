import argparse
import glob
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import sys

import torch
import torch.nn.functional as F

# Allow running `python ablation_eval.py ...` from within thermalmodel/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from thermalmodel.dataLoader import ThermalDataset
from thermalmodel.guidance_model import ThermalGuidanceNet
from thermalmodel.guidance_pipeline import export_fp16_from_fp32


@dataclass
class Metrics:
    units: str
    mean_mse: float
    mean_rmse: float
    mean_grad: float
    max_rmse: float


@dataclass
class Bench:
    device: str
    batch_size: int
    warmup_batches: int
    timed_batches: int
    avg_batch_time_s: float
    avg_case_time_s: float


@dataclass
class VariantResult:
    name: str
    ckpt_path: str
    metrics: Metrics
    bench: Bench


def _device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _maybe_temp_stats(ckpt: Dict[str, Any]) -> Optional[Dict[str, float]]:
    st = ckpt.get("stats")
    if isinstance(st, dict) and ("temp_min" in st) and ("temp_max" in st):
        try:
            return {"temp_min": float(st["temp_min"]), "temp_max": float(st["temp_max"])}
        except Exception:
            return None
    return None


def _denorm_temp_k(t01: torch.Tensor, st: Dict[str, float]) -> torch.Tensor:
    scale = (st["temp_max"] - st["temp_min"])
    return t01 * scale + st["temp_min"]


def _spatial_gradient_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    # Similar to guidance_model.py behavior: use Sobel-based gradient loss.
    # pred/gt: (B,1,H,W)
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

    px = F.conv2d(pred, sobel_x, padding=1)
    py = F.conv2d(pred, sobel_y, padding=1)
    gx = F.conv2d(gt, sobel_x, padding=1)
    gy = F.conv2d(gt, sobel_y, padding=1)
    return torch.mean(torch.abs(px - gx) + torch.abs(py - gy))


def eval_ckpt(
    ckpt_path: str,
    batch_size: int = 8,
    seed: int = 0,
    prefer_device: str = "cuda",
) -> Metrics:
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

    mse_sum = 0.0
    rmse_sum = 0.0
    grad_sum = 0.0
    max_rmse = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            totalp = batch.get("total_power")

            power = batch["power"].to(device)
            layout = batch["layout"].to(device)
            temp = batch["temp"].to(device)
            totalp = totalp.to(device) if totalp is not None else None

            pred, _pred_avg = model(power, layout, totalp)

            pred_eval = pred.detach().cpu()
            temp_eval = temp.detach().cpu()

            units = "norm"
            if st is not None:
                pred_eval = _denorm_temp_k(pred_eval, st)
                temp_eval = _denorm_temp_k(temp_eval, st)
                units = "K"

            mse = F.mse_loss(pred_eval, temp_eval, reduction="none").mean(dim=(1, 2, 3))
            rmse = torch.sqrt(mse)
            grad = _spatial_gradient_loss(pred_eval, temp_eval)

            for b in range(mse.shape[0]):
                mse_sum += float(mse[b].item())
                rmse_sum += float(rmse[b].item())
                max_rmse = max(max_rmse, float(rmse[b].item()))
                n += 1

            grad_sum += float(grad.item())

    return Metrics(
        units=units,
        mean_mse=mse_sum / max(n, 1),
        mean_rmse=rmse_sum / max(n, 1),
        mean_grad=grad_sum / max(len(loader), 1),
        max_rmse=max_rmse,
    )


def benchmark_ckpt(
    ckpt_path: str,
    batch_size: int = 8,
    warmup_batches: int = 10,
    timed_batches: int = 50,
    prefer_device: str = "cuda",
) -> Bench:
    device = _device(prefer=prefer_device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ThermalGuidanceNet(base=int(ckpt.get("base", 32))).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # Dummy inputs for pure model latency.
    power = torch.zeros(batch_size, 1, 128, 128, device=device)
    layout = torch.zeros(batch_size, 1, 128, 128, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        # Warmup
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

    avg_batch = (t1 - t0) / max(timed_batches, 1)
    avg_case = avg_batch / max(batch_size, 1)

    return Bench(
        device=str(device),
        batch_size=batch_size,
        warmup_batches=warmup_batches,
        timed_batches=timed_batches,
        avg_batch_time_s=avg_batch,
        avg_case_time_s=avg_case,
    )


def _find_ckpts(ckpt_dir: str, tag: str) -> List[str]:
    # New naming scheme: guidance_net_{tag}_epXXXX_...
    pat_new = os.path.join(ckpt_dir, f"guidance_net_{tag}_ep*_seed*_bs*_lr*_base*_gw*.pth")
    out = sorted(glob.glob(pat_new))
    if out:
        return out

    # Backward-compatible: older fp32 naming (no tag/seed fields) e.g. guidance_net_epoch50_bs8_lr5e-04_base64_gradw0.1.pth
    # Only used for tag=="fp32".
    if tag == "fp32":
        pat_old = os.path.join(os.path.dirname(ckpt_dir), "guidance_net_epoch*_bs*_lr*_base*_gradw*.pth")
        out_old = sorted(glob.glob(pat_old))
        return out_old

    return []


def _best_by_mean_rmse(ckpts: List[str], batch_size: int, seed: int, prefer_device: str) -> Tuple[str, Metrics]:
    best_path = ""
    best_metrics: Optional[Metrics] = None

    for p in ckpts:
        m = eval_ckpt(p, batch_size=batch_size, seed=seed, prefer_device=prefer_device)
        if best_metrics is None or m.mean_rmse < best_metrics.mean_rmse:
            best_metrics = m
            best_path = p

    assert best_metrics is not None
    return best_path, best_metrics


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32_dir", type=str, default="")
    ap.add_argument("--qat_dir", type=str, default="")
    ap.add_argument("--ptq_dir", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_bs", type=int, default=8)
    ap.add_argument("--bench_bs", type=int, default=8)
    ap.add_argument("--warmup_batches", type=int, default=10)
    ap.add_argument("--timed_batches", type=int, default=50)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    base_dir = os.path.dirname(__file__)
    fp32_dir = args.fp32_dir or os.path.join(base_dir, "checkpoints", "fp32")
    qat_dir = args.qat_dir or os.path.join(base_dir, "checkpoints", "qat")
    ptq_dir = args.ptq_dir or os.path.join(base_dir, "checkpoints", "ptq")

    fp32_ckpts = _find_ckpts(fp32_dir, "fp32")
    qat_ckpts = _find_ckpts(qat_dir, "qat")

    if len(fp32_ckpts) == 0:
        raise SystemExit(f"No fp32 checkpoints found in {fp32_dir}")
    if len(qat_ckpts) == 0:
        raise SystemExit(f"No qat checkpoints found in {qat_dir}")

    # 1) Select best fp32 and best qat by mean RMSE
    best_fp32_path, best_fp32_metrics = _best_by_mean_rmse(
        fp32_ckpts, batch_size=args.eval_bs, seed=args.seed, prefer_device=args.device
    )
    best_qat_path, best_qat_metrics = _best_by_mean_rmse(
        qat_ckpts, batch_size=args.eval_bs, seed=args.seed, prefer_device=args.device
    )

    # 2) Export fp16 from best fp32 (PTQ-fp16)
    stem = os.path.splitext(os.path.basename(best_fp32_path))[0]
    ptq_fp16_path = os.path.join(ptq_dir, f"{stem}_fp16.pth")
    export_fp16_from_fp32(best_fp32_path, ptq_fp16_path)

    # 3) Evaluate + benchmark all three variants
    results: List[VariantResult] = []

    fp32_bench = benchmark_ckpt(
        best_fp32_path,
        batch_size=args.bench_bs,
        warmup_batches=args.warmup_batches,
        timed_batches=args.timed_batches,
        prefer_device=args.device,
    )
    results.append(VariantResult(name="fp32(best)", ckpt_path=best_fp32_path, metrics=best_fp32_metrics, bench=fp32_bench))

    qat_bench = benchmark_ckpt(
        best_qat_path,
        batch_size=args.bench_bs,
        warmup_batches=args.warmup_batches,
        timed_batches=args.timed_batches,
        prefer_device=args.device,
    )
    results.append(VariantResult(name="qat(best)", ckpt_path=best_qat_path, metrics=best_qat_metrics, bench=qat_bench))

    ptq_metrics = eval_ckpt(ptq_fp16_path, batch_size=args.eval_bs, seed=args.seed, prefer_device=args.device)
    ptq_bench = benchmark_ckpt(
        ptq_fp16_path,
        batch_size=args.bench_bs,
        warmup_batches=args.warmup_batches,
        timed_batches=args.timed_batches,
        prefer_device=args.device,
    )
    results.append(VariantResult(name="ptq_fp16(from fp32 best)", ckpt_path=ptq_fp16_path, metrics=ptq_metrics, bench=ptq_bench))

    # Print summary
    print("==== Ablation Summary (select best ckpt by mean RMSE) ====")
    print(f"seed={args.seed} eval_bs={args.eval_bs} bench_bs={args.bench_bs} device={args.device}")
    for r in results:
        m = r.metrics
        b = r.bench
        print(
            f"[{r.name}]\n"
            f"  ckpt: {r.ckpt_path}\n"
            f"  metrics({m.units}): mean_rmse={m.mean_rmse:.6f} mean_mse={m.mean_mse:.6f} mean_grad={m.mean_grad:.6f} max_rmse={m.max_rmse:.6f}\n"
            f"  bench({b.device}): avg_case_ms={b.avg_case_time_s*1e3:.3f} avg_batch_ms={b.avg_batch_time_s*1e3:.3f}"
        )

    if args.out_json:
        payload = {
            "cfg": {
                "seed": args.seed,
                "eval_bs": args.eval_bs,
                "bench_bs": args.bench_bs,
                "warmup_batches": args.warmup_batches,
                "timed_batches": args.timed_batches,
                "device": args.device,
            },
            "best_fp32": {"ckpt": best_fp32_path, "metrics": best_fp32_metrics.__dict__},
            "best_qat": {"ckpt": best_qat_path, "metrics": best_qat_metrics.__dict__},
            "ptq_fp16": {"ckpt": ptq_fp16_path},
            "results": [
                {
                    "name": r.name,
                    "ckpt": r.ckpt_path,
                    "metrics": r.metrics.__dict__,
                    "bench": r.bench.__dict__,
                }
                for r in results
            ],
        }
        _write_json(args.out_json, payload)
        print(f"[out] wrote: {args.out_json}")


if __name__ == "__main__":
    main()
