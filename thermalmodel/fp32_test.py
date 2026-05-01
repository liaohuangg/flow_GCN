import argparse
import os
import sys
import time
from typing import Any, Dict

import torch

# Allow running `python fp32_ptqfp16_test.py ...` from within thermalmodel/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from thermalmodel.eval_guidance_ckpt import eval_ckpt_with_extremes


def _extract_train_params(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "epoch",
        "base",
        "batch_size",
        "lr",
        "grad_w",
        "seed",
        "ckpt_tag",
        "qat",
        "amp",
        "amp_dtype",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        if k in ckpt:
            v = ckpt[k]
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[k] = v
            else:
                out[k] = str(v)
    return out


def _fmt_metrics_line(name: str, metrics, meta: Dict[str, Any], ckpt_path: str) -> str:
    wi = meta.get("worst", {}).get("i", "")
    wj = meta.get("worst", {}).get("j", "")
    wr = meta.get("worst", {}).get("rmse", "")

    bi = meta.get("best", {}).get("i", "")
    bj = meta.get("best", {}).get("j", "")
    br = meta.get("best", {}).get("rmse", "")

    return (
        f"[{name}]\n"
        + "metrics"
        + f" units={metrics.units}"
        + f" mean_rmse={metrics.mean_rmse:.6f}"
        + f" min_rmse={metrics.min_rmse:.6f}"
        + f" max_rmse={metrics.max_rmse:.6f}"
        + f" mean_grad={metrics.mean_grad:.6f}"
        + f" max_ae={metrics.max_ae:.6f}"
        + f" mean_rmspe_pct={metrics.mean_rmspe_pct:.6f}"
        + f" worst_i={wi} worst_j={wj} worst_rmse={wr}"
        + f" best_i={bi} best_j={bj} best_rmse={br}"
        + f" ckpt={ckpt_path}"
        + "\n"
    )


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fp32_ckpt",
        type=str,
        default="/root/placement/flow_GCN/thermalmodel/checkpoints/guidance_b96_lr2e-4_ep200_20260430_resume/guidance_net_b96_lr2e-4_totalp_avg_resume_ep0195_seed0_bs8_lr2e-04_base96_gw0p1.pth",
    )
    ap.add_argument(
        "--ptq_fp16_ckpt",
        type=str,
        default="/root/placement/flow_GCN/thermalmodel/checkpoints/guidance_b96_lr2e-4_ep195_PTQ/guidance_net_b96_lr2e-4_totalp_avg_resume_ep0195_seed0_bs8_lr2e-04_base96_gw0p1_fp16.pth",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_bs", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument(
        "--out_metrics_log",
        type=str,
        default="/root/placement/flow_GCN/thermalmodel/test_result/fp32_ptqfp16_test.log",
    )

    # Timing mode (first N cases in thermal_map order)
    ap.add_argument("--time_n_cases", type=int, default=0, help="If >0, run timing over first N cases and write per_case_time.log")
    ap.add_argument(
        "--thermal_map_root",
        type=str,
        default="/root/placement/flow_GCN/Dataset/dataset/output/thermal/thermal_map",
    )
    ap.add_argument(
        "--out_time_log",
        type=str,
        default="/root/placement/flow_GCN/thermalmodel/test_result/per_case_time.log",
    )
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--out_fig_dir", type=str, default="")
    ap.add_argument("--append", action="store_true", help="append logs instead of overwrite")
    return ap


def _list_first_n_cases(thermal_map_root: str, n: int):
    import os
    import re

    power_dir = os.path.join(thermal_map_root, "powercsv")
    pat = re.compile(r"system_power_(\d+)_(\d+)\.csv$")
    cases = []
    for fn in os.listdir(power_dir):
        m = pat.match(fn)
        if not m:
            continue
        cases.append((int(m.group(1)), int(m.group(2))))
    cases.sort()
    return cases[:n]


def _mk_timing_dataset(*, thermal_map_root: str, cases):
    from thermalmodel.dataLoader import ThermalDataset

    return ThermalDataset(
        thermal_map_rel=os.path.relpath(thermal_map_root, start=_PROJECT_ROOT),
        hotspot_cfg_rel="Dataset/dataset/output/thermal/hotspot_config",
        power_grid_size=128,
        temp_grid_size=64,
        cases=cases,
    )


@torch.no_grad()
def _time_model_ckpt(
    ckpt_path: str,
    *,
    n_cases: int,
    eval_bs: int,
    seed: int,
    device: str,
    thermal_map_root: str,
    num_workers: int,
    pin_memory: bool,
):
    from thermalmodel.guidance_model import ThermalGuidanceNet

    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ThermalGuidanceNet(base=int(ckpt.get("base", 32))).to(dev)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    cases = _list_first_n_cases(thermal_map_root, n_cases)
    ds_t0 = time.perf_counter()
    dataset = _mk_timing_dataset(thermal_map_root=thermal_map_root, cases=cases)
    ds_t1 = time.perf_counter()
    ds_build_s = ds_t1 - ds_t0

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # timing totals
    t_wall0 = time.perf_counter()
    t_dl_s = 0.0
    t_fwd_s = 0.0
    n_seen = 0

    if dev.type == "cuda":
        torch.cuda.synchronize()

    for batch in loader:
        # DataLoader time: measured as time spent between end of last forward and now.
        # Since we can't intercept worker threads, we approximate by timing the host-to-device + batch prep.
        dl0 = time.perf_counter()
        power = batch["power"].to(dev, non_blocking=False)
        layout = batch["layout"].to(dev, non_blocking=False)
        totalp = batch.get("total_power")
        totalp = totalp.to(dev, non_blocking=False) if totalp is not None else None
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dl1 = time.perf_counter()
        t_dl_s += (dl1 - dl0)

        # forward time
        if dev.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(power, layout, totalp)
            end.record()
            torch.cuda.synchronize()
            t_fwd_s += start.elapsed_time(end) / 1e3
        else:
            f0 = time.perf_counter()
            _ = model(power, layout, totalp)
            t_fwd_s += time.perf_counter() - f0

        n_seen += power.shape[0]

    if dev.type == "cuda":
        torch.cuda.synchronize()

    t_wall1 = time.perf_counter()
    t_total_s = t_wall1 - t_wall0

    return {
        "ckpt": ckpt_path,
        "device": str(dev),
        "n_cases": int(n_seen),
        "eval_bs": int(eval_bs),
        "seed": int(seed),
        "thermal_map_root": thermal_map_root,
        "cases_desc": f"first {n_cases} cases from thermal_map/powercsv sorted by (i,j)",
        "dataset_build_s": float(ds_build_s),
        "total_wall_s": float(t_total_s),
        "forward_only_s": float(t_fwd_s),
        "dataloader_like_s": float(t_dl_s),
        "per_case_total_ms": float(t_total_s * 1e3 / max(n_seen, 1)),
        "per_case_forward_ms": float(t_fwd_s * 1e3 / max(n_seen, 1)),
        "per_case_dataloader_like_ms": float(t_dl_s * 1e3 / max(n_seen, 1)),
        "total_minus_forward_s": float(t_total_s - t_fwd_s),
        "total_minus_forward_minus_dl_s": float(t_total_s - t_fwd_s - t_dl_s),
    }


def main() -> None:
    args = build_argparser().parse_args()

    # 1) metrics log (same as before)
    os.makedirs(os.path.dirname(args.out_metrics_log), exist_ok=True)

    ckpt_fp32 = torch.load(args.fp32_ckpt, map_location="cpu")
    ckpt_ptq = torch.load(args.ptq_fp16_ckpt, map_location="cpu")

    fp32_train_params = _extract_train_params(ckpt_fp32)
    ptq_train_params = _extract_train_params(ckpt_ptq)

    m_fp32, meta_fp32 = eval_ckpt_with_extremes(
        args.fp32_ckpt,
        batch_size=args.eval_bs,
        seed=args.seed,
        prefer_device=args.device,
        out_fig_dir=args.out_fig_dir,
    )
    m_ptq, meta_ptq = eval_ckpt_with_extremes(
        args.ptq_fp16_ckpt,
        batch_size=args.eval_bs,
        seed=args.seed,
        prefer_device=args.device,
        out_fig_dir=args.out_fig_dir,
    )

    lines = []
    lines.append("# fp32 vs ptq-fp16 test results")
    lines.append(f"date={time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("env=conda:chipdiffusion")
    lines.append(f"project_root={_PROJECT_ROOT}")
    lines.append("")

    lines.append("[test_params]")
    lines.append(f"seed={args.seed}")
    lines.append(f"eval_bs={args.eval_bs}")
    lines.append(f"device={args.device}")
    lines.append("split=train=0.9 test=0.1 (torch.utils.data.random_split generator seed)")
    lines.append("")

    lines.append("[units]")
    lines.append("All error metrics are in Kelvin (K). Note: temperature differences in K equal differences in °C.")
    lines.append("")

    lines.append("[metric_defs]")
    lines.append("RMSE: sqrt(mean((pred - gt)^2)) over all pixels per sample")
    lines.append("Max AE: max(abs(pred - gt)) over all pixels across the whole test set")
    lines.append("RMSPE (%): 100 * sqrt(mean(((pred - gt)/gt)^2)) over pixels per sample, then averaged over batches")
    lines.append("mean_grad: mean(|Sobel(pred) - Sobel(gt)|) (same as eval_guidance_ckpt.py)")
    lines.append("")

    lines.append("[train_params]")
    lines.append(f"fp32: {fp32_train_params}")
    lines.append(f"ptq-fp16: {ptq_train_params}")
    lines.append("")

    lines.append(_fmt_metrics_line("fp32", m_fp32, meta_fp32, args.fp32_ckpt))
    lines.append(_fmt_metrics_line("ptq-fp16", m_ptq, meta_ptq, args.ptq_fp16_ckpt))

    with open(args.out_metrics_log, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"WROTE {args.out_metrics_log}")

    # 2) timing log (optional)
    if args.time_n_cases and args.time_n_cases > 0:
        fp32_t = _time_model_ckpt(
            args.fp32_ckpt,
            n_cases=args.time_n_cases,
            eval_bs=args.eval_bs,
            seed=args.seed,
            device=args.device,
            thermal_map_root=args.thermal_map_root,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        ptq_t = _time_model_ckpt(
            args.ptq_fp16_ckpt,
            n_cases=args.time_n_cases,
            eval_bs=args.eval_bs,
            seed=args.seed,
            device=args.device,
            thermal_map_root=args.thermal_map_root,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

        mode = "a" if args.append else "w"
        with open(args.out_time_log, mode, encoding="utf-8") as f:
            if not args.append:
                f.write(f"# per-case timing (model + hotspot)\n")
            f.write(f"\n[model_timing]\n")
            f.write(f"date={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("env=conda:chipdiffusion\n")
            f.write(f"cases={fp32_t['cases_desc']}\n")
            f.write(f"units=seconds (s) and milliseconds (ms)\n\n")

            for name, t in [("fp32", fp32_t), ("ptq-fp16", ptq_t)]:
                f.write(f"[{name}]\n")
                f.write(f"ckpt={t['ckpt']}\n")
                f.write(f"device={t['device']}\n")
                f.write(f"n_cases={t['n_cases']} eval_bs={t['eval_bs']} num_workers={args.num_workers} pin_memory={bool(args.pin_memory)}\n")
                f.write(f"dataset_build_s={t['dataset_build_s']:.6f}\n")
                f.write(f"total_wall_s={t['total_wall_s']:.6f}\n")
                f.write(f"forward_only_s={t['forward_only_s']:.6f}\n")
                f.write(f"dataloader_like_s={t['dataloader_like_s']:.6f}\n")
                f.write(f"per_case_total_ms={t['per_case_total_ms']:.6f}\n")
                f.write(f"per_case_forward_ms={t['per_case_forward_ms']:.6f}\n")
                f.write(f"per_case_dataloader_like_ms={t['per_case_dataloader_like_ms']:.6f}\n")
                f.write(f"total_minus_forward_s={t['total_minus_forward_s']:.6f}\n")
                f.write(f"total_minus_forward_minus_dl_s={t['total_minus_forward_minus_dl_s']:.6f}\n")
                f.write("\n")

        print(f"WROTE {args.out_time_log}")


if __name__ == "__main__":
    main()
