import os
import sys
from datetime import datetime

import torch

# Allow running `python test.py` from within thermalmodel/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from thermalmodel.dataLoader import ThermalDataset
from thermalmodel.q_model import ThermalUNetMultiTask
from thermalmodel.draw_thermal_fig import plot_thermal_grid_overlay


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _log(msg: str, fp):
    fp.write(msg + "\n")
    fp.flush()
    print(msg)


def _sample_score(pred_grid: torch.Tensor, true_grid: torch.Tensor) -> torch.Tensor:
    # per-sample RMSE over H,W
    diff = pred_grid - true_grid
    return torch.sqrt(torch.mean(diff * diff, dim=(1, 2, 3)))


def main():
    device = _device()

    # Data
    dataset = ThermalDataset(
        thermal_map_rel="Dataset/dataset/output/thermal/thermal_map",
        hotspot_cfg_rel="Dataset/dataset/output/thermal/hotspot_config",
        grid_size=64,
    )

    n_total = len(dataset)
    n_train = int(n_total * 0.9)
    n_val = n_total - n_train

    # Use a deterministic split for testing so results are reproducible
    gen = torch.Generator().manual_seed(0)
    train_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=gen)

    batch_size = 8
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", "thermal_unet_epoch200.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = ThermalUNetMultiTask(in_channels=3, base=32).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Logging
    out_dir = os.path.join(os.path.dirname(__file__), "test_result")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "result.log")

    fig_dir = os.path.join(out_dir, "test_thermal_fig")
    os.makedirs(fig_dir, exist_ok=True)

    # Metrics accumulators
    grid_abs_sum = 0.0
    grid_sq_sum = 0.0
    avg_abs_sum = 0.0
    avg_sq_sum = 0.0
    consistency_sum = 0.0
    gt_gap_sum = 0.0

    n_grid_vals = 0
    n_avg_vals = 0

    pred_grid_min = float("inf")
    pred_grid_max = float("-inf")
    gt_grid_min = float("inf")
    gt_grid_max = float("-inf")

    # For each block of 100 test cases, save the worst sample (largest grid RMSE)
    block_size = 100
    block_idx = 0
    in_block = 0
    best_worst = None  # dict with score and tensors

    def _flush_block():
        nonlocal best_worst, block_idx
        if not best_worst:
            return
        i = int(best_worst["i"])
        j = int(best_worst["j"])
        score = float(best_worst["score"])
        flp = os.path.join(
            _PROJECT_ROOT,
            "Dataset/dataset/output/thermal/hotspot_config",
            f"system_{i}_config",
            "system.flp",
        )

        pred = best_worst["pred"]
        gt = best_worst["gt"]

        # de-normalize to Kelvin if stats are available
        if (temp_min is not None) and (temp_max is not None):
            scale = (temp_max - temp_min)
            pred = pred * scale + temp_min
            gt = gt * scale + temp_min

        pred_path = os.path.join(fig_dir, f"block{block_idx:03d}_i{i}_j{j}_pred_rmse{score:.6f}.png")
        gt_path = os.path.join(fig_dir, f"block{block_idx:03d}_i{i}_j{j}_gt.png")

        # Use shared color scale for fair visual comparison (per-case)
        vmin = float(min(pred.min().item(), gt.min().item()))
        vmax = float(max(pred.max().item(), gt.max().item()))

        plot_thermal_grid_overlay(
            flp,
            pred,
            pred_path,
            title=f"Pred (block {block_idx}, i={i}, j={j}) RMSE={score:.6f}",
            vmin=vmin,
            vmax=vmax,
            units="K",
        )
        plot_thermal_grid_overlay(
            flp,
            gt,
            gt_path,
            title=f"GT (block {block_idx}, i={i}, j={j})",
            vmin=vmin,
            vmax=vmax,
            units="K",
        )

        best_worst = None

    with open(log_path, "w", encoding="utf-8") as fp:
        _log("==== Thermal Test ====", fp)
        _log(f"time: {datetime.now().isoformat(timespec='seconds')}", fp)
        _log(f"device: {device} (cuda_available={torch.cuda.is_available()})", fp)
        if torch.cuda.is_available():
            try:
                _log(f"cuda_device: {torch.cuda.get_device_name(0)}", fp)
            except Exception:
                pass
        _log(f"checkpoint: {ckpt_path}", fp)
        _log(f"dataset_total: {n_total} | train: {n_train} | test: {n_val}", fp)
        _log(f"batch_size: {batch_size} | test_batches: {len(test_loader)}", fp)

        stats = ckpt.get("stats")
        temp_min = None
        temp_max = None
        if isinstance(stats, dict):
            _log(f"stats: {stats}", fp)
            try:
                temp_min = float(stats["temp_min"])
                temp_max = float(stats["temp_max"])
            except Exception:
                temp_min = None
                temp_max = None

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                power = batch["power"].to(device)
                layout = batch["layout"].to(device)
                totalp = batch["total_power"].to(device)
                temp = batch["temp"].to(device)
                avg = batch["avg_temp"].to(device)

                pred_grid, pred_avg = model(power, layout, totalp)

                # grid metrics
                diff_grid = (pred_grid - temp)
                grid_abs_sum += diff_grid.abs().sum().item()
                grid_sq_sum += (diff_grid ** 2).sum().item()
                n_grid_vals += diff_grid.numel()

                # avg metrics
                diff_avg = (pred_avg - avg)
                avg_abs_sum += diff_avg.abs().sum().item()
                avg_sq_sum += (diff_avg ** 2).sum().item()
                n_avg_vals += diff_avg.numel()

                # consistency
                pred_grid_mean = pred_grid.mean(dim=(2, 3))
                true_grid_mean = temp.mean(dim=(2, 3))
                consistency_sum += (pred_grid_mean - pred_avg).abs().sum().item()
                gt_gap_sum += (true_grid_mean - avg).abs().sum().item()

                # ranges
                pred_grid_min = min(pred_grid_min, float(pred_grid.min().item()))
                pred_grid_max = max(pred_grid_max, float(pred_grid.max().item()))
                gt_grid_min = min(gt_grid_min, float(temp.min().item()))
                gt_grid_max = max(gt_grid_max, float(temp.max().item()))

                # per-sample selection within each 100-case block
                scores = _sample_score(pred_grid, temp).detach().cpu()
                for b in range(scores.shape[0]):
                    if in_block == 0 and best_worst is not None:
                        _flush_block()
                        block_idx += 1

                    score = float(scores[b].item())
                    i = int(batch["i"][b])
                    j = int(batch["j"][b])

                    if (best_worst is None) or (score > float(best_worst["score"])):
                        best_worst = {
                            "score": score,
                            "i": i,
                            "j": j,
                            "pred": pred_grid[b].detach().cpu(),
                            "gt": temp[b].detach().cpu(),
                        }

                    in_block += 1
                    if in_block >= block_size:
                        _flush_block()
                        block_idx += 1
                        in_block = 0

                if batch_idx % 20 == 0:
                    _log(f"[test] it {batch_idx:04d}/{len(test_loader)-1:04d}", fp)

        # flush last partial block
        if best_worst is not None:
            _flush_block()

        grid_mae = grid_abs_sum / max(n_grid_vals, 1)
        grid_rmse = (grid_sq_sum / max(n_grid_vals, 1)) ** 0.5
        avg_mae = avg_abs_sum / max(n_avg_vals, 1)
        avg_rmse = (avg_sq_sum / max(n_avg_vals, 1)) ** 0.5
        consistency_mae = consistency_sum / max(n_avg_vals, 1)
        gt_gap_mae = gt_gap_sum / max(n_avg_vals, 1)

        _log("---- Metrics (normalized [0,1]) ----", fp)
        _log(f"grid_mae: {grid_mae:.8f}", fp)
        _log(f"grid_rmse: {grid_rmse:.8f}", fp)
        _log(f"avg_mae: {avg_mae:.8f}", fp)
        _log(f"avg_rmse: {avg_rmse:.8f}", fp)
        _log(f"consistency_mae(|mean(grid_pred)-avg_pred|): {consistency_mae:.8f}", fp)
        _log(f"data_gap_mae(|mean(grid_gt)-avg_gt|): {gt_gap_mae:.8f}", fp)
        _log(
            f"grid_range pred[{pred_grid_min:.6f},{pred_grid_max:.6f}] gt[{gt_grid_min:.6f},{gt_grid_max:.6f}]",
            fp,
        )
        _log("===============================", fp)


if __name__ == "__main__":
    main()
