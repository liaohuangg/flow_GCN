import os
import sys
from typing import Tuple

# Allow running `python train.py` from within thermalmodel/
# by adding project root to sys.path.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from thermalmodel.dataLoader import ThermalDataset
from thermalmodel.q_model import ThermalUNetMultiTask


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_loss(
    pred_grid: torch.Tensor,
    true_grid: torch.Tensor,
    pred_avg: torch.Tensor,
    true_avg: torch.Tensor,
    alpha: float = 0.5,
    physics_beta: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    # losses on normalized [0,1]
    grid_loss = nn.L1Loss()(pred_grid, true_grid)
    avg_loss = nn.MSELoss()(pred_avg, true_avg)

    loss = grid_loss + alpha * avg_loss

    phys = None
    if physics_beta and physics_beta > 0:
        # mean over H,W for each sample; shape (B,1)
        grid_mean = pred_grid.mean(dim=(2, 3))
        phys = torch.abs(grid_mean - pred_avg).mean()
        loss = loss + physics_beta * phys

    metrics = {
        "loss": float(loss.detach().cpu()),
        "grid_l1": float(grid_loss.detach().cpu()),
        "avg_mse": float(avg_loss.detach().cpu()),
    }
    if phys is not None:
        metrics["physics"] = float(phys.detach().cpu())
    return loss, metrics


def main():
    torch.manual_seed(0)

    device = _device()

    dataset = ThermalDataset(
        thermal_map_rel="Dataset/dataset/output/thermal/thermal_map",
        hotspot_cfg_rel="Dataset/dataset/output/thermal/hotspot_config",
        grid_size=64,
    )

    n_total = len(dataset)
    n_train = int(n_total * 0.9)
    n_val = n_total - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    batch_size = 8
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)

    print("==== Thermal Training Config ====")
    print(f"device: {device} (cuda_available={torch.cuda.is_available()})")
    if torch.cuda.is_available():
        try:
            print(f"cuda_device: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    print(f"dataset_total: {n_total} | train: {n_train} | val: {n_val}")
    print(f"batch_size: {batch_size} | train_batches: {num_train_batches} | val_batches: {num_val_batches}")
    print("===============================")

    model = ThermalUNetMultiTask(in_channels=3, base=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    alpha = 0.5
    physics_beta = 0.0  # set >0 to enable avg≈mean(temp_grid) regularization

    out_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    print_every = 10

    for epoch in range(1, 201):
        model.train()
        train_m = {"loss": 0.0, "grid_l1": 0.0, "avg_mse": 0.0, "physics": 0.0}
        steps = 0

        for batch_idx, batch in enumerate(train_loader):
            power = batch["power"].to(device)
            layout = batch["layout"].to(device)
            totalp = batch["total_power"].to(device)
            temp = batch["temp"].to(device)
            avg = batch["avg_temp"].to(device)

            pred_grid, pred_avg = model(power, layout, totalp)
            loss, metrics = compute_loss(
                pred_grid,
                temp,
                pred_avg,
                avg,
                alpha=alpha,
                physics_beta=physics_beta,
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            steps += 1
            for k in train_m:
                if k in metrics:
                    train_m[k] += metrics[k]

            # per-batch logging (compare avg temp and temp grid)
            grid_l1_batch = torch.mean(torch.abs(pred_grid - temp)).item()
            grid_rmse_batch = torch.sqrt(torch.mean((pred_grid - temp) ** 2)).item()
            avg_abs_batch = torch.mean(torch.abs(pred_avg - avg)).item()
            avg_rmse_batch = torch.sqrt(torch.mean((pred_avg - avg) ** 2)).item()

            pred_grid_mean = pred_grid.mean(dim=(2, 3))
            true_grid_mean = temp.mean(dim=(2, 3))
            mean_consistency_batch = torch.mean(torch.abs(pred_grid_mean - pred_avg)).item()
            mean_gt_gap_batch = torch.mean(torch.abs(true_grid_mean - avg)).item()

            pred_grid_min = float(pred_grid.min().item())
            pred_grid_max = float(pred_grid.max().item())
            true_grid_min = float(temp.min().item())
            true_grid_max = float(temp.max().item())

            pred_avg_mean = float(pred_avg.mean().item())
            true_avg_mean = float(avg.mean().item())

            if batch_idx % print_every == 0:
                print(
                    f"[train] ep {epoch:04d} it {batch_idx:04d} "
                    f"loss {metrics['loss']:.6f} gridL1 {grid_l1_batch:.6f} gridRMSE {grid_rmse_batch:.6f} "
                    f"avgAbs {avg_abs_batch:.6f} avgRMSE {avg_rmse_batch:.6f} "
                    f"avg(pred) {pred_avg_mean:.6f} avg(gt) {true_avg_mean:.6f} "
                    f"grid[min,max] pred[{pred_grid_min:.3f},{pred_grid_max:.3f}] gt[{true_grid_min:.3f},{true_grid_max:.3f}] "
                    f"| |mean(grid_pred)-avg_pred| {mean_consistency_batch:.6f} "
                    f"| |mean(grid_gt)-avg_gt| {mean_gt_gap_batch:.6f}"
                )

        for k in train_m:
            train_m[k] /= max(steps, 1)

        model.eval()
        val_m = {"loss": 0.0, "grid_l1": 0.0, "avg_mse": 0.0, "physics": 0.0}
        vsteps = 0
        with torch.no_grad():
            for batch in val_loader:
                power = batch["power"].to(device)
                layout = batch["layout"].to(device)
                totalp = batch["total_power"].to(device)
                temp = batch["temp"].to(device)
                avg = batch["avg_temp"].to(device)

                pred_grid, pred_avg = model(power, layout, totalp)
                loss, metrics = compute_loss(
                    pred_grid,
                    temp,
                    pred_avg,
                    avg,
                    alpha=alpha,
                    physics_beta=physics_beta,
                )

                vsteps += 1
                for k in val_m:
                    if k in metrics:
                        val_m[k] += metrics[k]

        for k in val_m:
            val_m[k] /= max(vsteps, 1)

        print(
            f"Epoch {epoch:04d} "
            f"| train loss {train_m['loss']:.6f} grid {train_m['grid_l1']:.6f} avg {train_m['avg_mse']:.6f} "
            f"| val loss {val_m['loss']:.6f} grid {val_m['grid_l1']:.6f} avg {val_m['avg_mse']:.6f}"
        )

        if epoch % 10 == 0:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "stats": dataset.stats.to_dict(),
            }
            torch.save(ckpt, os.path.join(out_dir, f"thermal_unet_epoch{epoch}.pth"))


if __name__ == "__main__":
    main()
