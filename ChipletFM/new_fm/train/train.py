from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from new_fm.data.adapter import ChipletLayoutDataset
from new_fm.data.schema import collate_layout_samples
from new_fm.models.factory import build_model
from new_fm.models.flow_model import FlowMatchingModel
from new_fm.utils.config import load_config, save_config
from new_fm.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


def train(config: dict) -> None:
    train_cfg = config["train"]
    data_cfg = config["dataset"]
    set_seed(int(train_cfg.get("seed", 123)))
    device = _resolve_device(train_cfg.get("device", "auto"))
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    _print_header("ChipletFM Flow Matching Training")
    _print_kv("output_dir", str(output_dir))
    _print_kv("dataset", str(data_cfg["path"]))
    _print_kv("device", str(device))

    train_ds = ChipletLayoutDataset(
        data_cfg["path"],
        split="train",
        max_samples=data_cfg.get("train_max_samples"),
        seed=int(data_cfg.get("seed", 123)),
    )
    val_ds = ChipletLayoutDataset(
        data_cfg["path"],
        split="val",
        max_samples=data_cfg.get("val_max_samples"),
        seed=int(data_cfg.get("seed", 123)),
    )
    legacy_single_graph_batch = bool(train_cfg.get("legacy_single_graph_batch", False))
    if legacy_single_graph_batch:
        train_loader = DataLoader(
            train_ds,
            batch_size=1,
            sampler=RandomSampler(train_ds, replacement=True),
            collate_fn=collate_layout_samples,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=True,
            collate_fn=collate_layout_samples,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=1 if legacy_single_graph_batch else int(train_cfg["batch_size"]),
        shuffle=False,
        collate_fn=collate_layout_samples,
    )

    first = train_ds[0]
    model = build_model(
        node_dim=first.node_feat.shape[1],
        edge_dim=first.edge_attr.shape[1],
        config=config["model"],
        device=str(device),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    best_val = float("inf")
    global_step = 0
    history_path = output_dir / "history.jsonl"
    max_steps = train_cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    epochs = int(train_cfg["epochs"])
    if max_steps is not None:
        epochs = max(epochs, math.ceil(max_steps / max(len(train_loader), 1)))

    _print_kv("train_samples", str(len(train_ds)))
    _print_kv("val_samples", str(len(val_ds)))
    _print_kv("batch_size", str(train_cfg["batch_size"]))
    _print_kv("epochs", str(epochs))
    _print_kv("max_steps", str(max_steps if max_steps is not None else "none"))
    _print_kv("model", str(config["model"].get("name", "native")))
    _print_rule()

    stop_training = False
    started_at = time.time()
    last_log_at = started_at
    for epoch in range(epochs):
        model.train()
        train_values = []
        epoch_started_at = time.time()
        for batch in train_loader:
            batch = batch.to(device)
            if legacy_single_graph_batch:
                batch = batch.repeat_single(int(train_cfg["batch_size"]))
            loss = model.loss(batch)
            train_values.append(float(loss.item()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % int(train_cfg.get("log_every", 20)) == 0:
                lr = optimizer.param_groups[0]["lr"]
                now = time.time()
                _print_step(
                    step=global_step,
                    max_steps=max_steps,
                    epoch=epoch,
                    train_loss=float(loss.item()),
                    val_loss=None,
                    lr=lr,
                    elapsed=now - started_at,
                    interval=now - last_log_at,
                )
                last_log_at = now
            if max_steps is not None and global_step >= max_steps:
                stop_training = True
                break

        val_loss = evaluate_loss(
            model,
            val_loader,
            device,
            repeat_single=legacy_single_graph_batch,
            repeats=int(train_cfg["batch_size"]),
        )
        train_loss = sum(train_values) / max(len(train_values), 1)
        lr = optimizer.param_groups[0]["lr"]
        _print_epoch(
            step=global_step,
            max_steps=max_steps,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=lr,
            epoch_time=time.time() - epoch_started_at,
        )
        with history_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "lr": lr,
                    }
                )
                + "\n"
            )
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "step": global_step,
            "epoch": epoch,
            "val_loss": val_loss,
            "node_dim": first.node_feat.shape[1],
            "edge_dim": first.edge_attr.shape[1],
        }
        torch.save(state, output_dir / "latest.ckpt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, output_dir / "best.ckpt")
            _print_success(f"new best checkpoint saved: val_loss={best_val:.6f}")
        if stop_training:
            break
    _print_rule()
    _print_success(f"training finished at step {global_step}; best_val={best_val:.6f}")


@torch.no_grad()
def evaluate_loss(
    model: FlowMatchingModel,
    loader: DataLoader,
    device: torch.device,
    repeat_single: bool = False,
    repeats: int = 1,
) -> float:
    model.eval()
    values = []
    for batch in loader:
        batch = batch.to(device)
        if repeat_single:
            if len(batch.sample_id) != 1:
                raise ValueError("repeat_single validation requires val batch_size=1")
            batch = batch.repeat_single(repeats)
        values.append(float(model.loss(batch).item()))
    return sum(values) / max(len(values), 1)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


GREEN = "\033[92m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _color(text: str, color: str = GREEN) -> str:
    return f"{color}{text}{RESET}"


def _print_header(title: str) -> None:
    print(_color("=" * 72, BOLD + GREEN))
    print(_color(f" {title}", BOLD + GREEN))
    print(_color("=" * 72, BOLD + GREEN))


def _print_rule() -> None:
    print(_color("-" * 72, DIM + GREEN))


def _print_kv(key: str, value: str) -> None:
    print(f"{_color(key.rjust(14), GREEN)} : {value}")


def _print_success(message: str) -> None:
    print(_color(f"[ok] {message}", BOLD + GREEN))


def _print_step(
    step: int,
    max_steps: int | None,
    epoch: int,
    train_loss: float,
    val_loss: float | None,
    lr: float,
    elapsed: float,
    interval: float,
) -> None:
    total = str(max_steps) if max_steps is not None else "?"
    val_part = "" if val_loss is None else f" val={val_loss:.6f}"
    print(
        _color("[train]", BOLD + GREEN)
        + f" step={step}/{total} epoch={epoch}"
        + f" loss={train_loss:.6f}{val_part}"
        + f" lr={lr:.3g}"
        + f" elapsed={_fmt_time(elapsed)}"
        + f" interval={interval:.1f}s"
    )


def _print_epoch(
    step: int,
    max_steps: int | None,
    epoch: int,
    train_loss: float,
    val_loss: float,
    lr: float,
    epoch_time: float,
) -> None:
    total = str(max_steps) if max_steps is not None else "?"
    print(
        _color("[epoch]", BOLD + GREEN)
        + f" step={step}/{total} epoch={epoch}"
        + f" train={train_loss:.6f} val={val_loss:.6f}"
        + f" lr={lr:.3g} time={_fmt_time(epoch_time)}"
    )


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


if __name__ == "__main__":
    main()
