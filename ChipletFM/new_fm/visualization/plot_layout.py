from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from new_fm.data.adapter import ChipletLayoutDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-pt", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--img-size", type=int, default=1024)
    args = parser.parse_args()

    img_size = (args.img_size, args.img_size)
    if args.sample_pt:
        plot_saved_sample(args.sample_pt, args.output, img_size=img_size)
    else:
        if not args.dataset:
            raise ValueError("--dataset is required when --sample-pt is not provided")
        plot_dataset_sample(args.dataset, args.split, args.index, args.output, img_size=img_size)


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q

    return int(r * 255), int(g * 255), int(b * 255)


def visualize_placement(
    x: torch.Tensor,
    size: torch.Tensor,
    is_macros: torch.Tensor | None = None,
    is_ports: torch.Tensor | None = None,
    plot_pins: bool = False,
    plot_edges: bool = False,
    edge_index: torch.Tensor | None = None,
    edge_attr: torch.Tensor | None = None,
    img_size: tuple[int, int] = (1024, 1024),
    mask: torch.Tensor | None = None,
) -> np.ndarray:
    """PIL rectangle renderer adapted from the legacy visualizer.

    Inputs use this project's [0, 1] normalized centers and sizes. They are
    mapped to the legacy visualizer's [-1, 1] canvas before drawing.
    """
    width, height = img_size
    base_image = Image.new("RGBA", (width, height), "white")

    x = x[:, :2].detach().cpu().float()
    size = size[:, :2].detach().cpu().float()
    x_old = x * 2.0 - 1.0
    size_old = size * 2.0
    if is_macros is not None:
        is_macros = is_macros.detach().cpu().bool()
    if is_ports is not None:
        is_ports = is_ports.detach().cpu().bool()
    mask = is_ports if is_ports is not None and mask is None else mask
    if mask is not None:
        mask = mask.detach().cpu().bool()

    def canvas_to_pixel_coord(coord: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(coord)
        output[:, 0] = (0.5 + coord[:, 0] / 2.0) * width
        output[:, 1] = (0.5 - coord[:, 1] / 2.0) * height
        return output

    node_count = x_old.shape[0]
    h_step = 0.2 / max(node_count, 1) if is_macros is not None else 1.0 / max(node_count, 1)
    h_offsets = {"macro": 0.0, "port": 0.35, "sc": 0.55}

    left_bottom = x_old - size_old / 2.0
    right_top = x_old + size_old / 2.0
    inbounds = torch.logical_and(left_bottom >= -1, right_top <= 1)
    inbounds = torch.logical_and(inbounds[:, 0], inbounds[:, 1])

    left_bottom_px = canvas_to_pixel_coord(left_bottom)
    right_top_px = canvas_to_pixel_coord(right_top)

    for i in range(node_count):
        image = Image.new("RGBA", base_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        if is_macros is not None:
            if is_macros[i]:
                h_offset = h_offsets["macro"]
            elif mask is None or not mask[i]:
                h_offset = h_offsets["sc"]
            else:
                h_offset = h_offsets["port"]
        else:
            h_offset = 0.0
        color = hsv_to_rgb(
            i * h_step + h_offset,
            1 if (mask is None or not mask[i]) else 0.2,
            0.9 if inbounds[i] else 0.5,
        )
        draw.rectangle(
            [
                float(left_bottom_px[i, 0]),
                float(right_top_px[i, 1]),
                float(right_top_px[i, 0]),
                float(left_bottom_px[i, 1]),
            ],
            fill=(*color, 160),
            width=0,
        )
        base_image = Image.alpha_composite(base_image, image)

    draw = ImageDraw.Draw(base_image)
    if (plot_edges or plot_pins) and edge_index is not None and edge_attr is not None:
        edge_index = edge_index.detach().cpu().long()
        edge_attr = edge_attr.detach().cpu().float()
        unique_edges = edge_attr.shape[0] // 2
        if unique_edges > 0 and edge_attr.shape[1] >= 4:
            pin_attr = edge_attr[:unique_edges] * 2.0
            u_pos = pin_attr[:, :2] + x_old[edge_index[0, :unique_edges]]
            v_pos = pin_attr[:, 2:4] + x_old[edge_index[1, :unique_edges]]
            u_pos = canvas_to_pixel_coord(u_pos)
            v_pos = canvas_to_pixel_coord(v_pos)
            if plot_edges:
                for i in range(unique_edges):
                    draw.line(
                        [
                            tuple(u_pos[i].detach().cpu().numpy()),
                            tuple(v_pos[i].detach().cpu().numpy()),
                        ],
                        fill="gray",
                    )
            if plot_pins:
                draw.point([(row[0], row[1]) for row in u_pos.detach().cpu().numpy()], fill="black")
                draw.point([(row[0], row[1]) for row in v_pos.detach().cpu().numpy()], fill="yellow")

    return np.array(base_image)[:, :, :3]


def plot_saved_sample(
    sample_pt: str | Path,
    output: str | Path,
    img_size: tuple[int, int] = (1024, 1024),
) -> None:
    obj = torch.load(sample_pt, map_location="cpu")
    target_img = visualize_placement(obj["target_pos"], obj["size"], img_size=img_size)
    generated_img = visualize_placement(obj["generated_pos"], obj["size"], img_size=img_size)
    save_comparison([("Original", target_img), ("Generated", generated_img)], output)


def plot_saved_sample_from_tensors(
    target_pos: torch.Tensor,
    generated_pos: torch.Tensor,
    size: torch.Tensor,
    output: str | Path,
    img_size: tuple[int, int] = (1024, 1024),
    is_macros: torch.Tensor | None = None,
    is_ports: torch.Tensor | None = None,
) -> None:
    target_img = visualize_placement(
        target_pos, size, is_macros=is_macros, is_ports=is_ports, img_size=img_size
    )
    generated_img = visualize_placement(
        generated_pos, size, is_macros=is_macros, is_ports=is_ports, img_size=img_size
    )
    save_comparison([("Original", target_img), ("Generated", generated_img)], output)


def plot_dataset_sample(
    dataset: str | Path,
    split: str,
    index: int,
    output: str | Path,
    img_size: tuple[int, int] = (1024, 1024),
) -> None:
    sample = ChipletLayoutDataset(dataset, split=split)[index]
    is_ports, is_macros = _flags_from_node_feat(sample.node_feat)
    image = visualize_placement(
        sample.target_pos,
        sample.size,
        is_macros=is_macros,
        is_ports=is_ports,
        edge_index=sample.edge_index,
        edge_attr=sample.edge_attr,
        img_size=img_size,
    )
    save_image(image, output, title=f"{split}[{index}]")


def save_comparison(images: list[tuple[str, np.ndarray]], output: str | Path) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 6))
    if len(images) == 1:
        axes = [axes]
    for ax, (title, image) in zip(axes, images):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output}")


def save_image(image: np.ndarray, output: str | Path, title: str | None = None) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved {output}")


def _flags_from_node_feat(node_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if node_feat.shape[1] >= 5:
        return node_feat[:, 3] > 0.5, node_feat[:, 4] > 0.5
    n = node_feat.shape[0]
    return torch.zeros(n, dtype=torch.bool), torch.ones(n, dtype=torch.bool)


if __name__ == "__main__":
    main()
