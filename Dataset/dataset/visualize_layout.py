#!/usr/bin/env python3
"""visualize_layout.py

只做一件事：
- 读入 *已生成的布局* JSON（例如 output/placement/system_10426.json）
- 直接可视化并保存图片

不进行 ILP 求解，不依赖 Gurobi。

输入 JSON 期望格式（最少字段）：
{
  "chiplets": [
    {"name": "A", "x-position": 0.0, "y-position": 0.0, "width": 10.0, "height": 5.0, "power": 12.0},
    ...
  ],
  "connections": [
    {"node1": "A", "node2": "B", "wireCount": 128, ...},
    ...
  ]
}

用法：
  python visualize_layout.py \
    --input /root/placement/flow_GCN/Dataset/dataset/output/placement/system_10426.json \
    --output-dir /root/placement/flow_GCN/Dataset/dataset/output/fig

输出：
  output-dir/<stem>.png  （默认）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def _import_tool_from_same_dir() -> Tuple[object, object]:
    """确保可以从本目录导入 tool.py，并在导入 pyplot 前强制使用 Agg 后端。"""

    import sys

    script_dir = Path(__file__).parent.resolve()
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    # 避免在无 GUI 环境触发 Qt/Wayland 后端问题
    import matplotlib

    matplotlib.use("Agg")

    from tool import ChipletNode, draw_chiplet_diagram  # type: ignore

    return ChipletNode, draw_chiplet_diagram


def visualize_placement_json(
    placement_json_path: str,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    save_format: str = "png",
    show: bool = False,
    title: Optional[str] = None,
) -> str:
    """读取 output/placement 的布局 JSON 并生成图片。

    返回生成图片路径。
    """

    ChipletNode, draw_chiplet_diagram = _import_tool_from_same_dir()

    p = Path(placement_json_path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    chiplets = data.get("chiplets", [])
    if not isinstance(chiplets, list) or not chiplets:
        raise ValueError(f"layout JSON 格式错误：缺少 chiplets 列表。文件: {placement_json_path}")

    nodes: List[object] = []
    layout: Dict[str, Tuple[float, float]] = {}
    labels: Dict[str, str] = {}

    for c in chiplets:
        if not isinstance(c, dict):
            raise ValueError(f"layout JSON 格式错误：chiplets 元素必须是对象。当前: {c}")

        name = c.get("name")
        x = c.get("x-position")
        y = c.get("y-position")
        w = c.get("width")
        h = c.get("height")
        power = float(c.get("power", 0) or 0)

        if name is None or x is None or y is None or w is None or h is None:
            raise ValueError(f"layout JSON chiplet 字段不完整: {c}")

        name_s = str(name)
        nodes.append(
            ChipletNode(
                name=name_s,
                dimensions={"x": float(w), "y": float(h)},
                phys=[],
                power=power,
            )
        )
        layout[name_s] = (float(x), float(y))
        labels[name_s] = f"{name_s}\nP={power:g}"

    # 可选：从 layout JSON 里读取 connections 画箭头（如果为空则只画块）
    edges = []
    conns = data.get("connections", [])
    if isinstance(conns, list):
        for conn in conns:
            if not isinstance(conn, dict):
                continue
            n1, n2 = conn.get("node1"), conn.get("node2")
            if n1 is None or n2 is None:
                continue
            edges.append((str(n1), str(n2)))

    if output_path is None:
        out_dir = Path(output_dir) if output_dir else p.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{p.stem}.{save_format}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # draw_chiplet_diagram 的标题在 tool.py 内部没有单独参数，这里先不强加。
    # 如果你确实需要 title，建议在 tool.draw_chiplet_diagram 里加参数（再单独改）。
    _ = title

    draw_chiplet_diagram(
        nodes=nodes,
        edges=edges,
        save_path=str(out),
        layout=layout,
        labels=labels,
        grid_size=1.0,
        rotations=None,  # placement JSON 里 width/height 已经是最终尺寸，不做二次旋转
    )

    if show:
        import matplotlib.pyplot as plt

        plt.show()

    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read placement JSON and visualize layout")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", "-i", default=None, help="输入单个布局 JSON（output/placement/system_i.json）")
    src.add_argument("--start", type=int, default=None, help="批量绘图起始 id（包含）")

    parser.add_argument("--end", type=int, default=None, help="批量绘图结束 id（包含，与 --start 配合）")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="批量绘图的输入目录（默认: <script_dir>/output/placement）",
    )

    parser.add_argument("--output", "-o", default=None, help="输出图片完整路径（仅单文件模式优先级最高）")
    parser.add_argument("--output-dir", default=None, help="输出目录（自动命名为 <stem>.<format>）")
    parser.add_argument("--format", "-f", default="png", choices=["png", "svg"], help="保存格式")
    parser.add_argument("--show", action="store_true", help="显示图形窗口（批量模式不建议）")
    parser.add_argument("--title", default=None, help="预留参数（当前不生效）")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()

    if args.input is not None:
        out = visualize_placement_json(
            placement_json_path=args.input,
            output_path=args.output,
            output_dir=args.output_dir,
            save_format=args.format,
            show=args.show,
            title=args.title,
        )
        print(out)
        return

    # batch mode
    if args.start is None or args.end is None:
        raise SystemExit("批量绘图需要同时提供 --start 和 --end")
    if args.end < args.start:
        raise SystemExit("--end 必须 >= --start")

    in_dir = Path(args.input_dir).resolve() if args.input_dir else (script_dir / "output" / "placement")
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (script_dir / "output" / "fig")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(int(args.start), int(args.end) + 1):
        inp = in_dir / f"system_{i}.json"
        if not inp.exists():
            print(f"[skip] missing: {inp}")
            continue
        out_path = out_dir / f"system_{i}.{args.format}"
        out = visualize_placement_json(
            placement_json_path=str(inp),
            output_path=str(out_path),
            output_dir=None,
            save_format=args.format,
            show=False,  # batch mode: avoid GUI
            title=args.title,
        )
        print(out)


if __name__ == "__main__":
    main()
