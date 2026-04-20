#!/usr/bin/env python3
"""
EMIB 芯粒布局可视化脚本（无固定芯粒）。

从 Gurobi ILP 求解结果提取数据，绘制：
- 芯粒矩形（灰色，标注 ID 与 power）
- 硅桥中心点（红色圆点）
- 蓝色细线互联（路径：芯粒 16x16 网格点 → 硅桥中心点 → 目标芯粒网格点）

用法:
  python visualize_emib_layout.py --input <json_path> [--output <png|svg>] [--show]
  或作为模块调用 run_visualization(...)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def draw_from_solution(
    result,
    post: Optional[dict],
    nodes: list,
    edge_map: dict,
    save_path: str,
    title: str = "EMIB Chiplet Layout",
    save_format: str = "png",
    show: bool = False,
    figsize: Tuple[float, float] = (10, 8),
    display_grid_size: Optional[int] = 4,
    ctx=None,
) -> dict:
    """
    根据已有求解结果直接生成布局图（不重新求解）。
    post 为 None 时，若提供 ctx 且含 EMIB 变量，则从 ctx 提取硅桥位置。
    
    参数
    ----
    result : ILPPlacementResult
        求解结果
    post : dict | None
        run_emib_post_process 返回值；可为 None
    nodes : list
        芯粒列表
    edge_map : dict
        边映射
    save_path : str
        图片保存路径
    ctx : 可选，ILP 上下文；post 为 None 时从此提取 emib_placements
    
    返回
    ----
    dict
        结构化输出
    """
    from tool import extract_layout_data_for_vis, draw_emib_layout_diagram

    vis_data = extract_layout_data_for_vis(result, post, nodes, edge_map, ctx=ctx)
    return draw_emib_layout_diagram(
        chiplet_layout=vis_data["chiplet_layout"],
        chiplet_dims=vis_data["chiplet_dims"],
        emib_placements=vis_data["emib_placements"],
        emib_connections=vis_data["emib_connections"],
        chiplet_power=vis_data.get("chiplet_power"),
        title=title,
        show_axes=True,
        save_path=save_path,
        save_format=save_format,
        show=show,
        figsize=figsize,
        display_grid_size=display_grid_size,
    )


# 需在 thermal-placement 项目环境下运行（含 Gurobi）
def run_visualization(
    input_json_path: str,
    output_path: Optional[str] = None,
    save_format: str = "png",
    show: bool = False,
    title: str = "EMIB Chiplet Layout",
    display_grid_size: Optional[int] = 4,
) -> dict:
    """
    运行完整可视化流程：加载 JSON → ILP 求解 → 后处理 → 绘图。
    
    参数
    ----
    input_json_path : str
        输入 JSON 路径（含 chiplets, connections）
    output_path : str | None
        输出图片路径，None 则使用默认 output_gurobi_compact/fig/<stem>_emib_vis.png
    save_format : str
        保存格式，如 "png", "svg"
    show : bool
        是否弹出显示窗口
    title : str
        图形标题
    
    返回
    ----
    dict
        结构化输出: emib_coords, wire_start_end, emib_edge_centers
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from tool import (
        load_emib_placement_json,
        run_emib_post_process,
        extract_layout_data_for_vis,
        draw_emib_layout_diagram,
    )
    from ilp_method_compact import build_placement_ilp_model
    from ilp_EMIB_search_compact import _solve_once_with_gap

    nodes, edges, edge_map, name_to_idx = load_emib_placement_json(input_json_path)
    min_w = min(float(n.dimensions.get("x", 0)) for n in nodes)
    min_h = min(float(n.dimensions.get("y", 0)) for n in nodes)
    for e in edges:
        e["EMIB_max_width"] = min(e["EMIB_max_width"], min_w, min_h)

    ctx = build_placement_ilp_model(nodes=nodes, edges=edges, verbose=False)
    result = _solve_once_with_gap(ctx=ctx, nodes=nodes, gap=0, time_limit=120)
    if result.status not in ("Optimal", "Feasible"):
        raise RuntimeError(f"ILP 求解失败: {result.status}")

    post = run_emib_post_process(ctx=ctx, result=result, nodes=nodes, edge_map=edge_map, name_to_idx=name_to_idx)
    vis_data = extract_layout_data_for_vis(result, post, nodes, edge_map)
    out_path = output_path or str(Path(input_json_path).stem + "_emib_vis." + save_format)
    struct = draw_emib_layout_diagram(
        chiplet_layout=vis_data["chiplet_layout"],
        chiplet_dims=vis_data["chiplet_dims"],
        emib_placements=vis_data["emib_placements"],
        emib_connections=vis_data["emib_connections"],
        chiplet_power=vis_data.get("chiplet_power"),
        title=title,
        show_axes=True,
        save_path=out_path,
        save_format=save_format,
        show=show,
        display_grid_size=4,
    )
    return struct


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMIB 布局可视化")
    parser.add_argument("--input", "-i", required=True, help="输入 JSON 路径")
    parser.add_argument("--output", "-o", help="输出图片路径")
    parser.add_argument("--format", "-f", default="png", choices=["png", "svg"], help="保存格式")
    parser.add_argument("--show", action="store_true", help="显示图形窗口")
    parser.add_argument("--title", default="EMIB Chiplet Layout", help="图形标题")
    parser.add_argument(
        "--grid-size", "-g", type=int, default=4,
        help="蓝线网格规模：16 为 16x16=256 条，4 为 4x4=16 条（每块中心），默认 4",
    )
    args = parser.parse_args()
    run_visualization(
        input_json_path=args.input,
        output_path=args.output,
        save_format=args.format,
        show=args.show,
        title=args.title,
        display_grid_size=args.grid_size,
    )
    print("可视化完成")
