#!/usr/bin/env python3
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
    title: str = "Chiplet Layout",
    save_format: str = "png",
    show: bool = False,
    figsize: Tuple[float, float] = (10, 8),
    display_grid_size: Optional[int] = 4,
    ctx=None,
) -> dict:
    """
    根据已有求解结果直接生成布局图（不重新求解）。
    post 为 None 时，若提供 ctx 且含 EMIB 变量，则从 ctx 提取硅桥位置。
    6
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

'''
python visualize_layout.py  --input /root/placement/flow_GCN/Dataset/dataset/output/placement/system_10426.json  --output-dir /root/placement/flow_GCN/Dataset/dataset/output/fig     --format png 

python visualize_layout.py --start 10425 --end 11026 --input-dir /root/placement/flow_GCN/Dataset/dataset/output/placement --output-dir /root/placement/flow_GCN/Dataset/dataset/output/fig  --format png 
'''