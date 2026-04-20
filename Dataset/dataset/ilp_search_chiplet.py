from __future__ import annotations

from typing import Dict, Tuple, List, Optional
from pathlib import Path
from copy import deepcopy
import math
import os
import random

import gurobipy as gp
from gurobipy import GRB

from tool import (
    load_emib_placement_json,
    EMIBNode,
    generate_placement_json_with_EMIB,
)
from visualize_layout import draw_from_solution
from ilp_method_chiplet import (
    ILPModelContext,
    ILPPlacementResult,
    add_absolute_value_constraint_big_m,
    log_objective_breakdown,
    solve_placement_ilp_from_model,
)
from ilp_method_chiplet import build_placement_ilp_model


def _resolve_output_base(project_root: Path) -> Path:
    base_env = os.getenv("EMIB_OUTPUT_BASE")
    if base_env:
        base_path = Path(base_env)
        if not base_path.is_absolute():
            base_path = project_root / base_path
        return base_path
    return project_root / "output_gurobi_EMIB_chiplet"


def build_emib_node_dict(
    edge_map: Dict[Tuple[str, str], dict],
    name_to_idx: Dict[str, int],
) -> Dict[Tuple[int, int], EMIBNode]:
    """
    从 JSON 互联信息构建 EMIBNode 字典。
    key: (i, j) chiplet 标号对，i < j
    value: EMIBNode，信息来自 JSON，width=2*EMIB_bump_width，height=EMIB_length
    """
    emib_node_dict: Dict[Tuple[int, int], EMIBNode] = {}
    for (a, b), edge in edge_map.items():
        if a not in name_to_idx or b not in name_to_idx:
            continue
        i, j = name_to_idx[a], name_to_idx[b]
        if i == j:
            continue
        if i > j:
            i, j = j, i
        bump_width = float(edge.get("EMIB_bump_width", 0) or 0)
        emib_length = float(edge.get("EMIB_length", 0) or 0)
        emib_node_dict[(i, j)] = EMIBNode(
            node1=edge.get("node1", a),
            node2=edge.get("node2", b),
            wireCount=int(edge.get("wireCount", 0) or 0),
            EMIBType=str(edge.get("EMIBType", "") or ""),
            EMIB_length=emib_length,
            EMIB_bump_width=bump_width,
            EMIB_max_width=float(edge.get("EMIB_max_width", 0) or 0),
            width=2.0 * bump_width,
            height=emib_length,
        )
    return emib_node_dict


def _get_var_value(model: gp.Model, var_name: str) -> Optional[float]:
    v = model.getVarByName(var_name)
    if v is None:
        return None
    try:
        return float(v.X)
    except Exception:
        return None


def _compute_objective_terms_from_model(
    ctx: ILPModelContext,
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    计算并返回目标函数中的关键项数值（与 ilp_method_EMIB_chiplet.py 的建模变量一致）：
    - wirelength: interfaceC 用 dx_abs_{i}_{j}+dy_abs_{i}_{j}，EMIB 用 dx_abs_i+dy_abs_i+dx_abs_j+dy_abs_j
    - t: bbox_area_proxy_t
    - aspect_ratio_penalty: aspect_ratio_penalty 变量的值（若未启用则为 None）
    """
    model = ctx.model
    wirelength_val = 0.0

    # interfaceC：chiplet 到 chiplet
    all_connected = getattr(ctx, "all_connected_pairs", {}) or {}
    for (i, j), edge in all_connected.items():
        if getattr(edge, "EMIBType", "") != "interfaceC":
            continue
        dx = _get_var_value(model, f"dx_abs_{i}_{j}") or _get_var_value(model, f"dx_abs_{j}_{i}")
        dy = _get_var_value(model, f"dy_abs_{i}_{j}") or _get_var_value(model, f"dy_abs_{j}_{i}")
        if dx is not None and dy is not None:
            wirelength_val += getattr(edge, "wireCount", 1) * (dx + dy)

    # EMIB（interfaceB/interfaceA）：chiplet→EMIB→chiplet
    emib_connected = getattr(ctx, "EMIB_connected_pairs", {}) or {}
    for (i, j), edge in emib_connected.items():
        dx_i = _get_var_value(model, f"dx_abs_i_{i}_{j}")
        dy_i = _get_var_value(model, f"dy_abs_i_{i}_{j}")
        dx_j = _get_var_value(model, f"dx_abs_j_{i}_{j}")
        dy_j = _get_var_value(model, f"dy_abs_j_{i}_{j}")
        if dx_i is not None and dy_i is not None and dx_j is not None and dy_j is not None:
            wirelength_val += getattr(edge, "wireCount", 1) * (dx_i + dy_i + dx_j + dy_j)

    t_val = _get_var_value(model, "bbox_area_proxy_t")
    aspect_val = _get_var_value(model, "aspect_ratio_penalty")
    return wirelength_val, t_val, aspect_val


# 可配置：边界框放宽因子（第二次搜索时 W、H 乘以此因子）
DEFAULT_BBOX_RELAX_FACTOR = 1.5


def _compute_initial_bbox(nodes: List) -> Tuple[float, float]:
    """计算初始芯片边界框 W、H（与 ilp_method_EMIB_chiplet.py 中估计逻辑一致）"""
    total_area = 0.0
    for node in nodes:
        w = float(node.dimensions.get("x", 0.0))
        h = float(node.dimensions.get("y", 0.0))
        total_area += w * h
    estimated_side = math.ceil(math.sqrt(total_area * 2))
    W = estimated_side * 3
    H = estimated_side * 3
    return W, H


def _run_three_phase_solve(
    ctx: ILPModelContext,
    nodes: List,
) -> ILPPlacementResult:
    """三阶段求解（MIPGap 逐步放宽），返回结果"""
    result = _solve_once_with_gap(ctx=ctx, nodes=nodes, gap=0.0, time_limit=300)
    #result = _solve_once_with_gap(ctx=ctx, nodes=nodes, gap=0.8, time_limit=3600)
    if result.status == "NoSolution":
        print(f"[EMIB] 第一阶段无可行解，切换到第二阶段 MIPGap=0.3 继续尝试。")
        result = _solve_once_with_gap(ctx=ctx, nodes=nodes, gap=0.3, time_limit=300)
    if result.status == "NoSolution":
        print(f"[EMIB] 第二阶段无可行解，切换到第三阶段 MIPGap=0.8 继续尝试。")
        result = _solve_once_with_gap(ctx=ctx, nodes=nodes, gap=0.8, time_limit=3600)
    return result


def _run_random_gap_solve(
    ctx: ILPModelContext,
    nodes: List,
    *,
    time_limit: int = 300,
) -> ILPPlacementResult:
    """
    Single-phase solve with a random MIPGap in [0.0, 1.0].

    This is used to replace the original _run_three_phase_solve behavior when generating
    diverse feasible placements quickly.
    """
    gap = random.uniform(0.0, 1.0)
    result = _solve_once_with_gap(ctx=ctx, nodes=nodes, gap=gap, time_limit=time_limit)
    # Attach the gap used for this solve (kept dynamic to avoid broad dataclass changes).
    try:
        setattr(result, "mip_gap", float(gap))
    except Exception:
        pass
    return result


# 单次求解（固定 60s），允许返回可行解（非最优）
def _solve_once_with_gap(
    *,
    ctx: ILPModelContext,
    nodes: List,
    gap: float,
    time_limit: int = 60,
    mip_focus: int = 3,
    heuristics: float = 0.5,
) -> ILPPlacementResult:
    """
    单次求解（固定 time_limit 秒），允许返回可行解（非最优）。
    - 有可行解：返回 status="Optimal" 或 "Feasible"，objective_value 为 ObjVal
    - 无可行解：返回 status="NoSolution"，objective_value 为 inf
    """
    import time as _time

    model = ctx.model
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = gap
    model.Params.MIPFocus = mip_focus
    model.Params.Heuristics = heuristics
    model.Params.LogToConsole = True

    start = _time.time()
    model.optimize()
    solve_time = _time.time() - start

    status = model.Status
    sol_count = int(getattr(model, "SolCount", 0))

    # 有可行解（包括最优/非最优/超时有解）
    if sol_count > 0 and status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        layout: Dict[str, Tuple[float, float]] = {}
        rotations: Dict[str, bool] = {}
        cx_grid_val: Dict[str, float] = {}
        cy_grid_val: Dict[str, float] = {}
        for k, node in enumerate(nodes):
            x_val = float(ctx.x_grid_var[k].X) if ctx.x_grid_var.get(k) is not None else 0.0
            y_val = float(ctx.y_grid_var[k].X) if ctx.y_grid_var.get(k) is not None else 0.0
            r_val = float(ctx.r[k].X) if ctx.r.get(k) is not None else 0.0
            layout[node.name] = (x_val, y_val)
            rotations[node.name] = bool(r_val > 0.5)
            cx_grid_val[node.name] = float(ctx.cx_grid_var[k].X) if ctx.cx_grid_var.get(k) is not None else 0.0
            cy_grid_val[node.name] = float(ctx.cy_grid_var[k].X) if ctx.cy_grid_var.get(k) is not None else 0.0

        try:
            bw_val = float(ctx.bbox_w.X) if ctx.bbox_w is not None else 0.0
            bh_val = float(ctx.bbox_h.X) if ctx.bbox_h is not None else 0.0
        except Exception:
            bw_val, bh_val = 0.0, 0.0

        status_str = "Optimal" if status == GRB.OPTIMAL else "Feasible"
        obj_val = float(model.ObjVal)
        print(
            f"[EMIB] 单次求解完成: MIPGap={gap}, 状态={status_str}, Obj={obj_val:.6f}, 用时={solve_time:.2f}s, SolCount={sol_count}"
        )
        log_objective_breakdown(ctx, model)

        aspect_ratio_val = 0.0
        try:
            aspect_var = model.getVarByName("aspect_ratio_penalty")
            if aspect_var is not None:
                aspect_ratio_val = float(aspect_var.X)
        except Exception:
            aspect_ratio_val = 0.0

        emib_placements = None
        emib_connected = getattr(ctx, "EMIB_connected_pairs", None)
        if emib_connected and getattr(ctx, "EMIB_x_grid_var", None):
            def _r3(v):
                return round(float(v or 0), 3)
            emib_placements = []
            for (i, j), _ in emib_connected.items():
                ex = float(ctx.EMIB_x_grid_var[(i, j)].X) if (i, j) in ctx.EMIB_x_grid_var else 0.0
                ey = float(ctx.EMIB_y_grid_var[(i, j)].X) if (i, j) in ctx.EMIB_y_grid_var else 0.0
                ew = float(ctx.EMIB_w_var[(i, j)].X) if (i, j) in ctx.EMIB_w_var else 0.0
                eh = float(ctx.EMIB_h_var[(i, j)].X) if (i, j) in ctx.EMIB_h_var else 0.0
                er = bool(ctx.r_EMIB[(i, j)].X > 0.5) if (i, j) in ctx.r_EMIB else False
                na = nodes[i].name if i < len(nodes) else str(i)
                nb = nodes[j].name if j < len(nodes) else str(j)
                emib_placements.append({
                    "node1": na, "node2": nb,
                    "EMIB-x-position": _r3(ex), "EMIB-y-position": _r3(ey),
                    "EMIB_width": _r3(ew), "EMIB_length": _r3(eh),
                    "EMIB-rotation": 1 if er else 0,
                })

        res = ILPPlacementResult(
            layout=layout,
            rotations=rotations,
            objective_value=obj_val,
            status=status_str,
            solve_time=solve_time,
            bounding_box=(bw_val, bh_val),
            cx_grid_var=cx_grid_val,
            cy_grid_var=cy_grid_val,
            emib_placements=emib_placements,
            aspect_ratio_penalty=aspect_ratio_val,
        )
        try:
            setattr(res, "mip_gap", float(gap))
        except Exception:
            pass
        return res

    # 无可行解
    print(f"[EMIB] 单次求解无可行解: MIPGap={gap}, 状态码={status}, 用时={solve_time:.2f}s, SolCount={sol_count}")
    empty_layout = {node.name: (0.0, 0.0) for node in nodes}
    empty_rot = {node.name: False for node in nodes}
    res = ILPPlacementResult(
        layout=empty_layout,
        rotations=empty_rot,
        objective_value=float("inf"),
        status="NoSolution",
        solve_time=solve_time,
        bounding_box=(0.0, 0.0),
        cx_grid_var={node.name: 0.0 for node in nodes},
        cy_grid_var={node.name: 0.0 for node in nodes},
        emib_placements=None,
        aspect_ratio_penalty=None,
    )
    try:
        setattr(res, "mip_gap", float(gap))
    except Exception:
        pass
    return res


def search_multiple_solutions(
    num_solutions: int = 3,
    min_shared_length: float = 0.5,
    input_json_path: Optional[str] = None,
    nodes: Optional[List] = None,
    edges: Optional[List[Tuple[int, int]]] = None,
    fixed_chiplet_idx: Optional[int] = None,
    min_pair_dist_diff: Optional[float] = None,  # chiplet对之间距离差异的最小阈值；此参数控制距离排除约束：至少有一对chiplet的距离差必须 >= min_pair_dist_diff
    time_limit: int = 600,  # 求解时间限制（秒），默认10分钟
    output_dir: Optional[str] = None,  # 输出目录，用于保存.lp文件；如果为None，则使用默认路径
    image_output_dir: Optional[str] = None,  # 图片输出目录，用于保存图片；如果为None则使用output_dir
    placement_output_path: Optional[str] = None,  # placement JSON 完整输出路径；如果为None则使用 output_gurobi_EMIB_chiplet/placement/{输入json名}.json
    bbox_relax_factor: float = DEFAULT_BBOX_RELAX_FACTOR,  # 第一次搜索失败后，放宽边界框时的乘数因子（W、H 各乘以此值）
) -> List[ILPPlacementResult]:
    """
    搜索多个不同的解。
    流程：1) 用预设边界框执行 ILP 求解；2) 若无合法解，按 bbox_relax_factor 放宽边界框后重试；
    3) 若仍无合法解，输出「搜索解失败，无合法解」并终止。
    
    参数:
        num_solutions: 需要搜索的解的数量
        min_shared_length: 相邻chiplet之间的最小共享边长
        input_json_path: 可选，从JSON文件加载输入
        nodes: 可选，chiplet节点列表（如果提供input_json_path则忽略此参数）
        edges: 可选，连接关系列表（如果提供input_json_path则忽略此参数）
        fixed_chiplet_idx: 固定位置的chiplet索引
        min_pair_dist_diff: chiplet对之间距离差异的最小阈值
        output_dir: 输出目录，用于保存.lp文件
        image_output_dir: 图片输出目录
        placement_output_path: placement JSON 输出路径
        bbox_relax_factor: 第一次搜索失败后，放宽边界框时的乘数因子（W、H 各乘以此值），可配置
    """
    if input_json_path is None:
        raise ValueError("EMIB 搜索需要提供 input_json_path 以加载 JSON 文件")

    nodes, edges, edge_map, name_to_idx = load_emib_placement_json(input_json_path)

    
    # edges 预处理，找到nodes中的chiplet的width和height的最小值，将edges中的EMIB_max_width更新为最小值
    # 防止出现EMIB_max_width大于chiplet的width或height的情况，这是不合理的
    # min_width = float("inf")
    # min_height = float("inf")
    # for i, node in enumerate(nodes):
    #     min_width = min(float(node.dimensions.get("x", 0.0)), min_width)
    #     min_height = min(float(node.dimensions.get("y", 0.0)), min_height)
    # # edges 为键值对列表（字典），可原地修改
    # for edge in edges:
    #     edge["EMIB_max_width"] = min(edge["EMIB_max_width"], min_width, min_height)
    # print("预处理完成，EMIB_max_width更新为最小值:", min_width, min_height)

    # 此处为紧凑求解
    # 将edge["EMIB_max_width"]设置为0.0，不考虑芯片间距限制
    # for edge in edges:
    #     edge["EMIB_max_width"] = 0.0
    # # 输出硅桥互联信息（edge_map 中每条连接）
    # print("[EMIB] 硅桥互联信息：")
    # for (a, b), e in sorted(edge_map.items()):
    #     print(f"  ({a}, {b}): node1={e.get('node1')}, node2={e.get('node2')}, "
    #           f"wireCount={e.get('wireCount')}, EMIBType={e.get('EMIBType')}, "
    #           f"EMIB_length={e.get('EMIB_length')}, EMIB_max_width={e.get('EMIB_max_width')}, "
    #           f"EMIB_bump_width={e.get('EMIB_bump_width')}, EMIB_width={e.get('EMIB_width')}")

    solutions = []
    
    # 确定 min_pair_dist_diff 的值：如果为None，则使用默认值1.0
    if min_pair_dist_diff is None:
        min_pair_dist_diff = 1.0

    # edges 已是 6 元组：(node1, node2, wireCount, EMIBType, EMIB_length, EMIB_max_width)
    # print(f"\n[EMIB] 求解：{len(edges)} 条连接")

    # 1. 从 JSON 互联信息构建 EMIBNode 字典：key=(i,j) chiplet 标号，value=EMIBNode
    emib_node_dict = build_emib_node_dict(edge_map, name_to_idx)
    # print(f"[EMIB] EMIBNode 字典：{len(emib_node_dict)} 条互联")

    # 2. 计算初始边界框尺寸
    W_initial, H_initial = _compute_initial_bbox(nodes)
    # print(f"[EMIB] 初始边界框 W={W_initial:.2f}, H={H_initial:.2f}")

    # 3. 第一次 ILP 求解（传入 EMIBNode 字典）
    ctx = build_placement_ilp_model(
        nodes=nodes,
        emib_nodes=emib_node_dict,
        W=W_initial,
        H=H_initial,
        fixed_chiplet_idx=fixed_chiplet_idx,
        min_shared_length=min_shared_length,
    )
    project_root = Path(__file__).parent.parent
    output_base = _resolve_output_base(project_root)
    if output_dir is None:
        output_dir_path = output_base
    else:
        output_dir_path = Path(output_dir)
        if not output_dir_path.is_absolute():
            output_dir_path = project_root / output_dir_path
    output_dir_path.mkdir(parents=True, exist_ok=True)
    lp_file = output_dir_path / "constraints_gurobi.lp"
    ctx.model.write(str(lp_file))

    # Replace the original three-phase solve with a random-gap single solve.
    result = _run_random_gap_solve(ctx=ctx, nodes=nodes, time_limit=300)

    # 4. 若第二次仍失败，输出提示并终止
    if result.status == "NoSolution":
        print("[EMIB] 搜索解失败，无合法解")
        return solutions

    if result.status in ("Optimal", "Feasible"):
        print(f"[EMIB] 求解成功（{result.status}）")
        has_emib_vars = getattr(ctx, "EMIB_x_grid_var", None) is not None

        # 1. 生成 placement JSON（从 ILP 的result中提取硅桥位置）
        if has_emib_vars:
            try:
                if placement_output_path is not None:
                    placement_path = Path(placement_output_path)
                    if not placement_path.is_absolute():
                        placement_path = project_root / placement_path
                    placement_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    input_stem = Path(input_json_path).stem
                    # 固定写入 gcn_thermal/dataset/output/placement
                    placement_dir = project_root / "dataset" / "output" / "placement"
                    placement_dir.mkdir(parents=True, exist_ok=True)
                    placement_path = placement_dir / f"{input_stem}.json"
                generate_placement_json_with_EMIB(
                    result=result,
                    post=None,
                    nodes=nodes,
                    edge_map=edge_map,
                    output_path=str(placement_path),
                    ctx=ctx,
                )
                print(f"[EMIB] placement JSON 已保存: {placement_path}")
            except Exception as e:
                print(f"[EMIB] 警告：生成 placement JSON 失败: {e}")
                import traceback
                traceback.print_exc()

        # 2. 保存布局图片（从 ctx 提取硅桥位置绘图）
        if has_emib_vars:
            try:
                if image_output_dir is not None:
                    image_output_dir_path = Path(image_output_dir)
                    if not image_output_dir_path.is_absolute():
                        image_output_dir_path = project_root / image_output_dir_path
                else:
                    # 固定写入 gcn_thermal/dataset/output/fig
                    image_output_dir_path = project_root / "dataset" / "output" / "fig"

                image_output_dir_path.mkdir(parents=True, exist_ok=True)
                input_stem = Path(input_json_path).stem
                image_path = image_output_dir_path / f"{input_stem}.png"

                draw_from_solution(
                    result=result,
                    post=None,
                    nodes=nodes,
                    edge_map=edge_map,
                    save_path=str(image_path),
                    show=False,
                    ctx=ctx,
                )
                print(f"[EMIB] 布局图片已保存: {image_path}")
            except Exception as e:
                print(f"[EMIB] 警告：保存布局图片失败: {e}")
                import traceback
                traceback.print_exc()

        solutions.append(result)

    return solutions
