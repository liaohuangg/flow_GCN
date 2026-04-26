"""
使用整数线性规划（ILP）进行 chiplet 布局优化（Gurobi版本）。

主要特性
--------
1. **相邻约束**：有连边的 chiplet 必须水平或垂直相邻（紧靠），并且共享边长度不少于给定下界。
2. **旋转约束**：每个 chiplet 允许 0°/90° 旋转，由二进制变量 ``r_k`` 控制宽高交换。
3. **非重叠约束**：任意两块 chiplet 之间不能重叠。
4. **外接方框约束**：显式构造覆盖所有 chiplet 的外接矩形，并对其宽高建立线性约束。
5. **多目标优化**：目标函数为

   ``β1 * wirelength + β2 * t``

   其中 ``wirelength`` 是所有连边中心点间的曼哈顿距离之和，
   ``t`` 是通过 AM–GM 凸近似得到的"面积代理"变量，用来近似外接矩形面积。

实现依赖 Gurobi Optimizer。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import gurobipy as gp
from gurobipy import GRB

try:
    from tool import ChipletNode, draw_chiplet_diagram, EMIBNode, print_emib_node_contents
except ImportError:
    from .tool import ChipletNode, draw_chiplet_diagram, EMIBNode, print_emib_node_contents


def _get_beta_from_env(env_name: str, default: float) -> float:
    """读取环境变量中的 beta 值，若无或格式错误则使用默认值。"""
    value = os.getenv(env_name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"[EMIB] Warning: invalid {env_name}='{value}', fallback to {default}")
        return default


@dataclass
class ILPPlacementResult:
    """ILP求解结果"""

    # 网格坐标（chiplet左下角，grid单位）
    layout: Dict[str, Tuple[float, float]]  # name -> (x_grid, y_grid)
    rotations: Dict[str, bool]  # name -> 是否旋转
    objective_value: float
    status: str
    solve_time: float
    bounding_box: Tuple[float, float]  # (W, H) 边界框尺寸（grid单位）
    # 中心坐标（grid单位；注意：这里保存的是“2×中心坐标”，用于保持整数域）
    cx_grid_var: Dict[str, float]
    cy_grid_var: Dict[str, float]
    # EMIB 硅桥布局（仅 EMIB chiplet 模型有）：每条含 node1, node2, EMIB-x-position, EMIB-y-position, EMIB_width, EMIB_length, EMIB-rotation 等
    emib_placements: Optional[List[dict]] = None
    # 长宽比惩罚（|bbox_w - bbox_h|），用于记录目标函数中的长宽比项
    aspect_ratio_penalty: Optional[float] = None

@dataclass
class ILPModelContext:
    """
    ILP 模型上下文。

    - `model`  : 已经构建好的 Gurobi 模型（包含变量、约束和目标函数，但可以继续加约束）
    - `x, y`  : 每个 chiplet 左下角坐标变量（用于排除解等约束）
    - `r`     : 每个 chiplet 的旋转变量
    - `z1, z2`: 每对有连边的 chiplet 的"相邻方式"变量（水平/垂直）
    - `z1L, z1R, z2D, z2U`: 每对有连边的 chiplet 的相对方向变量（左、右、下、上）
    - `all_connected_pairs` : dict, key=(i,j) 且 i<j, value=edge 字典 {node1, node2, wireCount, EMIBType, EMIB_length, EMIB_max_width}
    - `bbox_w, bbox_h` : 外接方框宽和高对应的变量
    - `W, H`  : 外接边界框的上界尺寸（建模阶段确定）
    - `fixed_chiplet_idx` : 已废弃，不再使用固定芯粒约束（保留此字段以保持接口兼容性）
    """

    model: gp.Model
    nodes: List[ChipletNode]
    edges: List  # 键值对列表 [{"node1", "node2", "wireCount", "EMIBType", "EMIB_length", "EMIB_max_width"}, ...]

    x_grid_var: Dict[int, gp.Var]
    y_grid_var: Dict[int, gp.Var]
    r: Dict[int, gp.Var]
    cx_grid_var: Dict[int, gp.Var]
    cy_grid_var: Dict[int, gp.Var]
    z1: Dict[Tuple[int, int], gp.Var]
    z2: Dict[Tuple[int, int], gp.Var]
    z1L: Dict[Tuple[int, int], gp.Var]
    z1R: Dict[Tuple[int, int], gp.Var]
    z2D: Dict[Tuple[int, int], gp.Var]
    z2U: Dict[Tuple[int, int], gp.Var]
    all_connected_pairs: Dict[Tuple[int, int], dict]

    bbox_w: gp.Var
    bbox_h: gp.Var

    W: float
    H: float
    fixed_chiplet_idx: Optional[int] = None

    # 归一化目标函数相关（用于求解后输出各项结果）
    ref_wirelength: Optional[float] = None
    ref_t: Optional[float] = None
    ref_power: Optional[float] = None
    ref_aspect: Optional[float] = None
    beta_wire: Optional[float] = None
    beta_area: Optional[float] = None
    beta_aspect: Optional[float] = None
    beta_power: Optional[float] = None

    # EMIB Chiplet 布局变量（仅当使用 emib_nodes 构建时存在）
    EMIB_connected_pairs: Optional[Dict[Tuple[int, int], Any]] = None
    EMIB_x_grid_var: Optional[Dict[Tuple[int, int], Any]] = None
    EMIB_y_grid_var: Optional[Dict[Tuple[int, int], Any]] = None
    EMIB_w_var: Optional[Dict[Tuple[int, int], Any]] = None
    EMIB_h_var: Optional[Dict[Tuple[int, int], Any]] = None
    r_EMIB: Optional[Dict[Tuple[int, int], Any]] = None

    # 为了兼容性，添加 prob 属性（指向 model）
    @property
    def prob(self):
        """为了兼容性，返回 model"""
        return self.model

def add_absolute_value_constraint_big_m(
    model: gp.Model,
    abs_var: gp.Var,
    orig_var: gp.Var,
    M: float,
    constraint_prefix: str,
) -> None:
    """
    使用Big-M方法添加绝对值约束：abs_var = |orig_var|
    
    实现方法（参考Big-M方法）：
    1. 创建二进制变量 is_positive，表示 orig_var >= 0
    2. 使用4个约束强制 abs_var = |orig_var|
       - 当 orig_var >= 0 时 (is_positive=1): abs_var = orig_var
       - 当 orig_var < 0 时 (is_positive=0): abs_var = -orig_var
    3. 使用2个约束强制 is_positive 的正确性
    """
    # 创建二进制变量：is_positive = 1 当且仅当 orig_var >= 0
    is_positive = model.addVar(
        name=f"{constraint_prefix}_is_positive",
        vtype=GRB.BINARY
    )
    
    # 约束1: 当 orig_var >= 0 时 (is_positive=1)，约束简化为: abs_var >= orig_var
    model.addConstr(
        abs_var >= orig_var - M * (1 - is_positive),
        name=f"{constraint_prefix}_abs_ge_orig"
    )
    
    # 约束2: 当 orig_var >= 0 时 (is_positive=1)，约束简化为: abs_var <= orig_var
    model.addConstr(
        abs_var <= orig_var + M * (1 - is_positive),
        name=f"{constraint_prefix}_abs_le_orig"
    )
    
    # 约束3: 当 orig_var < 0 时 (is_positive=0)，约束简化为: abs_var >= -orig_var
    model.addConstr(
        abs_var >= -orig_var - M * is_positive,
        name=f"{constraint_prefix}_abs_ge_neg_orig"
    )
    
    # 约束4: 当 orig_var < 0 时 (is_positive=0)，约束简化为: abs_var <= -orig_var
    model.addConstr(
        abs_var <= -orig_var + M * is_positive,
        name=f"{constraint_prefix}_abs_le_neg_orig"
    )
    
    # 约束5: 强制 is_positive = 1 当 orig_var >= 0
    model.addConstr(
        orig_var >= -M * (1 - is_positive),
        name=f"{constraint_prefix}_force_positive"
    )
    
    # 约束6: 强制 is_positive = 0 当 orig_var < 0
    epsilon = 0.001
    model.addConstr(
        orig_var <= M * is_positive,
        name=f"{constraint_prefix}_force_negative"
    )


def select_high_power_indices_by_density(
    n: int,
    nodes: List,
    chiplet_w_orig_grid: Dict[int, float],
    chiplet_h_orig_grid: Dict[int, float],
    top_ratio: float = 0.3,
) -> Tuple[set[int], Optional[float]]:
    """
    按单位面积功耗对芯粒排序，选出前 top_ratio 百分比的芯粒。

    返回:
        (high_power_indices, density_threshold)
        high_power_indices: 单位面积功耗 ≥ 阈值的芯粒下标集合
        density_threshold : 选出的最低单位面积功耗；若无有效芯粒则为 None
    """
    density_list: List[Tuple[int, float]] = []
    for i in range(n):
        p_i = float(getattr(nodes[i], "power", 0.0) or 0.0)
        w_i = float(chiplet_w_orig_grid.get(i, 0.0) or 0.0)
        h_i = float(chiplet_h_orig_grid.get(i, 0.0) or 0.0)
        area_i = w_i * h_i
        if p_i <= 0.0 or area_i <= 0.0:
            continue
        density_list.append((i, p_i / area_i))

    if not density_list:
        return set(), None

    density_list.sort(key=lambda x: x[1], reverse=True)
    k = max(1, int(len(density_list) * top_ratio))
    density_threshold = density_list[k - 1][1]
    high_indices = {idx for idx, dens in density_list if dens >= density_threshold}
    return high_indices, density_threshold


def compute_normalization_factors(
    n: int,
    nodes: List,
    chiplet_w_orig_grid: Dict[int, float],
    chiplet_h_orig_grid: Dict[int, float],
    all_connected_pairs: Dict[Tuple[int, int], dict],
    power_aware_enabled: bool,
) -> Tuple[float, float, float, float]:
    """
    先验估算归一化基准（Static Scaling），用于多目标加权求和的量级对齐。
    修改后：所有返回的参考值都调整到100左右的数量级

    返回
    ----
    (ref_wirelength, ref_t, ref_power, ref_aspect)
        线长、面积代理、功耗项、长宽比偏差的参考值（均为100量级）
    """
    # 1. 总面积与特征长度
    total_area = sum(
        chiplet_w_orig_grid[i] * chiplet_h_orig_grid[i] for i in range(n)
    )
    L_avg = math.sqrt(total_area) if total_area > 0 else 1.0
    print(f"[DEBUG] L_avg: {L_avg}")
    print(f"[DEBUG] total_area: {total_area}")
    # 2.  scaling factors
    # (1) 面积：所有 chiplet 面积和 * 2
    ref_t = L_avg * 2.0

    # (2) 线长：线的根数 * sqrt( chiplet 面积和 )
    total_wire_count = sum(
        e.get("wireCount", 1) if isinstance(e, dict) else getattr(e, "wireCount", 1)
        for e in all_connected_pairs.values()
    )
    ref_wirelength = max(total_wire_count * L_avg / 2.0, 1.0)

    # (3) 长宽比：所有 chiplet 长边之和 - 所有 chiplet 短边之和
    sum_long = 0.0
    sum_short = 0.0
    for i in range(n):
        w = chiplet_w_orig_grid[i]
        h = chiplet_h_orig_grid[i]
        sum_long += max(w, h)
        sum_short += min(w, h)
    ref_aspect = max(sum_long - sum_short, 1.0)

    # (4) 功耗：选取单位面积功耗排名前 30% 的芯粒，计算
    #     ~ L_avg * [ Σ(p_i * p_j) for pairs + Σ(p_i * p_i) for each i ]，但只对高功耗密度芯粒
    ref_power = 1.0
    if power_aware_enabled:
        high_idxs, density_threshold = select_high_power_indices_by_density(
            n, nodes, chiplet_w_orig_grid, chiplet_h_orig_grid, top_ratio=0.3
        )
        if high_idxs:
            pair_sum = 0.0
            self_sum = 0.0
            high_list = sorted(high_idxs)
            for a in range(len(high_list)):
                i = high_list[a]
                p_i = float(getattr(nodes[i], "power", 0.0) or 0.0)
                self_sum += p_i * p_i
                for b in range(a + 1, len(high_list)):
                    j = high_list[b]
                    p_j = float(getattr(nodes[j], "power", 0.0) or 0.0)
                    pair_sum += p_i * p_j

            # 让参考功耗量级在 L_avg 附近，按高功耗密度芯粒数目轻微缩放
            scale = max(len(high_list), 1)
            ref_power = max(L_avg / (n / 4.0) * (pair_sum + self_sum) / scale, 1.0)

    print(f"[DEBUG] ref_wirelength: {ref_wirelength}")
    print(f"[DEBUG] ref_t: {ref_t}")
    print(f"[DEBUG] ref_power: {ref_power}")
    print(f"[DEBUG] ref_aspect: {ref_aspect}")
    return ref_wirelength, ref_t, ref_power, ref_aspect


def log_objective_breakdown(ctx: "ILPModelContext", model: gp.Model) -> None:
    """
    输出归一化目标函数各项的分解结果到标准输出。
    可在 solve_placement_ilp_from_model 或 _solve_once_with_gap 求解成功后调用。
    """
    if getattr(ctx, "ref_wirelength", None) is None or getattr(ctx, "ref_t", None) is None:
        return
    try:
        v_wl = model.getVarByName("wirelength")
        v_t = model.getVarByName("bbox_area_proxy_t")
        v_asp = model.getVarByName("aspect_ratio_penalty")
        v_pwr = model.getVarByName("power_aware_penalty")
        val_wl = float(v_wl.X) if v_wl else 0.0
        val_t = float(v_t.X) if v_t else 0.0
        val_asp = float(v_asp.X) if v_asp else 0.0
        val_pwr = float(v_pwr.X) if v_pwr else 0.0
        norm_wl = val_wl / ctx.ref_wirelength
        norm_t = val_t / ctx.ref_t
        norm_asp = val_asp / (ctx.ref_aspect or 1.0)
        norm_pwr = val_pwr / (ctx.ref_power or 1.0)
        contrib_wl = (ctx.beta_wire or 1.0) * norm_wl
        contrib_t = (ctx.beta_area or 1.0) * norm_t
        contrib_asp = (ctx.beta_aspect or 0.0) * norm_asp
        contrib_pwr = (ctx.beta_power or 0.0) * norm_pwr
        print("\n[未归一化实际值]")
        print(f"  线长(wirelength)={val_wl:.4f}, 面积代理(t)={val_t:.4f}, "
              f"功耗惩罚(power)={val_pwr:.4f}, 高宽比惩罚(aspect)={val_asp:.4f}")
        print("[目标函数各项分解]")
        print(f"  线长: raw={val_wl:.2f}, norm={norm_wl:.4f}, contrib={contrib_wl:.4f}")
        print(f"  面积代理t: raw={val_t:.2f}, norm={norm_t:.4f}, contrib={contrib_t:.4f}")
        print(f"  长宽比偏差: raw={val_asp:.2f}, norm={norm_asp:.4f}, contrib={contrib_asp:.4f}")
        print(f"  功耗惩罚: raw={val_pwr:.2f}, norm={norm_pwr:.4f}, contrib=-{contrib_pwr:.4f}")
    except Exception as e:
        print(f"  [目标分解输出跳过: {e}]")


def solve_placement_ilp_from_model(
    ctx: ILPModelContext,
    time_limit: int = 600,  # 默认10分钟
    verbose: bool = True,
) -> ILPPlacementResult:
    """
    在已有 ILPModelContext 上调用求解器并抽取解。

    可以在多轮求解之间往 ctx.model 上继续添加约束（例如排除解约束）。
    """
    import time

    model = ctx.model
    nodes = ctx.nodes
    x_grid_var, y_grid_var, r = ctx.x_grid_var, ctx.y_grid_var, ctx.r
    cx_grid_var, cy_grid_var = ctx.cx_grid_var, ctx.cy_grid_var
    W, H = ctx.W, ctx.H

    start_time = time.time()

    if verbose:
        print("\n开始求解ILP问题...")
        print(f"变量数量: {model.NumVars}")
        print(f"约束数量: {model.NumConstrs}")

    # 设置 Gurobi 参数
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('LogToConsole', 1 if verbose else 0)

    try:
        model.optimize()
        solve_time = time.time() - start_time

        # 获取求解状态
        status_map = {
            GRB.OPTIMAL: "Optimal",
            GRB.INFEASIBLE: "Infeasible",
            GRB.UNBOUNDED: "Unbounded",
            GRB.TIME_LIMIT: "TimeLimit",
            GRB.INTERRUPTED: "Interrupted",
        }
        status_str = status_map.get(model.status, f"Unknown({model.status})")

        # if verbose:
        print(f"\n求解状态: {status_str}")
        print(f"求解时间: {solve_time:.2f} 秒")
        if model.status == GRB.OPTIMAL or model.status == GRB.FEASIBLE:
            print(f"目标函数值: {model.ObjVal:.2f}")
            log_objective_breakdown(ctx, model)

        # 提取解
        layout: Dict[str, Tuple[float, float]] = {}
        rotations: Dict[str, bool] = {}
        cx_grid_val: Dict[str, float] = {}
        cy_grid_val: Dict[str, float] = {}
        for k, node in enumerate(nodes):
            if model.status == GRB.OPTIMAL:
                x_val = float(x_grid_var[k].X) if x_grid_var[k] is not None else 0.0
                y_val = float(y_grid_var[k].X) if y_grid_var[k] is not None else 0.0
                r_val = float(r[k].X) if r[k] is not None else 0.0
                layout[node.name] = (x_val, y_val)
                rotations[node.name] = bool(r_val > 0.5)

                # 这里的 cx/cy 是“2×中心坐标”的整数值（与建模约束一致）
                cx_grid_val[node.name] = float(cx_grid_var[k].X) if cx_grid_var.get(k) is not None else 0.0
                cy_grid_val[node.name] = float(cy_grid_var[k].X) if cy_grid_var.get(k) is not None else 0.0
            else:
                layout[node.name] = (0.0, 0.0)
                rotations[node.name] = False
                cx_grid_val[node.name] = 0.0
                cy_grid_val[node.name] = 0.0
        obj_value = (
            model.ObjVal if model.status == GRB.OPTIMAL else float("inf")
        )

        # 使用求解得到的 bbox_w / bbox_h 作为返回的边界框尺寸
        try:
            bw_val = ctx.bbox_w.X if ctx.bbox_w is not None else None
            bh_val = ctx.bbox_h.X if ctx.bbox_h is not None else None
        except Exception:
            bw_val, bh_val = None, None

        bbox_tuple = (
            float(bw_val) if bw_val is not None else 0.0,
            float(bh_val) if bh_val is not None else 0.0,
        )

        return ILPPlacementResult(
            layout=layout,
            rotations=rotations,
            objective_value=obj_value,
            status=status_str,
            solve_time=solve_time,
            bounding_box=bbox_tuple,
            cx_grid_var=cx_grid_val,
            cy_grid_var=cy_grid_val,
        )

    except Exception as e:
        solve_time = time.time() - start_time
        if verbose:
            print(f"\n求解出错: {e}")
            import traceback

            traceback.print_exc()

        # 返回空解
        layout = {node.name: (0.0, 0.0) for node in nodes}
        rotations = {node.name: False for node in nodes}
        return ILPPlacementResult(
            layout=layout,
            rotations=rotations,
            objective_value=float("inf"),
            status="Error",
            solve_time=solve_time,
            bounding_box=(W if W else 100.0, H if H else 100.0),
        )


def build_placement_ilp_model(
    nodes: List[ChipletNode],
    edges: Optional[List] = None,  # 键值对列表（当 emib_nodes 未提供时使用）
    emib_nodes: Optional[Dict[Tuple[int, int], "EMIBNode"]] = None,  # key=(i,j) chiplet 标号，value=EMIBNode
    W: Optional[float] = None,
    H: Optional[float] = None,
    time_limit: int = 600,  # 默认10分钟
    verbose: bool = True,
    min_shared_length: float = 0.0,
    minimize_bbox_area: bool = True,
    distance_weight: float = 1.0,
    area_weight: float = 2.0,
    fixed_chiplet_idx: Optional[int] = None,  # 已废弃，不再使用固定芯粒约束
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2,
    power_aware_enabled: bool = True,
) -> ILPModelContext:
    """
    使用连续坐标ILP求解chiplet布局（不再使用 grid_size 网格化）。
    
    与build_placement_ilp_model的主要区别：
    1. 坐标/尺寸使用连续变量（与输入尺寸同单位）
    2. silicon_bridge 连接采用“紧邻/贴边”约束（间隙为0）
    3. shared edge length 直接使用实际单位（不做网格化换算）
    
    参数
    ----
    fixed_chiplet_idx: Optional[int]
        已废弃，不再使用固定芯粒约束（保留此参数以保持接口兼容性）
    其他参数同build_placement_ilp_model
    """
    import math
    
    n = len(nodes)
    name_to_idx = {node.name: i for i, node in enumerate(nodes)}
    
    # ============ 步骤1: 读取已知条件 ============
    # 1.1 读取芯片原始尺寸 chiplet_w_orig chiplet_h_orig 表示chiplet的长宽
    chiplet_w_orig = {}
    chiplet_h_orig = {}
    for i, node in enumerate(nodes):
        chiplet_w_orig[i] = float(node.dimensions.get("x", 0.0))
        chiplet_h_orig[i] = float(node.dimensions.get("y", 0.0))
        print(f"node {i} w: {chiplet_w_orig[i]}, h: {chiplet_h_orig[i]}")
    
    # 不再网格化：直接使用原始尺寸（连续）
    chiplet_w_orig_grid = {i: chiplet_w_orig[i] for i in range(n)}
    chiplet_h_orig_grid = {i: chiplet_h_orig[i] for i in range(n)}
    
    # 1.2 读取连接关系：优先使用 emib_nodes，否则从 edges 解析
    all_connected_pairs: Dict[Tuple[int, int], Any] = {}
    if emib_nodes is not None:
        for (i, j), emib_node in emib_nodes.items():
            if i >= j:
                i, j = j, i
            all_connected_pairs[(i, j)] = emib_node
    elif edges is not None:
        for edge in edges:
            if not isinstance(edge, dict) or not all(k in edge for k in ("node1", "node2", "wireCount", "EMIBType", "EMIB_length", "EMIB_max_width", "EMIB_bump_width")):
                raise ValueError(f"边格式错误：每条边需含 node1, node2, wireCount, EMIBType, EMIB_length, EMIB_max_width, EMIB_bump_width。当前边: {edge}")
            src_name = edge["node1"]
            dst_name = edge["node2"]
            if src_name not in name_to_idx or dst_name not in name_to_idx:
                continue
            i, j = name_to_idx[src_name], name_to_idx[dst_name]
            if i == j:
                continue
            if i > j:
                i, j = j, i
            bump = float(edge.get("EMIB_bump_width", 0) or 0)
            all_connected_pairs[(i, j)] = EMIBNode(
                node1=src_name, node2=dst_name,
                wireCount=int(edge.get("wireCount", 0) or 0),
                EMIBType=str(edge.get("EMIBType", "") or ""),
                EMIB_length=float(edge.get("EMIB_length", 0) or 0),
                EMIB_bump_width=bump,
                EMIB_max_width=float(edge.get("EMIB_max_width", 0) or 0),
                width=2.0 * bump,
                height=float(edge.get("EMIB_length", 0) or 0),
            )
    else:
        raise ValueError("build_placement_ilp_model 需提供 emib_nodes 或 edges")

    # EMIB_connected_pairs: 形式与 all_connected_pairs 相同，但排除 EMIBType 为 interfaceC 的连接
    EMIB_connected_pairs: Dict[Tuple[int, int], Any] = {
        (i, j): e for (i, j), e in all_connected_pairs.items() if e.EMIBType != "interfaceC"
    }

    if verbose:
        print(f"连接统计: 总计 {len(all_connected_pairs)} 对")
        print("连接关系（EMIBNode 内容）:")
        print_emib_node_contents(
            all_connected_pairs,
            key_formatter=lambda k: f"({nodes[k[0]].name},{nodes[k[1]].name})",
        )
    # 1.3 估算芯片边界框尺寸（使用实际尺寸单位）
    if W is None or H is None:
        total_area = sum(chiplet_w_orig_grid[i] * chiplet_h_orig_grid[i] for i in range(n))
        print(f"total_area: {total_area}")
        estimated_side = math.ceil(math.sqrt(total_area * 2))
        print(f"estimated_side: {estimated_side}")
        if W is None:
            W = estimated_side * 3
        if H is None:
            H = estimated_side * 3
        print(f"Estimated W: {W}, H: {H}")
    
    if verbose:
        print(f"连续布局: W={W}, H={H}")
        print(f"问题规模: {n} 个模块, {len(all_connected_pairs)} 对连接")
        print("连接关系 [node1, node2, wireCount, EMIBType, EMIB_length, EMIB_max_width]:")
        for (i, j), e in all_connected_pairs.items():
            print(f"  ({i},{j}): [{e.node1}, {e.node2}, {e.wireCount}, {e.EMIBType}, {e.EMIB_length}, {e.EMIB_max_width}]")
    
    # ============ 步骤2: 创建ILP问题 ============
    model = gp.Model("ChipletPlacementGrid")
    
    # 大M常数
    M = max(W, H) * 2 # 确保 M 足够覆盖任何两个组件之间的距离

    # ============ 步骤3: 定义变量 ============
    # 3.1 二进制变量：旋转变量
    r = {}
    for k in range(n):
        r[k] = model.addVar(name=f"r_{k}", vtype=GRB.BINARY)
    
    r_EMIB = {}
    for (i, j), emib_node in EMIB_connected_pairs.items():
        r_EMIB[(i, j)] = model.addVar(name=f"r_EMIB_{i}_{j}", vtype=GRB.BINARY)

    # 3.2 整数变量：实际宽度和高度
    w_var = {}
    h_var = {}
    for k in range(n):
        w_min = min(chiplet_w_orig_grid[k], chiplet_h_orig_grid[k])
        w_max = max(chiplet_w_orig_grid[k], chiplet_h_orig_grid[k])
        w_var[k] = model.addVar(name=f"w_var_{k}", lb=w_min, ub=w_max, vtype=GRB.CONTINUOUS)
        h_var[k] = model.addVar(name=f"h_var_{k}", lb=w_min, ub=w_max, vtype=GRB.CONTINUOUS)
    
     # 构建 EMIB 芯粒变量
    EMIB_w_var = {}
    EMIB_h_var = {}
    for (i, j), emib_node in EMIB_connected_pairs.items():
        w_min = min(emib_node.width, emib_node.height)
        w_max = max(emib_node.width, emib_node.height)
        EMIB_w_var[(i, j)] = model.addVar(name=f"EMIB_w_var_{i}_{j}", lb=w_min, ub=w_max, vtype=GRB.CONTINUOUS)
        EMIB_h_var[(i, j)] = model.addVar(name=f"EMIB_h_var_{i}_{j}", lb=w_min, ub=w_max, vtype=GRB.CONTINUOUS)
    print(f"EMIB_w_var: {EMIB_w_var}")
    print(f"EMIB_h_var: {EMIB_h_var}")
    print(f"EMIB_connected_pairs: {EMIB_connected_pairs}")
    # 3.3 整数变量：每个chiplet在grid中的左下角坐标（grid索引）
    # 坐标的上下界为0到W - w_var[k]和0到H - h_var[k] 不能超过边界框 - 实际长宽（考虑旋转）
    x_grid_var = {}
    y_grid_var = {}
    for k in range(n):
        x_grid_var[k] = model.addVar(
            name=f"x_grid_var_{k}",
            lb=0,
            ub=W,
            vtype=GRB.CONTINUOUS
        )
        y_grid_var[k] = model.addVar(
            name=f"y_grid_var_{k}",
            lb=0,
            ub=H,
            vtype=GRB.CONTINUOUS
        )

     # 构建 EMIB 芯粒坐标变量
    EMIB_x_grid_var = {}
    EMIB_y_grid_var = {}
    for (i, j), emib_node in EMIB_connected_pairs.items():
        EMIB_x_grid_var[(i, j)] = model.addVar(name=f"EMIB_x_grid_var_{i}_{j}", lb=0, ub=W, vtype=GRB.CONTINUOUS)
        EMIB_y_grid_var[(i, j)] = model.addVar(name=f"EMIB_y_grid_var_{i}_{j}", lb=0, ub=H, vtype=GRB.CONTINUOUS)

    # 构建 EMIB 芯粒中心坐标变量
    EMIB_cx_grid_var = {}
    EMIB_cy_grid_var = {}
    for (i, j), emib_node in EMIB_connected_pairs.items():
        EMIB_cx_grid_var[(i, j)] = model.addVar(name=f"EMIB_cx_grid_var_{i}_{j}", lb=0, ub=2 * W, vtype=GRB.CONTINUOUS)
        EMIB_cy_grid_var[(i, j)] = model.addVar(name=f"EMIB_cy_grid_var_{i}_{j}", lb=0, ub=2 * H, vtype=GRB.CONTINUOUS)

    # 3.4 辅助变量：中心坐标（为保持整数域，这里使用“2×中心坐标”）
    cx_grid_var = {}
    cy_grid_var = {}
    for k in range(n):
        cx_grid_var[k] = model.addVar(name=f"cx_grid_var_{k}", lb=0, ub=2 * W, vtype=GRB.CONTINUOUS)
        cy_grid_var[k] = model.addVar(name=f"cy_grid_var_{k}", lb=0, ub=2 * H, vtype=GRB.CONTINUOUS)
    
    # 3.5 二进制变量：控制相邻方式
    z1 = {}
    z2 = {}
    z1L = {}
    z1R = {}
    z2D = {}
    z2U = {}
    
    # 3.6 二进制变量：控制相邻方式（仅对需要相邻约束的 (i,j)，即 EMIBType != interfaceC）
    for (i, j), edge in all_connected_pairs.items():
        if edge.EMIBType == "interfaceC":  # interfaceC 为基板走线，不需要相邻约束
            continue
        z1[(i, j)] = model.addVar(name=f"z1_{i}_{j}", vtype=GRB.BINARY)
        z2[(i, j)] = model.addVar(name=f"z2_{i}_{j}", vtype=GRB.BINARY)
        z1L[(i, j)] = model.addVar(name=f"z1L_{i}_{j}", vtype=GRB.BINARY)
        z1R[(i, j)] = model.addVar(name=f"z1R_{i}_{j}", vtype=GRB.BINARY)
        z2D[(i, j)] = model.addVar(name=f"z2D_{i}_{j}", vtype=GRB.BINARY)
        z2U[(i, j)] = model.addVar(name=f"z2U_{i}_{j}", vtype=GRB.BINARY)
    
    # 3.7 辅助变量：整个布局的中心点位置
    cx_center = model.addVar(name=f"cx_center", lb=0, ub=W, vtype=GRB.CONTINUOUS)
    cy_center = model.addVar(name=f"cy_center", lb=0, ub=H, vtype=GRB.CONTINUOUS)
    # ============ 步骤4: 定义约束 ============

    # 4.1 旋转约束 & 边界约束
    for k in range(n):
        # 旋转约束  
        model.addConstr(
            w_var[k] == chiplet_w_orig_grid[k] + r[k] * (chiplet_h_orig_grid[k] - chiplet_w_orig_grid[k]),
            name=f"width_rotation_{k}"
        )
        model.addConstr(
            h_var[k] == chiplet_h_orig_grid[k] + r[k] * (chiplet_w_orig_grid[k] - chiplet_h_orig_grid[k]),
            name=f"height_rotation_{k}"
        )
        # 边界约束
        model.addConstr(x_grid_var[k] <= W - w_var[k], name=f"x_grid_var_ub_{k}")
        model.addConstr(y_grid_var[k] <= H - h_var[k], name=f"y_grid_var_ub_{k}")
    
    # 4.2 构建 EMIB 芯粒旋转约束
    for (i, j), emib_node in EMIB_connected_pairs.items():
        # 核心规则：
        # r_EMIB=0 → 不旋转 → w=原始width，h=原始height
        # r_EMIB=1 → 旋转 → w=原始height，h=原始width
        model.addConstr(
            EMIB_w_var[(i, j)] == emib_node.width * (1 - r_EMIB[(i, j)]) + emib_node.height * r_EMIB[(i, j)],
            name=f"EMIB_width_rotation_{i}_{j}"
        )
        model.addConstr(
            EMIB_h_var[(i, j)] == emib_node.height * (1 - r_EMIB[(i, j)]) + emib_node.width * r_EMIB[(i, j)],
            name=f"EMIB_height_rotation_{i}_{j}"
        )
        # 边界约束（保留）
        model.addConstr(EMIB_x_grid_var[(i, j)] <= W - EMIB_w_var[(i, j)], name=f"EMIB_x_grid_var_ub_{i}_{j}")
        model.addConstr(EMIB_y_grid_var[(i, j)] <= H - EMIB_h_var[(i, j)], name=f"EMIB_y_grid_var_ub_{i}_{j}")

    # 4.2 中心坐标定义
    # 约束：cx2 = x + w/2, cy2 = y + h/2
    for k in range(n):
        model.addConstr(cx_grid_var[k] == x_grid_var[k] + w_var[k] / 2, name=f"cx_def_{k}")
        model.addConstr(cy_grid_var[k] == y_grid_var[k] + h_var[k] / 2, name=f"cy_def_{k}")
    
    for (i, j), emib_node in EMIB_connected_pairs.items():
        model.addConstr(EMIB_cx_grid_var[(i, j)] == EMIB_x_grid_var[(i, j)] + EMIB_w_var[(i, j)] / 2, name=f"EMIB_cx_def_{i}_{j}")
        model.addConstr(EMIB_cy_grid_var[(i, j)] == EMIB_y_grid_var[(i, j)] + EMIB_h_var[(i, j)] / 2, name=f"EMIB_cy_def_{i}_{j}")

     # 4.3 相邻约束：遍历 EMIB_connected_pairs（已排除 interfaceC）
    # 此处相邻改用EMIB的重叠来实现
    for (i, j), emib_node in EMIB_connected_pairs.items():
        # 规则1: 必须相邻，且只能选一种方式
        model.addConstr(
            z1[(i, j)] + z2[(i, j)] == 1,
            name=f"must_adjacent_sb_{i}_{j}"
        )
        
        # 规则2: 如果水平相邻，要么 i 在左，要么 i 在右
        model.addConstr(
            z1L[(i, j)] + z1R[(i, j)] == z1[(i, j)],
            name=f"horizontal_direction_sb_{i}_{j}"
        )
        
        # 规则3: 如果垂直相邻，要么 i 在下，要么 i 在上
        model.addConstr(
            z2D[(i, j)] + z2U[(i, j)] == z2[(i, j)],
            name=f"vertical_direction_sb_{i}_{j}"
        )
        # 强制绑定：
        # 水平相邻（z1=1）→ r_EMIB=0（不旋转）
        # 垂直相邻（z2=1）→ r_EMIB=1（旋转）
        model.addConstr(r_EMIB[(i, j)] == z2[(i, j)], name=f"EMIB_rotate_eq_z2_{i}_{j}")

        # 规则4: 水平相邻的具体约束
        # 水平相邻，EMIB Chiplet不进行旋转

        # 如果 i 在左（z1L[i,j] = 1）：布局为 [i][EMIB][j]
        # 约束：EMIB chiplet的左边需要与chiplet i 重叠不少于bump_width 且不与chiplet j重叠
        # 由于后续有chiplet i与chiplet j的非重叠约束，此处不添加
        eps = 0.001
        model.addConstr(
            x_grid_var[j] - (x_grid_var[i] + w_var[i]) >= 0 - M * (1 - z1L[(i, j)]),
            name=f"horizontal_left_dist_lb_{i}_{j}"
        )
        model.addConstr(
            (x_grid_var[i] + w_var[i]) - EMIB_x_grid_var[(i, j)] >= emib_node.EMIB_bump_width - eps - M * (1 - z1L[(i, j)]),
            name=f"EMIB_left_overlap_{i}_{j}"
        )
        model.addConstr(
            (EMIB_x_grid_var[(i, j)] + EMIB_w_var[(i, j)]) - x_grid_var[j] >= emib_node.EMIB_bump_width - eps - M * (1 - z1L[(i, j)]),
            name=f"EMIB_right_overlap_{i}_{j}"
        )
     

        # 如果 i 在右（z1R[i,j] = 1）：布局为 [j][EMIB][i]
        # 约束：EMIB chiplet的左边需要与chiplet j 重叠不少于bump_width 且不与chiplet i重叠
        # 由于后续有chiplet i与chiplet j的非重叠约束，此处不添加
        model.addConstr(
            x_grid_var[i] - (x_grid_var[j] + w_var[j]) >= 0 - M * (1 - z1R[(i, j)]),
            name=f"horizontal_right_dist_lb_{i}_{j}"
        )
        model.addConstr(
            (x_grid_var[j] + w_var[j]) - EMIB_x_grid_var[(i, j)] >= emib_node.EMIB_bump_width - eps - M * (1 - z1R[(i, j)]),
            name=f"EMIB_left_overlap_right_{i}_{j}"
        )
        model.addConstr(
            (EMIB_x_grid_var[(i, j)] + EMIB_w_var[(i, j)]) - x_grid_var[i] >= emib_node.EMIB_bump_width - eps - M * (1 - z1R[(i, j)]),
            name=f"EMIB_right_overlap_right_{i}_{j}"
        )
       

        # 规则5: 垂直相邻的具体约束
        # 垂直相邻，EMIB Chiplet 可旋转（由 r_EMIB 控制）
        # EMIB 芯粒旋转约束, 只有当两个有硅桥互联芯粒上下相邻时，对应的EMIB才进行旋转
        # 如果 i 在下（z2D[i,j] = 1）：布局为 [i][EMIB][j]（从下到上）
        model.addConstr(
            y_grid_var[j] - (y_grid_var[i] + h_var[i]) >= 0 - M * (1 - z2D[(i, j)]),
            name=f"vertical_down_dist_lb_{i}_{j}"
        )
        model.addConstr(
            (y_grid_var[i] + h_var[i]) - EMIB_y_grid_var[(i, j)] >= emib_node.EMIB_bump_width - eps - M * (1 - z2D[(i, j)]),
            name=f"EMIB_bottom_overlap_down_{i}_{j}"
        )
        model.addConstr(
            (EMIB_y_grid_var[(i, j)] + EMIB_h_var[(i, j)]) - y_grid_var[j] >= emib_node.EMIB_bump_width - eps - M * (1 - z2D[(i, j)]),
            name=f"EMIB_top_overlap_down_{i}_{j}"
        )
        

        # 如果 i 在上（z2U[i,j] = 1）：布局为 [j][EMIB][i]（从下到上）
        model.addConstr(
            y_grid_var[i] - (y_grid_var[j] + h_var[j]) >= 0 - M * (1 - z2U[(i, j)]),
            name=f"vertical_up_dist_lb_{i}_{j}"
        )
        model.addConstr(
            (y_grid_var[j] + h_var[j]) - EMIB_y_grid_var[(i, j)] >= emib_node.EMIB_bump_width - eps - M * (1 - z2U[(i, j)]),
            name=f"EMIB_bottom_overlap_up_{i}_{j}"
        )
        model.addConstr(
            (EMIB_y_grid_var[(i, j)] + EMIB_h_var[(i, j)]) - y_grid_var[i] >= emib_node.EMIB_bump_width - eps - M * (1 - z2U[(i, j)]),
            name=f"EMIB_top_overlap_up_{i}_{j}"
        )
        
        
        # 4.4 共享边长约束
        # 约束1：如果水平相邻，那么[i][EMIB][j] 其中 EMIB要在chiplet i和j的垂直方向上重叠
        model.addConstr(
            (y_grid_var[i] + h_var[i]) >=  EMIB_y_grid_var[(i, j)] + EMIB_h_var[(i, j)] - M * (1 - z1[(i, j)]),
            name=f"shared_yi_ub1_{i}_{j}"
        )
        model.addConstr(
            y_grid_var[i] <= EMIB_y_grid_var[(i, j)] + M * (1 - z1[(i, j)]),
            name=f"shared_yi_ub2_{i}_{j}"
        )
        model.addConstr(
            (y_grid_var[j] + h_var[j]) >=  EMIB_y_grid_var[(i, j)] + EMIB_h_var[(i, j)] - M * (1 - z1[(i, j)]),
            name=f"shared_yj_ub1_{i}_{j}"
        )
        model.addConstr(
            y_grid_var[j] <= EMIB_y_grid_var[(i, j)] + M * (1 - z1[(i, j)]),
            name=f"shared_yj_ub2_{i}_{j}"
        )

        # 约束2：如果垂直相邻（z2=1），EMIB 要在 chiplet i 和 j 的水平方向上重叠
        model.addConstr(
            (x_grid_var[i] + w_var[i]) >= EMIB_x_grid_var[(i, j)] + EMIB_w_var[(i, j)] - M * (1 - z2[(i, j)]),
            name=f"shared_xi_ub1_{i}_{j}"
        )
        model.addConstr(
            x_grid_var[i] <= EMIB_x_grid_var[(i, j)] + M * (1 - z2[(i, j)]),
            name=f"shared_xi_ub2_{i}_{j}"
        )
        model.addConstr(
            (x_grid_var[j] + w_var[j]) >= EMIB_x_grid_var[(i, j)] + EMIB_w_var[(i, j)] - M * (1 - z2[(i, j)]),
            name=f"shared_xj_ub1_{i}_{j}"
        )
        model.addConstr(
            x_grid_var[j] <= EMIB_x_grid_var[(i, j)] + M * (1 - z2[(i, j)]),
            name=f"shared_xj_ub2_{i}_{j}"
        )
    
    # chiplet的非重叠约束
    # 4.5 非重叠约束
    # 定义非重叠约束的二进制变量（对于所有模块对，不仅仅是连接的）
    p_left = {}
    p_right = {}
    p_down = {}
    p_up = {}
    
    all_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append((i, j))
            p_left[(i, j)] = model.addVar(name=f"p_left_{i}_{j}", vtype=GRB.BINARY)
            p_right[(i, j)] = model.addVar(name=f"p_right_{i}_{j}", vtype=GRB.BINARY)
            p_down[(i, j)] = model.addVar(name=f"p_down_{i}_{j}", vtype=GRB.BINARY)
            p_up[(i, j)] = model.addVar(name=f"p_up_{i}_{j}", vtype=GRB.BINARY)

    # 对于每对模块 (i, j)
    for i, j in all_pairs:
        # 放宽约束：至少满足一个非重叠条件
        model.addConstr(
            p_left[(i, j)] + p_right[(i, j)] + p_down[(i, j)] + p_up[(i, j)] >= 1,
            name=f"non_overlap_any_{i}_{j}"
        )
        
        # 情况1: i 在 j 的左边（x_i + w_i <= x_j）
        # 正向约束：p_left=1 → x_i + w_i <= x_j
        model.addConstr(
            x_grid_var[i] + w_var[i] - x_grid_var[j] <= M * (1 - p_left[(i, j)]),
            name=f"non_overlap_left_{i}_{j}"
        )
        # 修复漏洞2：正确的反向约束（x_i + w_i <= x_j → p_left=1）
        model.addConstr(
            x_grid_var[j] - (x_grid_var[i] + w_var[i]) <= M * p_left[(i, j)],
            name=f"non_overlap_left_rev_{i}_{j}"
        )
        
        # 情况2: i 在 j 的右边（x_j + w_j <= x_i）
        model.addConstr(
            x_grid_var[j] + w_var[j] - x_grid_var[i] <= M * (1 - p_right[(i, j)]),
            name=f"non_overlap_right_{i}_{j}"
        )
        model.addConstr(
            x_grid_var[i] - (x_grid_var[j] + w_var[j]) <= M * p_right[(i, j)],
            name=f"non_overlap_right_rev_{i}_{j}"
        )
        
        # 情况3: i 在 j 的下边（y_i + h_i <= y_j）
        model.addConstr(
            y_grid_var[i] + h_var[i] - y_grid_var[j] <= M * (1 - p_down[(i, j)]),
            name=f"non_overlap_down_{i}_{j}"
        )
        model.addConstr(
            y_grid_var[j] - (y_grid_var[i] + h_var[i]) <= M * p_down[(i, j)],
            name=f"non_overlap_down_rev_{i}_{j}"
        )
        
        # 情况4: i 在 j 的上边（y_j + h_j <= y_i）
        model.addConstr(
            y_grid_var[j] + h_var[j] - y_grid_var[i] <= M * (1 - p_up[(i, j)]),
            name=f"non_overlap_up_{i}_{j}"
        )
        model.addConstr(
            y_grid_var[i] - (y_grid_var[j] + h_var[j]) <= M * p_up[(i, j)],
            name=f"non_overlap_up_rev_{i}_{j}"
        )

    # 4.5.2 EMIB 之间的非重叠约束
    # 规则：若 EMIB 连接 chiplet A 和 B，筛选出所有「其中一端为 A 或其中一端为 B」的 EMIB，
    #       为当前 EMIB 与这些 EMIB 两两添加非重叠约束。（即共享至少一个 chiplet 的 EMIB 对需非重叠）
    emib_list = list(EMIB_connected_pairs.items())
    emib_non_overlap_pairs = []  # ((i,j), (k,l)) 需要添加非重叠约束的 EMIB 对
    for idx_a in range(len(emib_list)):
        (i, j), _ = emib_list[idx_a]
        chips_a = {i, j}  # 当前 EMIB 连接的两个 chiplet 索引
        for idx_b in range(idx_a + 1, len(emib_list)):
            (k, l), _ = emib_list[idx_b]
            chips_b = {k, l}
            if chips_a & chips_b:  # 共享至少一个 chiplet（一端为 A 或 B）
                emib_non_overlap_pairs.append(((i, j), (k, l)))

    p_EMIB_left = {}
    p_EMIB_right = {}
    p_EMIB_down = {}
    p_EMIB_up = {}

    for (i, j), (k, l) in emib_non_overlap_pairs:
        key = ((i, j), (k, l))
        p_EMIB_left[key] = model.addVar(name=f"p_EMIB_left_{i}_{j}_{k}_{l}", vtype=GRB.BINARY)
        p_EMIB_right[key] = model.addVar(name=f"p_EMIB_right_{i}_{j}_{k}_{l}", vtype=GRB.BINARY)
        p_EMIB_down[key] = model.addVar(name=f"p_EMIB_down_{i}_{j}_{k}_{l}", vtype=GRB.BINARY)
        p_EMIB_up[key] = model.addVar(name=f"p_EMIB_up_{i}_{j}_{k}_{l}", vtype=GRB.BINARY)

        # 至少满足一个方向（左/右/下/上）才不重叠
        model.addConstr(
            p_EMIB_left[key] + p_EMIB_right[key] + p_EMIB_down[key] + p_EMIB_up[key] >= 1,
            name=f"EMIB_non_overlap_any_{i}_{j}_{k}_{l}"
        )
        # 左侧：(i,j) 在 (k,l) 左边，即 (i,j) 右边界 <= (k,l) 左边界
        model.addConstr(
            EMIB_x_grid_var[(i, j)] + EMIB_w_var[(i, j)] - EMIB_x_grid_var[(k, l)] <= M * (1 - p_EMIB_left[key]),
            name=f"EMIB_non_overlap_left_{i}_{j}_{k}_{l}"
        )
        model.addConstr(
            EMIB_x_grid_var[(k, l)] - (EMIB_x_grid_var[(i, j)] + EMIB_w_var[(i, j)]) <= M * p_EMIB_left[key],
            name=f"EMIB_non_overlap_left_rev_{i}_{j}_{k}_{l}"
        )
        # 右侧：(i,j) 在 (k,l) 右边，即 (k,l) 右边界 <= (i,j) 左边界
        model.addConstr(
            EMIB_x_grid_var[(k, l)] + EMIB_w_var[(k, l)] - EMIB_x_grid_var[(i, j)] <= M * (1 - p_EMIB_right[key]),
            name=f"EMIB_non_overlap_right_{i}_{j}_{k}_{l}"
        )
        model.addConstr(
            EMIB_x_grid_var[(i, j)] - (EMIB_x_grid_var[(k, l)] + EMIB_w_var[(k, l)]) <= M * p_EMIB_right[key],
            name=f"EMIB_non_overlap_right_rev_{i}_{j}_{k}_{l}"
        )
        # 下侧：(i,j) 在 (k,l) 下边，即 (i,j) 上边界 <= (k,l) 下边界
        model.addConstr(
            EMIB_y_grid_var[(i, j)] + EMIB_h_var[(i, j)] - EMIB_y_grid_var[(k, l)] <= M * (1 - p_EMIB_down[key]),
            name=f"EMIB_non_overlap_down_{i}_{j}_{k}_{l}"
        )
        model.addConstr(
            EMIB_y_grid_var[(k, l)] - (EMIB_y_grid_var[(i, j)] + EMIB_h_var[(i, j)]) <= M * p_EMIB_down[key],
            name=f"EMIB_non_overlap_down_rev_{i}_{j}_{k}_{l}"
        )
        # 上侧：(i,j) 在 (k,l) 上边，即 (k,l) 上边界 <= (i,j) 下边界
        model.addConstr(
            EMIB_y_grid_var[(k, l)] + EMIB_h_var[(k, l)] - EMIB_y_grid_var[(i, j)] <= M * (1 - p_EMIB_up[key]),
            name=f"EMIB_non_overlap_up_{i}_{j}_{k}_{l}"
        )
        model.addConstr(
            EMIB_y_grid_var[(i, j)] - (EMIB_y_grid_var[(k, l)] + EMIB_h_var[(k, l)]) <= M * p_EMIB_up[key],
            name=f"EMIB_non_overlap_up_rev_{i}_{j}_{k}_{l}"
        )

    # if verbose and emib_non_overlap_pairs:
    #     print(f"EMIB 非重叠约束: {len(emib_non_overlap_pairs)} 对（共享至少一端 chiplet 的硅桥）")
    #     for (i, j), (k, l) in emib_non_overlap_pairs:
    #         na, nb = nodes[i].name, nodes[j].name
    #         nc, nd = nodes[k].name, nodes[l].name
    #         print(f"  EMIB ({na}-{nb}) vs ({nc}-{nd})")

    
    # 4.4.2 standard 连接：允许更宽松的约束（不强制紧邻，但鼓励靠近）
    # 对于 standard 连接，不强制紧邻
    # 可以通过线长目标函数来鼓励它们靠近
    # 不添加额外的相邻约束，让优化器通过最小化线长来自然靠近

    # 4.6 外接方框约束
    bbox_min_x = model.addVar(name="bbox_min_x", lb=0, ub=W, vtype=GRB.CONTINUOUS)
    bbox_max_x = model.addVar(name="bbox_max_x", lb=0, ub=W, vtype=GRB.CONTINUOUS)
    bbox_min_y = model.addVar(name="bbox_min_y", lb=0, ub=H, vtype=GRB.CONTINUOUS)
    bbox_max_y = model.addVar(name="bbox_max_y", lb=0, ub=H, vtype=GRB.CONTINUOUS)
    bbox_w = model.addVar(name="bbox_w", lb=0, ub=W, vtype=GRB.CONTINUOUS)
    bbox_h = model.addVar(name="bbox_h", lb=0, ub=H, vtype=GRB.CONTINUOUS)
    
    # 左下角坐标 + 宽度/高度 作为外接方框的边界
    for k in range(n):
        model.addConstr(bbox_min_x <= x_grid_var[k], name=f"bbox_min_x_{k}")
        model.addConstr(bbox_max_x >= x_grid_var[k] + w_var[k], name=f"bbox_max_x_{k}")
        model.addConstr(bbox_min_y <= y_grid_var[k], name=f"bbox_min_y_{k}")
        model.addConstr(bbox_max_y >= y_grid_var[k] + h_var[k], name=f"bbox_max_y_{k}")
    
    model.addConstr(bbox_w == bbox_max_x - bbox_min_x, name="bbox_w_def")
    model.addConstr(bbox_h == bbox_max_y - bbox_min_y, name="bbox_h_def")

    # 4.7 中心点坐标约束
    model.addConstr(cx_center == (bbox_max_x + bbox_min_x) / 2, name=f"cx_center_def")
    model.addConstr(cy_center == (bbox_max_y + bbox_min_y) / 2, name=f"cy_center_def")
    
    # 4.8 长宽比约束
    # if min_aspect_ratio is not None:
    #     # bbox_w / bbox_h >= min_aspect_ratio
    #     # 转换为线性约束: bbox_w >= min_aspect_ratio * bbox_h
    #     model.addConstr(
    #         bbox_w >= min_aspect_ratio * bbox_h,
    #         name="aspect_ratio_min"
    #     )
    #     if verbose:
    #         print(f"长宽比约束: bbox_w/bbox_h >= {min_aspect_ratio}")
    
    # if max_aspect_ratio is not None:
    #     # bbox_w / bbox_h <= max_aspect_ratio
    #     # 转换为线性约束: bbox_w <= max_aspect_ratio * bbox_h
    #     model.addConstr(
    #         bbox_w <= max_aspect_ratio * bbox_h,
    #         name="aspect_ratio_max"
    #     )
    #     if verbose:
    #         print(f"长宽比约束: bbox_w/bbox_h <= {max_aspect_ratio}")
    
    # 4.7 长宽比优化目标（最小化长宽比与理想值的偏差）
    # 理想长宽比设为1.0（正方形），使用 |bbox_w/bbox_h - 1| 的线性近似
    # aspect_ratio_penalty = None

    aspect_ratio_penalty = model.addVar(
        name="aspect_ratio_penalty",
        lb=0,
        ub=max(W, H),
        vtype=GRB.CONTINUOUS
    )
    # |bbox_w - bbox_h| <= aspect_ratio_diff
    model.addConstr(
        aspect_ratio_penalty >= bbox_w - bbox_h,
        name="aspect_ratio_diff_ge_w_minus_h"
    )
    model.addConstr(
        aspect_ratio_penalty >= bbox_h - bbox_w,
        name="aspect_ratio_diff_ge_h_minus_w"
    )

    
    # 4.8 功耗感知优化目标： 让功耗更大的芯片互相离得更远（power_aware_enabled 开关）
    power_aware_enabled = False #暂时关闭功耗感知优化
    power_aware_penalty = None
    if power_aware_enabled:
        power_aware_penalty = model.addVar(
            name="power_aware_penalty",
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS
        )
        power_aware_expr = gp.LinExpr()

        # ===================== 筛选单位面积功耗排名前30%的芯粒 =====================
        high_power_indices, density_threshold = select_high_power_indices_by_density(
            n, nodes, chiplet_w_orig_grid, chiplet_h_orig_grid, top_ratio=0.3
        )

        # ===================== 高功耗芯粒互相远离 =====================
        # 仅当高功耗芯粒数量>=2时，为这些芯粒两两添加互相远离约束
        if len(high_power_indices) >= 2:
            high_power_pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if i in high_power_indices and j in high_power_indices]
            print(f"[DEBUG] 高功耗芯粒互相远离: {len(high_power_pairs)} 对")
            for i, j in high_power_pairs:
                power_i = float(getattr(nodes[i], "power", 0.0) or 0.0)
                power_j = float(getattr(nodes[j], "power", 0.0) or 0.0)
                power_weight_ij = power_i * power_j
                if power_weight_ij == 0.0:
                    continue

                dx_grid_abs_ij = model.addVar(
                    name=f"dx_grid_abs_pair_{i}_{j}",
                    lb=0,
                    ub=W,
                    vtype=GRB.CONTINUOUS
                )
                dy_grid_abs_ij = model.addVar(
                    name=f"dy_grid_abs_pair_{i}_{j}",
                    lb=0,
                    ub=H,
                    vtype=GRB.CONTINUOUS
                )
                dx_grid_diff = model.addVar(
                    name=f"dx_grid_diff_{i}_{j}",
                    lb=-W,
                    ub=W,
                    vtype=GRB.CONTINUOUS
                )
                model.addConstr(
                    dx_grid_diff == cx_grid_var[i] - cx_grid_var[j],
                    name=f"dx_grid_diff_def_{i}_{j}"
                )
                add_absolute_value_constraint_big_m(
                    model=model,
                    abs_var=dx_grid_abs_ij,
                    orig_var=dx_grid_diff,
                    M=M,
                    constraint_prefix=f"dx_grid_abs_pair_{i}_{j}",
                )
                dy_grid_diff = model.addVar(
                    name=f"dy_grid_diff_{i}_{j}",
                    lb=-H,
                    ub=H,
                    vtype=GRB.CONTINUOUS
                )
                model.addConstr(
                    dy_grid_diff == cy_grid_var[i] - cy_grid_var[j],
                    name=f"dy_grid_diff_def_{i}_{j}"
                )
                add_absolute_value_constraint_big_m(
                    model=model,
                    abs_var=dy_grid_abs_ij,
                    orig_var=dy_grid_diff,
                    M=M,
                    constraint_prefix=f"dy_grid_abs_pair_{i}_{j}",
                )
                dist_curr_ij = model.addVar(
                    name=f"dist_curr_pair_{i}_{j}",
                    lb=0,
                    ub=W + H,
                    vtype=GRB.CONTINUOUS
                )
                model.addConstr(
                    dist_curr_ij == dx_grid_abs_ij + dy_grid_abs_ij,
                    name=f"dist_curr_pair_def_{i}_{j}"
                )
                power_aware_expr += power_weight_ij * dist_curr_ij
        else:
            print("[DEBUG] 高功耗芯粒数量<2，跳过互相远离约束")

        # ===================== 高功耗芯粒远离布局中心 =====================
        print(f"[DEBUG] 处理 {n} 个chiplet，用于功耗感知优化（远离中心）")

        # ===================== 新增：筛选功耗前30%的芯粒 =====================
        # 已在上文通过 select_high_power_indices_by_density 得到 high_power_indices，
        # 此处直接复用该集合来添加“远离中心”约束。
        # 处理边界情况：若前面未选出任何高功耗芯粒，则跳过远离中心约束
        if not high_power_indices:
            print("[DEBUG] 无有效功耗/面积的芯粒，跳过远离中心约束")
        else:
            # 仅对 high_power_indices 中的芯粒（单位面积功耗≥阈值），添加远离中心约束
            high_power_count = 0
            for i in range(n):
                if i not in high_power_indices:
                    continue
                p_i = float(getattr(nodes[i], "power", 0.0) or 0.0)
                high_power_count += 1
                print(
                    f"[DEBUG] chiplet {i} 属于单位面积功耗前30%，添加远离中心约束"
                )

                # 1. 计算芯粒i到中心的x方向差值（cx_grid_var[i] - 中心x）
                dx_center_diff_i = model.addVar(
                    name=f"dx_center_diff_{i}",
                    lb=-W,
                    ub=W,
                    vtype=GRB.CONTINUOUS
                )
                model.addConstr(
                    dx_center_diff_i == cx_grid_var[i] - cx_center,
                    name=f"dx_center_diff_def_{i}"
                )

                # 2. 绝对值约束：dx_center_abs_i = |cx_grid_var[i] - center_x_grid|
                dx_center_abs_i = model.addVar(
                    name=f"dx_center_abs_{i}",
                    lb=0,
                    ub=W,
                    vtype=GRB.CONTINUOUS
                )
                add_absolute_value_constraint_big_m(
                    model=model,
                    abs_var=dx_center_abs_i,
                    orig_var=dx_center_diff_i,
                    M=M,
                    constraint_prefix=f"dx_center_abs_{i}",
                )

                # 3. 计算芯粒i到中心的y方向差值（cy_grid_var[i] - 中心y）
                dy_center_diff_i = model.addVar(
                    name=f"dy_center_diff_{i}",
                    lb=-H,
                    ub=H,
                    vtype=GRB.CONTINUOUS
                )
                model.addConstr(
                    dy_center_diff_i == cy_grid_var[i] - cy_center,
                    name=f"dy_center_diff_def_{i}"
                )

                # 4. 绝对值约束：dy_center_abs_i = |cy_grid_var[i] - center_y_grid|
                dy_center_abs_i = model.addVar(
                    name=f"dy_center_abs_{i}",
                    lb=0,
                    ub=H,
                    vtype=GRB.CONTINUOUS
                )
                add_absolute_value_constraint_big_m(
                    model=model,
                    abs_var=dy_center_abs_i,
                    orig_var=dy_center_diff_i,
                    M=M,
                    constraint_prefix=f"dy_center_abs_{i}",
                )

                # 5. 芯粒i到中心的曼哈顿距离 = |dx| + |dy|
                dist_center_i = model.addVar(
                    name=f"dist_center_{i}",
                    lb=0,
                    ub=W+H,
                    vtype=GRB.CONTINUOUS
                )
                model.addConstr(
                    dist_center_i == dx_center_abs_i + dy_center_abs_i,
                    name=f"dist_center_def_{i}"
                )

                # ===================== 关键：仅高功耗芯粒加入惩罚项 =====================
                power_aware_expr += p_i * p_i *  dist_center_i

        print(f"[DEBUG] 共筛选出 {high_power_count} 个高功耗芯粒（前30%），完成远离中心约束添加")
        # 定义功耗惩罚项变量：power_aware_penalty = Σ p_i*p_j*(|dx|+|dy|)
        model.addConstr(power_aware_penalty == power_aware_expr, name="power_aware_penalty_def")

    # ============ 步骤5: 定义目标函数 ============
    # 5.1 线长（曼哈顿距离）
    # wirelength = Σ wireCount * 线长，分两种累加方式：
    #   - interfaceC（基板走线）：chiplet 中心到 chiplet 中心，线长 = |cx_i - cx_j| + |cy_i - cy_j|
    #   - interfaceB/interfaceA（硅桥）：chiplet→EMIB→chiplet，线长 = (chiplet_i 到 EMIB) + (chiplet_j 到 EMIB)
    # 注意：cx/cy 为 2×中心坐标，线长量级一致，优化时不影响
    wirelength = model.addVar(
        name="wirelength",
        lb=0,
        ub=1024 * 4.0 * (W + H) * max(1, len(all_connected_pairs)),
        vtype=GRB.CONTINUOUS,
    )
    wirelength_sum = gp.LinExpr()

    # 5.1.1 interfaceC：chiplet 到 chiplet 直连，累加 wireCount * (|dx| + |dy|)
    for (i, j), edge in all_connected_pairs.items():
        if edge.EMIBType != "interfaceC":
            continue
        wire_count = edge.wireCount
        dx_abs = model.addVar(name=f"dx_abs_{i}_{j}", lb=0, vtype=GRB.CONTINUOUS)
        dy_abs = model.addVar(name=f"dy_abs_{i}_{j}", lb=0, vtype=GRB.CONTINUOUS)
        dx_diff = model.addVar(name=f"dx_diff_{i}_{j}", lb=-W, ub=W, vtype=GRB.CONTINUOUS)
        dy_diff = model.addVar(name=f"dy_diff_{i}_{j}", lb=-H, ub=H, vtype=GRB.CONTINUOUS)
        model.addConstr(dx_diff == cx_grid_var[i] - cx_grid_var[j], name=f"dx_diff_def_{i}_{j}")
        model.addConstr(dy_diff == cy_grid_var[i] - cy_grid_var[j], name=f"dy_diff_def_{i}_{j}")
        add_absolute_value_constraint_big_m(
            model=model, abs_var=dx_abs, orig_var=dx_diff, M=M,
            constraint_prefix=f"dx_abs_{i}_{j}",
        )
        add_absolute_value_constraint_big_m(
            model=model, abs_var=dy_abs, orig_var=dy_diff, M=M,
            constraint_prefix=f"dy_abs_{i}_{j}",
        )
        wirelength_sum += wire_count * (dx_abs + dy_abs)

    # 5.1.2 interfaceB/interfaceA（EMIB）：chiplet→EMIB→chiplet，累加 wireCount * (dist_i + dist_j)
    # dist_i = |chiplet_i 中心 - EMIB 中心| 曼哈顿，dist_j 同理
    for (i, j), edge in EMIB_connected_pairs.items():
        wire_count = edge.wireCount
        dx_abs_i = model.addVar(name=f"dx_abs_i_{i}_{j}", lb=0, vtype=GRB.CONTINUOUS)
        dy_abs_i = model.addVar(name=f"dy_abs_i_{i}_{j}", lb=0, vtype=GRB.CONTINUOUS)
        dx_abs_j = model.addVar(name=f"dx_abs_j_{i}_{j}", lb=0, vtype=GRB.CONTINUOUS)
        dy_abs_j = model.addVar(name=f"dy_abs_j_{i}_{j}", lb=0, vtype=GRB.CONTINUOUS)
        dx_diff_i = model.addVar(name=f"dx_diff_i_{i}_{j}", lb=-W, ub=W, vtype=GRB.CONTINUOUS)
        dy_diff_i = model.addVar(name=f"dy_diff_i_{i}_{j}", lb=-H, ub=H, vtype=GRB.CONTINUOUS)
        dx_diff_j = model.addVar(name=f"dx_diff_j_{i}_{j}", lb=-W, ub=W, vtype=GRB.CONTINUOUS)
        dy_diff_j = model.addVar(name=f"dy_diff_j_{i}_{j}", lb=-H, ub=H, vtype=GRB.CONTINUOUS)
        model.addConstr(dx_diff_i == cx_grid_var[i] - EMIB_cx_grid_var[(i, j)], name=f"dx_diff_def_i_{i}_{j}")
        model.addConstr(dx_diff_j == cx_grid_var[j] - EMIB_cx_grid_var[(i, j)], name=f"dx_diff_def_j_{i}_{j}")
        model.addConstr(dy_diff_i == cy_grid_var[i] - EMIB_cy_grid_var[(i, j)], name=f"dy_diff_def_i_{i}_{j}")
        model.addConstr(dy_diff_j == cy_grid_var[j] - EMIB_cy_grid_var[(i, j)], name=f"dy_diff_def_j_{i}_{j}")
        add_absolute_value_constraint_big_m(
            model=model, abs_var=dx_abs_i, orig_var=dx_diff_i, M=M,
            constraint_prefix=f"dx_abs_i_{i}_{j}",
        )
        add_absolute_value_constraint_big_m(
            model=model, abs_var=dy_abs_i, orig_var=dy_diff_i, M=M,
            constraint_prefix=f"dy_abs_i_{i}_{j}",
        )
        add_absolute_value_constraint_big_m(
            model=model, abs_var=dx_abs_j, orig_var=dx_diff_j, M=M,
            constraint_prefix=f"dx_abs_j_{i}_{j}",
        )
        add_absolute_value_constraint_big_m(
            model=model, abs_var=dy_abs_j, orig_var=dy_diff_j, M=M,
            constraint_prefix=f"dy_abs_j_{i}_{j}",
        )
        wirelength_sum += wire_count * (dx_abs_i + dy_abs_i + dx_abs_j + dy_abs_j)

    model.addConstr(wirelength == wirelength_sum, name="wirelength_def")

    # 5.0 计算归一化基准（先验估算，用于多目标量级对齐）
    ref_wirelength, ref_t, ref_power, ref_aspect = compute_normalization_factors(
        n=n,
        nodes=nodes,
        chiplet_w_orig_grid=chiplet_w_orig_grid,
        chiplet_h_orig_grid=chiplet_h_orig_grid,
        all_connected_pairs=all_connected_pairs,
        power_aware_enabled=power_aware_enabled,
    )

    # 5.2 面积代理
    t = model.addVar(
        name="bbox_area_proxy_t",
        lb=0,
        ub=W+H,
        vtype=GRB.CONTINUOUS
    )
    # 4. 核心约束：让 t 合理代理面积（无冲突、紧凑）
    ## 约束1：t 至少 ≥ 宽/高（保证 t 不小于单个维度）
    # model.addConstr(t >= bbox_w, name="t_ge_width")
    # model.addConstr(t >= bbox_h, name="t_ge_height")
    
    ## 约束2：t 至少 ≥ 宽×高的"线性近似"（关键：用均值放大系数逼近面积）
    # 系数 alpha 取 0.5~1（平衡近似精度和约束紧凑性）
    alpha = 0.8
    model.addConstr(t >= alpha * (bbox_w + bbox_h), name="t_ge_scaled_mean")
    
    # 5.4 归一化后的目标函数
    # 权重 beta 表示相对重要程度，各项已除以参考值进行量级对齐
    beta_wire = _get_beta_from_env("EMIB_BETA_WIRE", 5.0)
    beta_area = _get_beta_from_env("EMIB_BETA_AREA", 20.0)
    beta_aspect = _get_beta_from_env("EMIB_BETA_ASPECT", 0.1)
    beta_power = _get_beta_from_env("EMIB_BETA_POWER", 0.0)
    # aspect_ratio_penalty = 1
    # power_aware_penalty = 1
    # 归一化项：每项除以参考值，使量级相近
    norm_wirelength = (1.0 / ref_wirelength) * wirelength
    norm_t = (1.0 / ref_t) * t
    norm_aspect = ((1.0 / ref_aspect) * aspect_ratio_penalty)
    norm_power = ((1.0 / ref_power) * power_aware_penalty) if power_aware_penalty is not None else 0

    # 最终目标：Minimize (W_norm + A_norm + Asp_norm - P_norm)
    # 功耗项取负号：越大越好（高功耗 chiplet 分散）
    objective = (
        beta_wire * norm_wirelength
        + beta_area * norm_t
        + beta_aspect * norm_aspect
        - beta_power * norm_power
    )
    model.setObjective(objective, GRB.MINIMIZE)
    if verbose:
        print(f"\n[Normalization Info]")
        print(f"  Ref Wirelength: {ref_wirelength:.2f}")
        print(f"  Ref Area Proxy (t): {ref_t:.2f}")
        print(f"  Ref Power Term: {ref_power:.2f}")
        print(f"  Ref Aspect: {ref_aspect:.2f}")
        obj_parts = [
            f"beta_wire({beta_wire})*wirelength/ref",
            f"beta_area({beta_area})*t/ref",
            f"beta_aspect({beta_aspect})*aspect_ratio/ref"
        ]
        if power_aware_penalty is not None:
            obj_parts.append(f"- beta_power({beta_power})*power/ref")
        print("目标函数: Min " + " + ".join(obj_parts))

    
    return ILPModelContext(
        model=model,
        nodes=nodes,
        edges=edges,
        x_grid_var=x_grid_var,
        y_grid_var=y_grid_var,
        r=r,
        z1=z1,
        z2=z2,
        z1L=z1L,
        z1R=z1R,
        z2D=z2D,
        z2U=z2U,
        all_connected_pairs=all_connected_pairs,
        bbox_w=bbox_w,
        bbox_h=bbox_h,
        W=W,
        H=H,
        fixed_chiplet_idx=fixed_chiplet_idx,
        cx_grid_var=cx_grid_var,
        cy_grid_var=cy_grid_var,
        ref_wirelength=ref_wirelength,
        ref_t=ref_t,
        ref_power=ref_power,
        ref_aspect=ref_aspect,
        beta_wire=beta_wire,
        beta_area=beta_area,
        beta_aspect=beta_aspect,
        beta_power=beta_power,
        EMIB_connected_pairs=EMIB_connected_pairs,
        EMIB_x_grid_var=EMIB_x_grid_var,
        EMIB_y_grid_var=EMIB_y_grid_var,
        EMIB_w_var=EMIB_w_var,
        EMIB_h_var=EMIB_h_var,
        r_EMIB=r_EMIB,
    )


def main():
    """
    主函数：使用连续坐标ILP模型进行单次求解并可视化结果。
    """
    from pathlib import Path

    # 设置参数
    time_limit = 600  # 10分钟
    min_shared_length = 0.1
    fixed_chiplet_idx = None  # 不再使用固定芯粒约束
    
    # 输出目录
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # JSON文件路径
    json_path = Path(__file__).parent.parent / "baseline" / "ICCAD23" / "test_input" / "2core.json"
    
    print("=" * 80)
    print("ILP单次求解测试 (Gurobi版本)")
    print("=" * 80)
    
    # 从JSON文件加载测试数据（6 元组格式）
    print(f"\n从JSON文件加载测试数据: {json_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON文件不存在: {json_path}")

    from tool import load_emib_placement_json
    nodes, edges, edge_map, name_to_idx = load_emib_placement_json(str(json_path))
    
    print(f"节点数量: {len(nodes)}")
    print(f"边数量: {len(edges)}")
    print(f"节点列表: {[n.name for n in nodes]}")
    print(f"边列表: {edges}")
    
    # 构建ILP模型
    print("\n构建ILP模型...")
    ctx = build_placement_ilp_model(
        nodes=nodes,
        edges=edges,
        W=None,  # 自动估算
        H=None,  # 自动估算
        verbose=True,
        min_shared_length=min_shared_length,
        minimize_bbox_area=True,
        distance_weight=1.0,
        area_weight=0.1,
        fixed_chiplet_idx=fixed_chiplet_idx,
    )
    
    # 导出LP文件
    lp_file = output_dir / "ilp_model_gurobi.lp"
    ctx.model.write(str(lp_file))
    print(f"LP模型文件已导出至: {lp_file}")
    
    # 求解
    print("\n开始求解...")
    result = solve_placement_ilp_from_model(
        ctx,
        time_limit=time_limit,
        verbose=True,
    )
    
    # 输出结果
    print("\n" + "=" * 80)
    print("求解结果")
    print("=" * 80)
    print(f"状态: {result.status}")
    print(f"求解时间: {result.solve_time:.2f} 秒")
    print(f"目标函数值: {result.objective_value:.2f}")
    print(f"边界框尺寸: {result.bounding_box[0]:.2f} x {result.bounding_box[1]:.2f}")
    
    print("\n布局结果:")
    for name, (x, y) in result.layout.items():
        rotated = result.rotations.get(name, False)
        rot_str = " (已旋转)" if rotated else ""
        print(f"  {name}: ({x:.2f}, {y:.2f}){rot_str}")
    
    # 可视化结果
    if result.status == "Optimal":
        print("\n生成可视化图表...")
        try:
            save_path = output_dir / "ilp_single_solution_gurobi.png"
            
            draw_edges = [(e["node1"], e["node2"], e["EMIBType"]) for e in edges]
            draw_chiplet_diagram(
                nodes=nodes,
                edges=draw_edges,
                layout=result.layout,  # 网格坐标
                save_path=str(save_path),
                rotations=result.rotations,
            )
            print(f"图表已保存至: {save_path}")
        except Exception as e:
            print(f"可视化失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n求解未达到最优解，跳过可视化")
    
    print("\n" + "=" * 80)
    print("完成")
    print("=" * 80)


if __name__ == "__main__":
    main()

