import json
import math
import os
import re
import shutil

import numpy as np

# 网格单位：0.01mm（所有网格点坐标为 0.01 的整数倍）
GRID_MM = 0.01
# 最小块尺寸（mm），避免 degenerate 块
MIN_DIMENSION_MM = 0.01
# TIM 块最小宽高（mm），过小会导致 HotSpot 热仿真失败
MIN_TIM_DIMENSION_MM = 0.02
# 输出精度（米，小数位）
OUTPUT_DECIMALS = 6

def round_to_grid_mm(value_mm):
    """将毫米值对齐到 0.01mm 网格（四舍五入），并舍入到2位小数避免浮点漂移导致重叠"""
    return round(round(float(value_mm) / GRID_MM) * GRID_MM, 2)

def mm_to_m(value_mm):
    """毫米转米，保留 OUTPUT_DECIMALS 位小数"""
    return round(float(value_mm) / 1000.0, OUTPUT_DECIMALS)

OVERLAP_TOLERANCE_MM2 = 1e-6
OVERLAP_TOLERANCE_M2 = 1e-12  # 米单位下的浮点容差

def _rects_overlap_pair(a, b, tol_area):
    """判断两矩形是否重叠（重叠面积>tol 则视为重叠）"""
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    x_ol = max(0, min(ax2, bx2) - max(ax1, bx1))
    y_ol = max(0, min(ay2, by2) - max(ay1, by1))
    return x_ol * y_ol > tol_area

def check_blocks_overlap(block_list, tol_area=OVERLAP_TOLERANCE_MM2):
    """检查块列表是否有重叠，有则抛异常。可忽略浮点误差级重叠（边界贴合）"""
    n = len(block_list)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = block_list[i], block_list[j]
            if _rects_overlap_pair(a, b, tol_area):
                raise ValueError(
                    f"布局重叠: {a['name']} 与 {b['name']}\n"
                    f"  {a['name']}: x={a['x']}, y={a['y']}, w={a['w']}, h={a['h']}\n"
                    f"  {b['name']}: x={b['x']}, y={b['y']}, w={b['w']}, h={b['h']}"
                )
    print("✅ 块列表无重叠")

def check_no_grid_overlap(chiplet_list, tim_list, unit_m=False):
    """
    全部处理完毕后，检查 chiplet 之间、TIM 块之间是否存在网格重叠。
    unit_m: True 表示块坐标为米，使用米单位容差；False 表示毫米
    """
    tol = OVERLAP_TOLERANCE_M2 if unit_m else OVERLAP_TOLERANCE_MM2
    # chiplet 之间
    for i in range(len(chiplet_list)):
        for j in range(i + 1, len(chiplet_list)):
            a, b = chiplet_list[i], chiplet_list[j]
            if _rects_overlap_pair(a, b, tol):
                raise ValueError(
                    f"Chiplet 网格重叠: {a['name']} 与 {b['name']}\n"
                    f"  {a['name']}: x={a['x']}, y={a['y']}, w={a['w']}, h={a['h']}\n"
                    f"  {b['name']}: x={b['x']}, y={b['y']}, w={b['w']}, h={b['h']}"
                )
    # TIM 块之间
    for i in range(len(tim_list)):
        for j in range(i + 1, len(tim_list)):
            a, b = tim_list[i], tim_list[j]
            if _rects_overlap_pair(a, b, tol):
                raise ValueError(
                    f"TIM 块网格重叠: {a['name']} 与 {b['name']}\n"
                    f"  {a['name']}: x={a['x']}, y={a['y']}, w={a['w']}, h={a['h']}\n"
                    f"  {b['name']}: x={b['x']}, y={b['y']}, w={b['w']}, h={b['h']}"
                )
    print("✅ Chiplet 之间、TIM 块之间均无网格重叠")

def generate_ptrace_file(chiplet_list, tim_list, json_path, output_ptrace_path, power_key="power"):
    """根据JSON文件中的power值生成.ptrace文件（同步Chiplet/TIM命名）"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    power_dict = {}
    if 'chiplets' in data:
        for chip in data['chiplets']:
            original_name = chip.get('name')
            if original_name:
                # 直接使用原始名称，不加前缀
                power = float(chip.get(power_key, 0.0))
                power_dict[original_name] = power

    all_modules = tim_list + chiplet_list
    power_values = [
        0.0 if m['name'].startswith('T') and m['name'][1:].isdigit() else power_dict.get(m['name'], 0.0)
        for m in all_modules
    ]
    module_names = [m['name'] for m in all_modules]

    # 写入原始 ptrace
    power_strings = [f"{p:.6f}" for p in power_values]
    with open(output_ptrace_path, 'w', encoding='utf-8') as f:
        f.write(' '.join(module_names) + '\n')
        f.write(' '.join(power_strings) + '\n')
        f.write(' '.join(power_strings) + '\n')

    # 额外生成 10 份：在同一个 system_i_config 下，给每个 chiplet 随机设置 1~200W（TIM 保持 0）
    # 生成文件：system_1.ptrace ... system_10.ptrace
    base_dir = os.path.dirname(os.path.abspath(output_ptrace_path))

    chiplet_idxs = [
        idx for idx, name in enumerate(module_names)
        if not (name.startswith('T') and name[1:].isdigit())
    ]

    rng = np.random.default_rng()
    for j in range(1, 11):
        swapped = power_values.copy()
        for idx in chiplet_idxs:
            orig = float(power_values[idx])
            # 强保证：随机功耗一定不等于原始 JSON power（相等就重新抽）
            v = float(rng.integers(1, 201))  # [1, 200]
            while abs(v - orig) < 1e-9:
                v = float(rng.integers(1, 201))
            swapped[idx] = v

        out_path = os.path.join(base_dir, f"system_{j}.ptrace")
        swapped_strings = [f"{p:.6f}" for p in swapped]
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(' '.join(module_names) + '\n')
            f.write(' '.join(swapped_strings) + '\n')
            f.write(' '.join(swapped_strings) + '\n')

    print(f"✅ 成功生成.ptrace文件：{output_ptrace_path}")
    print(f"📦 文件包含 {len(all_modules)} 个模块的功耗数据")
    print(f"✅ 同一布局额外生成 10 份随机功耗 ptrace：{base_dir}/system_[1-10].ptrace")

def load_json_layout(json_path):
    """
    1. 基础数据提取：读取 Chiplet 名称、左下角(x,y)、宽(w)、高(h)，单位 mm。
    2. 校验：长宽>0，坐标非负。
    """
    json_path = os.path.abspath(os.path.normpath(json_path))
    print(f"[gen_flp_trace] 读入 JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chiplets = []
    if 'chiplets' not in data:
        raise ValueError("JSON 中未找到 chiplets 字段")
    for idx, chip in enumerate(data['chiplets']):
        original_name = chip.get('name', chr(65 + idx))
        name = original_name  # 只使用大写字母，不加前缀
        x = float(chip['x-position'])
        y = float(chip['y-position'])
        w = float(chip['width'])
        h = float(chip['height'])
        if w <= 0 or h <= 0:
            raise ValueError(f"Chiplet {name}: 长宽必须>0, 当前 w={w}, h={h}")
        # if x < 0 or y < 0:
        #     raise ValueError(f"Chiplet {name}: 坐标必须非负, 当前 x={x}, y={y}")
        chiplets.append({'name': name, 'x': x, 'y': y, 'w': w, 'h': h})
    if not chiplets:
        raise ValueError("JSON 中无有效 Chiplet 数据")
    return chiplets


def _has_any_overlap(chiplet_list, tol=OVERLAP_TOLERANCE_MM2):
    """检查块列表中是否存在任意一对重叠"""
    n = len(chiplet_list)
    for i in range(n):
        for j in range(i + 1, n):
            if _rects_overlap_pair(chiplet_list[i], chiplet_list[j], tol):
                return True
    return False


def build_layout(chiplets):
    """
    2. 新外接框构建 + 3. Chiplet 网格对齐+平移居中
    相对位置不变：若舍入后产生重叠，则扩大画布后整体重新放置（同一套平移+舍入），不单独挪动某一块。
    返回: (chiplet_list_mm, square_side_mm)
    """
    # 先对宽高做网格对齐
    chiplets_grid = []
    for c in chiplets:
        w = round_to_grid_mm(c['w'])
        h = round_to_grid_mm(c['h'])
        w = max(w, MIN_DIMENSION_MM)
        h = max(h, MIN_DIMENSION_MM)
        chiplets_grid.append({'name': c['name'], 'x': c['x'], 'y': c['y'], 'w': w, 'h': h})
    chiplets = chiplets_grid

    min_x = min(c['x'] for c in chiplets)
    min_y = min(c['y'] for c in chiplets)
    max_x = max(c['x'] + c['w'] for c in chiplets)
    max_y = max(c['y'] + c['h'] for c in chiplets)
    old_cx = (min_x + max_x) / 2.0
    old_cy = (min_y + max_y) / 2.0
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    longest_side = round_to_grid_mm(max(bbox_w, bbox_h))

    # 在给定边长下做一次「平移居中 + 网格对齐」放置，相对关系一致
    def place_once(side_mm):
        new_cx = side_mm / 2.0
        new_cy = side_mm / 2.0
        shift_x = new_cx - old_cx
        shift_y = new_cy - old_cy
        out = []
        for c in chiplets:
            x_shifted = c['x'] + shift_x
            y_shifted = c['y'] + shift_y
            x_aligned = round_to_grid_mm(x_shifted)
            y_aligned = round_to_grid_mm(y_shifted)
            w, h = c['w'], c['h']
            x_aligned = round_to_grid_mm(max(0, min(x_aligned, side_mm - w)))
            y_aligned = round_to_grid_mm(max(0, min(y_aligned, side_mm - h)))
            out.append({'name': c['name'], 'x': x_aligned, 'y': y_aligned, 'w': w, 'h': h})
        return out

    chiplet_list = place_once(longest_side)
    max_expand = 10
    expand_step = round_to_grid_mm(0.02)  # 每次扩大 0.02mm

    while _has_any_overlap(chiplet_list) and max_expand > 0:
        longest_side = round_to_grid_mm(longest_side + expand_step)
        chiplet_list = place_once(longest_side)
        max_expand -= 1

    return chiplet_list, longest_side


def _rects_overlap(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    """轴对齐矩形是否重叠"""
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

def _can_merge_h(a, b):
    """水平相邻可合并：同 y、同 h，且左右相邻"""
    (ax, ay, aw, ah), (bx, by, bw, bh) = a, b
    if abs(ay - by) >= 1e-9 or abs(ah - bh) >= 1e-9:
        return False
    if abs((ax + aw) - bx) < 1e-9:
        return (ax, ay, aw + bw, ah)
    if abs((bx + bw) - ax) < 1e-9:
        return (bx, by, aw + bw, ah)
    return False

def _can_merge_v(a, b):
    """垂直相邻可合并：同 x、同 w，且上下相邻"""
    (ax, ay, aw, ah), (bx, by, bw, bh) = a, b
    if abs(ax - bx) >= 1e-9 or abs(aw - bw) >= 1e-9:
        return False
    if abs((ay + ah) - by) < 1e-9:
        return (ax, ay, aw, ah + bh)
    if abs((by + bh) - ay) < 1e-9:
        return (ax, by, aw, ah + bh)
    return False

def _merge_adjacent_rects(rects):
    """合并可组成大矩形的相邻矩形，使 TIM 块数尽可能少"""
    if not rects:
        return []
    lst = [(r['x'], r['y'], r['w'], r['h']) for r in rects]
    while True:
        merged_any = False
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = lst[i], lst[j]
                m = _can_merge_h(a, b) or _can_merge_v(a, b)
                if m:
                    lst[i] = m
                    lst.pop(j)
                    merged_any = True
                    break
            if merged_any:
                break
        if not merged_any:
            break
    return lst

# 全局常量（与主代码保持一致）
GRID_MM = 0.01
MIN_DIMENSION_MM = 0.01
OVERLAP_TOLERANCE_MM2 = 1e-6

def round_to_grid_mm(value_mm):
    """将毫米值对齐到 0.01mm 网格（四舍五入），并舍入到2位小数避免浮点漂移"""
    return round(round(float(value_mm) / GRID_MM) * GRID_MM, 2)

def _rects_overlap(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    """轴对齐矩形是否有重叠（只要有交集即视为重叠）"""
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

def _rect_fully_outside(rect, chiplet):
    """判断一个小矩形是否完全在某个Chiplet之外（不重叠）"""
    rx1, ry1, rx2, ry2 = rect
    cx1, cy1 = chiplet['x'], chiplet['y']
    cx2, cy2 = chiplet['x'] + chiplet['w'], chiplet['y'] + chiplet['h']
    # 完全在左/右/上/下 → 不重叠
    return (rx2 <= cx1) or (rx1 >= cx2) or (ry2 <= cy1) or (ry1 >= cy2)

def _can_merge_h(a, b):
    """水平相邻可合并：同 y、同 h，且左右边界贴合"""
    (ax, ay, aw, ah), (bx, by, bw, bh) = a, b
    if abs(ay - by) >= 1e-9 or abs(ah - bh) >= 1e-9:
        return False
    if abs((ax + aw) - bx) < 1e-9:
        return (ax, ay, aw + bw, ah)
    if abs((bx + bw) - ax) < 1e-9:
        return (bx, by, aw + bw, ah)
    return False

def _can_merge_v(a, b):
    """垂直相邻可合并：同 x、同 w，且上下边界贴合"""
    (ax, ay, aw, ah), (bx, by, bw, bh) = a, b
    if abs(ax - bx) >= 1e-9 or abs(aw - bw) >= 1e-9:
        return False
    if abs((ay + ah) - by) < 1e-9:
        return (ax, ay, aw, ah + bh)
    if abs((by + bh) - ay) < 1e-9:
        return (ax, by, aw, ah + bh)
    return False

def _merge_adjacent_rects(rects):
    """迭代合并相邻矩形，直到无法再合并，得到最少块数"""
    if not rects:
        return []
    lst = [(r['x'], r['y'], r['w'], r['h']) for r in rects]
    while True:
        merged_any = False
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = lst[i], lst[j]
                merged = _can_merge_h(a, b) or _can_merge_v(a, b)
                if merged:
                    lst[i] = merged
                    lst.pop(j)
                    merged_any = True
                    break
            if merged_any:
                break
        if not merged_any:
            break
    return lst

def _is_pure_blank_rect(rect_x1, rect_y1, rect_x2, rect_y2, chiplet_list):
    """判断切割线围成的矩形是否为纯空白（无任何Chiplet覆盖）"""
    for chip in chiplet_list:
        c_x1, c_y1 = chip['x'], chip['y']
        c_x2, c_y2 = chip['x'] + chip['w'], chip['y'] + chip['h']
        
        # 计算交集
        inter_x1 = max(rect_x1, c_x1)
        inter_y1 = max(rect_y1, c_y1)
        inter_x2 = min(rect_x2, c_x2)
        inter_y2 = min(rect_y2, c_y2)
        
        # 有有效交集则不是纯空白（用容差避免浮点误差把“边界贴合”判成重叠）
        if (inter_x2 - inter_x1) > 1e-9 and (inter_y2 - inter_y1) > 1e-9:
            return False
    return True

def _try_merge_rects(rect1, rect2):
    """
    尝试融合两个矩形：
    1. 水平融合：同y、同高度，左右边缘贴合
    2. 垂直融合：同x、同宽度，上下边缘贴合
    返回融合后的矩形（元组），失败则返回None
    """
    (x1, y1, w1, h1), (x2, y2, w2, h2) = rect1, rect2
    
    # 浮点误差容忍（1e-9）
    eps = 1e-9
    
    # 水平融合：y相同、高度相同，且一个的右边缘=另一个的左边缘
    if abs(y1 - y2) < eps and abs(h1 - h2) < eps:
        if abs((x1 + w1) - x2) < eps:
            return (x1, y1, w1 + w2, h1)
        if abs((x2 + w2) - x1) < eps:
            return (x2, y2, w1 + w2, h2)
    
    # 垂直融合：x相同、宽度相同，且一个的上边缘=另一个的下边缘
    if abs(x1 - x2) < eps and abs(w1 - w2) < eps:
        if abs((y1 + h1) - y2) < eps:
            return (x1, y1, w1, h1 + h2)
        if abs((y2 + h2) - y1) < eps:
            return (x1, y2, w1, h1 + h2)
    
    # 无法融合
    return None

def _merge_all_possible_rects(rect_list):
    """
    全量迭代融合：直到没有任何矩形可以融合为止
    输入：[(x,y,w,h), ...] 原始空白矩形列表
    输出：融合后的最少数量矩形列表
    """
    if not rect_list:
        return []
    
    # 复制列表避免修改原数据
    current_rects = rect_list.copy()
    
    # 循环融合：直到一轮迭代中没有任何融合发生
    while True:
        merged = False
        new_rects = []
        # 标记已融合的矩形索引
        merged_indices = set()
        
        # 遍历所有矩形对，尝试融合
        for i in range(len(current_rects)):
            if i in merged_indices:
                continue
            
            current = current_rects[i]
            found_merge = False
            
            # 和后续所有矩形尝试融合
            for j in range(i + 1, len(current_rects)):
                if j in merged_indices:
                    continue
                
                merged_rect = _try_merge_rects(current, current_rects[j])
                if merged_rect:
                    # 融合成功：添加新矩形，标记已融合
                    new_rects.append(merged_rect)
                    merged_indices.add(i)
                    merged_indices.add(j)
                    found_merge = True
                    merged = True
                    break
            
            # 未融合则保留原矩形
            if not found_merge:
                new_rects.append(current)
        
        # 更新矩形列表，无融合则退出
        current_rects = new_rects
        if not merged:
            break
    
    return current_rects

def get_tim_blocks(chiplet_list_mm, square_side_mm):
    """
    最终版 TIM 生成逻辑：
    1. 天然边界分割出最小空白矩形（边缘贴合Chiplet/外接框）
    2. 全量迭代融合，最大化减少TIM块数量
    3. 保证填满所有空白，且TIM块数量最少
    输入/输出单位：mm
    """
    # ========== 步骤1：收集所有"天然边界"（Chiplet边缘+外接框边缘），全部对齐到网格避免浮点导致重叠 ==========
    x_edges = {round_to_grid_mm(0.0), round_to_grid_mm(square_side_mm)}
    y_edges = {round_to_grid_mm(0.0), round_to_grid_mm(square_side_mm)}
    
    for chip in chiplet_list_mm:
        x_edges.add(round_to_grid_mm(chip['x']))
        x_edges.add(round_to_grid_mm(chip['x'] + chip['w']))
        y_edges.add(round_to_grid_mm(chip['y']))
        y_edges.add(round_to_grid_mm(chip['y'] + chip['h']))
    
    # 排序边缘
    x_edges_sorted = sorted(list(x_edges))
    y_edges_sorted = sorted(list(y_edges))

    # ========== 步骤2：分割出所有纯空白的最小矩形 ==========
    # 贴边（0 或 square_side_mm）的薄条不跳过，保证主 FLP 填满整块，各层 width/height 一致（HotSpot 要求）
    eps = 1e-6
    raw_blank_rects = []
    for i in range(len(x_edges_sorted) - 1):
        rect_x1 = x_edges_sorted[i]
        rect_x2 = x_edges_sorted[i+1]
        rect_w = rect_x2 - rect_x1
        rect_w = round_to_grid_mm(rect_w)
        at_x_boundary = rect_x1 <= eps or rect_x2 >= square_side_mm - eps
        if rect_w < MIN_TIM_DIMENSION_MM and not at_x_boundary:
            continue
        
        for j in range(len(y_edges_sorted) - 1):
            rect_y1 = y_edges_sorted[j]
            rect_y2 = y_edges_sorted[j+1]
            rect_h = rect_y2 - rect_y1
            rect_h = round_to_grid_mm(rect_h)
            at_y_boundary = rect_y1 <= eps or rect_y2 >= square_side_mm - eps
            if rect_h < MIN_TIM_DIMENSION_MM and not at_y_boundary:
                continue
            
            # 筛选纯空白矩形（宽高已对齐网格，避免浮点导致后续重叠）
            if _is_pure_blank_rect(rect_x1, rect_y1, rect_x2, rect_y2, chiplet_list_mm):
                raw_blank_rects.append((rect_x1, rect_y1, rect_w, rect_h))

    # ========== 步骤3：最大化融合所有可合并的矩形 ==========
    merged_rects = _merge_all_possible_rects(raw_blank_rects)

    # ========== 步骤4：生成最终TIM块列表 ==========
    tim_blocks = []
    for idx, (x, y, w, h) in enumerate(merged_rects):
        tim_blocks.append({
            'name': f"T{idx}",
            'x': round_to_grid_mm(x),
            'y': round_to_grid_mm(y),
            'w': round_to_grid_mm(w),
            'h': round_to_grid_mm(h),
            'specific_heat': 4000000,
            'thermal_resistivity': 0.25
        })

    # ========== 步骤5：面积校验 ==========
    tim_total_area = sum(t['w'] * t['h'] for t in tim_blocks)
    chip_total_area = sum(c['w'] * c['h'] for c in chiplet_list_mm)
    frame_total_area = square_side_mm * square_side_mm
    area_diff = abs((tim_total_area + chip_total_area) - frame_total_area)
    
    if area_diff > OVERLAP_TOLERANCE_MM2 * 100:
        print(f"⚠️  面积校验警告：TIM+Chiplet总面积与外接框偏差 {area_diff:.6f} mm²")
    else:
        print(f"✅ 面积校验通过：TIM+Chiplet总面积 = 外接框面积（{frame_total_area:.6f} mm²）")
    
    print(f"✅ TIM生成完成：原始空白矩形{len(raw_blank_rects)}个 → 融合后{len(tim_blocks)}个（数量最少）")
    return tim_blocks

def _rects_overlap(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    """轴对齐矩形是否重叠"""
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

def _can_merge_h(a, b):
    """水平相邻可合并：同 y、同 h，且左右相邻"""
    (ax, ay, aw, ah), (bx, by, bw, bh) = a, b
    if abs(ay - by) >= 1e-9 or abs(ah - bh) >= 1e-9:
        return False
    if abs((ax + aw) - bx) < 1e-9:
        return (ax, ay, aw + bw, ah)
    if abs((bx + bw) - ax) < 1e-9:
        return (bx, by, aw + bw, ah)
    return False

def _can_merge_v(a, b):
    """垂直相邻可合并：同 x、同 w，且上下相邻"""
    (ax, ay, aw, ah), (bx, by, bw, bh) = a, b
    if abs(ax - bx) >= 1e-9 or abs(aw - bw) >= 1e-9:
        return False
    if abs((ay + ah) - by) < 1e-9:
        return (ax, ay, aw, ah + bh)
    if abs((by + bh) - ay) < 1e-9:
        return (ax, by, aw, ah + bh)
    return False


def copy_config_templates(template_dir, target_dir, json_basename=None, bbox_longest_side=None):
    """复制配置模板并修改-s_sink/-s_spreader（>=芯片边长，避免 inordinate floorplan size）、floorplan引用"""
    files = ["example.config", "example.lcf", "example.materials"]
    for fname in files:
        src = os.path.join(template_dir, fname)
        dst = os.path.join(target_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            os.chmod(dst, 0o755)
            print(f"  ✅ 复制模板：{fname} -> {target_dir}")
    # 热沉/均热板边长必须 >= 芯片外接框边长，否则 HotSpot 报 "inordinate floorplan size!"
    if bbox_longest_side is not None:
        val = math.ceil(bbox_longest_side * 1000) / 1000 + 0.001
        val_str = f"{val:.3f}"
        config_path = os.path.join(target_dir, "example.config")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            content = re.sub(r"(-s_sink\s+)[\d.e+-]+", rf"\g<1>{val_str}", content)
            content = re.sub(r"(-s_spreader\s+)[\d.e+-]+", rf"\g<1>{val_str}", content)
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ 修改example.config：-s_sink/-s_spreader = {val_str}m（>=芯片边长）")
    # 修改floorplan文件引用
    if json_basename is not None:
        lcf_path = os.path.join(target_dir, "example.lcf")
        if os.path.exists(lcf_path):
            with open(lcf_path, 'r', encoding='utf-8') as f:
                content = f.read()
            content = content.replace("floorplan0.flp", f"{json_basename}_sub.flp")
            content = content.replace("floorplan2.flp", f"{json_basename}.flp")
            content = content.replace("floorplan1.flp", f"{json_basename}_C4.flp")
            with open(lcf_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ 修改example.lcf：floorplan引用替换为{json_basename}")

def blocks_mm_to_m(blocks):
    """将块列表从 mm 转为 m（用于 FLP 输出）"""
    out = []
    for b in blocks:
        o = b.copy()
        o['x'] = mm_to_m(o['x'])
        o['y'] = mm_to_m(o['y'])
        o['w'] = mm_to_m(o['w'])
        o['h'] = mm_to_m(o['h'])
        out.append(o)
    return out

def generate_sub_flp_file(square_side_m, json_basename, output_dir):
    """生成外接框简化FLP文件（Unit0，正方形边长 m）"""
    output_path = os.path.join(output_dir, f"system_sub.flp")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# HotSpot floorplan file (sub: outer bounding box)\n")
        f.write("# Format: <unit> <width> <height> <x> <y>\n")
        f.write("# Unit: meters\n\n")
        f.write(f"Unit0\t{square_side_m:.6f}\t{square_side_m:.6f}\t0.000000\t0.000000\n")
    print(f"✅ 生成Sub FLP：{output_path} (边长={square_side_m:.6f}m)")

def generate_C4_flp_file(square_side_m, json_basename, output_dir):
    """生成外接框简化FLP文件（Unit0，正方形边长 m）"""
    output_path = os.path.join(output_dir, f"{json_basename}_C4.flp")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# HotSpot floorplan file (sub: outer bounding box)\n")
        f.write("# Format: <unit> <width> <height> <x> <y>\n")
        f.write("# Unit: meters\n\n")
        f.write(f"Unit0\t{square_side_m:.6f}\t{square_side_m:.6f}\t0.000000\t0.000000\t2523888.888888889\t0.014130946773433819\n")
    print(f"✅ 生成C4 FLP：{output_path} (边长={square_side_m:.6f}m)")

def generate_flp_file(chiplet_list_m, tim_list_m, output_flp_path):
    """生成最终FLP文件（chiplet+TIM，单位 m，TIM含热学属性）"""
    with open(output_flp_path, 'w', encoding='utf-8') as f:
        f.write("# HotSpot floorplan file (chiplet + TIM)\n")
        f.write("# Format: <unit> <w> <h> <x> <y> [specific-heat] [thermal-resistivity]\n")
        f.write("# Unit: meters\n\n")
        for chip in chiplet_list_m:
            # Chiplet: 使用硅的默认热参数
            f.write(
                f"{chip['name']} {chip['w']:.6f} {chip['h']:.6f} "
                f"{chip['x']:.6f} {chip['y']:.6f} "
                f"\n"
            )
        for tim in tim_list_m:
            f.write(
                f"{tim['name']} {tim['w']:.6f} {tim['h']:.6f} "
                f"{tim['x']:.6f} {tim['y']:.6f} "
                f"{tim['specific_heat']} {tim['thermal_resistivity']}\n"
            )
    print(f"✅ 生成最终FLP：{output_flp_path}")
    print(f"📦 包含{len(chiplet_list_m)}个Chiplet + {len(tim_list_m)}个TIM")

def main(
    json_path,
    output_dir=None,
    output_flp_path=None,
    output_ptrace_path=None,
    fixed_system_names=False,
):
    """
    主函数：按新规范生成 FLP
    1. 基础数据提取+校验
    2. 新外接框（最长边正方形，(0,0)左下角）
    3. Chiplet 网格对齐+平移居中（0.01mm）
    4. TIM 切割+融合
    5. mm→m 转换，输出文件
    """
    json_basename = os.path.splitext(os.path.basename(json_path))[0]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.normpath(os.path.join(script_dir, "..", "config"))
    output_config_dir = os.path.abspath(os.path.normpath(
        output_dir if output_dir else os.path.join(config_dir, f"{json_basename}_config")
    ))
    os.makedirs(output_config_dir, exist_ok=True)

    if output_flp_path is None:
        output_flp_path = os.path.join(
            output_config_dir,
            "system.flp" if fixed_system_names else f"{json_basename}.flp",
        )
    if output_ptrace_path is None:
        output_ptrace_path = os.path.join(
            output_config_dir,
            "system.ptrace" if fixed_system_names else f"{json_basename}.ptrace",
        )

    # 1. 基础数据提取+校验
    chiplets = load_json_layout(json_path)
    print(f"\n📦 加载 {len(chiplets)} 个 Chiplet")

    # 2+3. 新外接框 + Chiplet 网格对齐+平移居中
    chiplet_list_mm, square_side_mm = build_layout(chiplets)
    print(f"📏 新正方形外接框边长（mm）：{square_side_mm}")
    check_blocks_overlap(chiplet_list_mm)

    # 4. TIM 切割+融合
    tim_list_mm = get_tim_blocks(chiplet_list_mm, square_side_mm)
    all_blocks_mm = chiplet_list_mm + tim_list_mm
    check_blocks_overlap(all_blocks_mm)

    # 5. mm→m 转换
    chiplet_list_m = blocks_mm_to_m(chiplet_list_mm)
    tim_list_m = blocks_mm_to_m(tim_list_mm)
    square_side_m = mm_to_m(square_side_mm)

    # 复制配置模板
    template_dir = os.path.join(config_dir, "template_config")
    if os.path.isdir(template_dir):
        print(f"\n📁 处理配置模板：{template_dir}")
        copy_config_templates(template_dir, output_config_dir, json_basename, square_side_m)
    else:
        print(f"\n⚠️  模板目录不存在：{template_dir}，跳过配置复制")

    # 复制 layer 文件（Chiplet.lcf）到输出目录，便于后续 HotSpot 直接使用
    lcf_src = os.path.join(script_dir, "Chiplet.lcf")
    lcf_dst = os.path.join(output_config_dir, "Chiplet.lcf")
    if os.path.isfile(lcf_src):
        shutil.copyfile(lcf_src, lcf_dst)
    else:
        print(f"\n⚠️  layer 文件不存在：{lcf_src}，跳过复制")

    # 生成输出文件
    print(f"\n📄 生成输出文件...")
    generate_flp_file(chiplet_list_m, tim_list_m, output_flp_path)
    generate_ptrace_file(chiplet_list_m, tim_list_m, json_path, output_ptrace_path)
    generate_sub_flp_file(square_side_m, json_basename, output_config_dir)
    # generate_C4_flp_file(square_side_m, json_basename, output_config_dir)

    # 全部处理完毕后，检查 chiplet 之间、TIM 块之间是否有网格重叠
    check_no_grid_overlap(chiplet_list_m, tim_list_m, unit_m=True)

    print(f"\n🎉 所有处理完成！输出目录：{output_config_dir}")


def _is_nonneg_int_string(s: str) -> bool:
    return bool(re.fullmatch(r"\d+", str(s)))


def batch_generate(
    input_dir,
    output_root,
    start_id=None,
    end_id=None,
):
    input_dir = os.path.abspath(os.path.normpath(input_dir))
    output_root = os.path.abspath(os.path.normpath(output_root))
    os.makedirs(output_root, exist_ok=True)

    if start_id is not None and end_id is not None and start_id > end_id:
        raise ValueError("start_id 不能大于 end_id")

    pattern = os.path.join(input_dir, "system_*.json")
    json_paths = sorted([p for p in glob.glob(pattern) if os.path.isfile(p)])
    if not json_paths:
        raise FileNotFoundError(f"未找到输入文件: {pattern}")

    processed = 0
    for json_path in json_paths:
        base = os.path.splitext(os.path.basename(json_path))[0]  # system_i
        if not base.startswith("system_"):
            continue
        id_str = base[len("system_") :]
        if not _is_nonneg_int_string(id_str):
            continue
        i = int(id_str)
        if start_id is not None and end_id is not None:
            if i < start_id or i > end_id:
                continue

        out_dir = os.path.join(output_root, f"{base}_config")
        os.makedirs(out_dir, exist_ok=True)
        main(
            json_path=json_path,
            output_dir=out_dir,
            output_flp_path=os.path.join(out_dir, "system.flp"),
            output_ptrace_path=os.path.join(out_dir, "system.ptrace"),
            fixed_system_names=True,
        )
        processed += 1

    print(f"\n✅ batch_generate 完成：共处理 {processed} 个 JSON，输出根目录：{output_root}")

if __name__ == "__main__":
    import argparse
    import glob
    parser = argparse.ArgumentParser(description='Chiplet布局生成HotSpot文件（缩减长宽消重叠，不偏移坐标）')
    parser.add_argument('--json_path', type=str, default=None, help='输入的JSON布局文件路径（mm）')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录（会在此目录下生成flp/ptrace并复制模板配置）')
    parser.add_argument('--output_flp', type=str, default=None, help='输出.flp文件路径，默认自动生成')
    parser.add_argument('--output_ptrace', type=str, default=None, help='输出.ptrace文件路径，默认自动生成')

    # 批处理模式（实现 gen.sh 的功能）
    parser.add_argument('--input_dir', type=str, default=None, help='批处理输入目录（匹配 system_*.json）')
    parser.add_argument('--output_root', type=str, default=None, help='批处理输出根目录（生成 system_i_config 子目录）')
    parser.add_argument('--start_id', type=int, default=None, help='批处理起始编号（包含）')
    parser.add_argument('--end_id', type=int, default=None, help='批处理结束编号（包含）')
    args = parser.parse_args()

    if args.input_dir is not None or args.output_root is not None:
        if not args.input_dir or not args.output_root:
            raise SystemExit("批处理模式需要同时提供 --input_dir 和 --output_root")
        if (args.start_id is None) != (args.end_id is None):
            raise SystemExit("批处理范围过滤需要同时提供 --start_id 和 --end_id")
        batch_generate(
            input_dir=args.input_dir,
            output_root=args.output_root,
            start_id=args.start_id,
            end_id=args.end_id,
        )
    else:
        if not args.json_path:
            raise SystemExit("单文件模式需要提供 --json_path，或使用批处理模式 --input_dir/--output_root")
        main(args.json_path, args.output_dir, args.output_flp, args.output_ptrace)

'''
批处理模式
python3 /root/workspace/flow_GCN/Dataset/dataset/hotspot/gen_flp_trace.py \
  --input_dir /root/workspace/flow_GCN/Dataset/dataset/output/placement \
  --output_root /root/workspace/flow_GCN/Dataset/dataset/output/thermal/hotspot_config \
  --start_id 1 --end_id 1000


单文件处理
python3 /root/workspace/flow_GCN/Dataset/dataset/hotspot/gen_flp_trace.py \
  --json_path /root/workspace/flow_GCN/Dataset/dataset/output/placement/system_1.json \
  --output_dir /root/workspace/flow_GCN/Dataset/dataset/output/thermal/hotspot_config/system_1_config
'''