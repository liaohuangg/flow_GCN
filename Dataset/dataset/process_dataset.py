"""
Chiplet 数据集处理脚本
- 读取 input_test/ 和 output/placement/ 下的 system_i.json
- 清洗字段后配对合并为统一数据集结构
- 保存为 pickle 文件到 pickle_dataset/ 目录
"""

import json
import pickle
import os
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input_test")
PLACEMENT_DIR = os.path.join(BASE_DIR, "output", "placement")
OUTPUT_DIR = os.path.join(BASE_DIR, "pickle_dataset")

# connections 中需要删除的字段
CONN_REMOVE_KEYS = {"EMIBType", "EMIB_length", "EMIB_max_width", "EMIB_bump_width"}

# placement JSON 中需要删除的顶层字段
PLACEMENT_REMOVE_KEYS = {"connections", "wirelength", "area", "aspect_ratio"}


def extract_system_id(filename):
    """从文件名中提取 system 编号，如 system_12.json -> 12"""
    m = re.match(r"system_(\d+)\.json$", filename)
    return int(m.group(1)) if m else None


def clean_connections(connections):
    """删除 connections 列表中每个元素的 EMIB 相关字段，只保留 node1/node2/wireCount"""
    cleaned = []
    for conn in connections:
        cleaned.append({k: v for k, v in conn.items() if k not in CONN_REMOVE_KEYS})
    return cleaned


def clean_placement(placement_data):
    """只保留 chiplets 数组作为 placement 信息，删除 connections/wirelength/area/aspect_ratio"""
    return placement_data.get("chiplets", [])


def build_dataset():
    """遍历目录、配对文件、清洗字段、构建统一数据集"""
    # 收集 input_test 下所有 system 编号
    input_files = {
        extract_system_id(f): f
        for f in os.listdir(INPUT_DIR)
        if f.endswith(".json") and extract_system_id(f) is not None
    }
    # 收集 placement 下所有 system 编号
    placement_files = {
        extract_system_id(f): f
        for f in os.listdir(PLACEMENT_DIR)
        if f.endswith(".json") and extract_system_id(f) is not None
    }

    # 取两个目录的交集，确保配对
    common_ids = sorted(set(input_files.keys()) & set(placement_files.keys()))
    logger.info(
        "input_test: %d 个文件, placement: %d 个文件, 配对: %d 个",
        len(input_files), len(placement_files), len(common_ids),
    )

    dataset = {}
    for sys_id in common_ids:
        input_path = os.path.join(INPUT_DIR, input_files[sys_id])
        placement_path = os.path.join(PLACEMENT_DIR, placement_files[sys_id])

        with open(input_path, "r") as f:
            input_data = json.load(f)
        with open(placement_path, "r") as f:
            placement_data = json.load(f)

        record = {
            "system_id": f"system_{sys_id}",
            "chiplets": input_data.get("chiplets", []),
            "connections": clean_connections(input_data.get("connections", [])),
            "placement": clean_placement(placement_data),
        }
        dataset[f"system_{sys_id}"] = record

    logger.info("数据集构建完成，共 %d 条记录", len(dataset))
    return dataset


def save_dataset(dataset):
    """将数据集保存为 pickle 文件，同时保存一份可读的 JSON 用于检查"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 保存 pickle
    pickle_path = os.path.join(OUTPUT_DIR, "chiplet_dataset.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(dataset, f)
    logger.info("Pickle 已保存: %s (%.2f MB)", pickle_path, os.path.getsize(pickle_path) / 1024 / 1024)

    # 保存一份 JSON 方便人工检查
    json_path = os.path.join(OUTPUT_DIR, "chiplet_dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    logger.info("JSON 已保存: %s", json_path)


if __name__ == "__main__":
    dataset = build_dataset()
    save_dataset(dataset)


'''
数据集中一共4154条数据, 系统编号从 system_1 到 system_4154。
每条数据包含以下字段：
{
  "system_i": {
    "system_id": "system_i",          // 字符串，系统唯一编号
    "chiplets": [                     // 数组，所有 chiplet 基本物理信息
      {
        "name": "str",                // chiplet 名称（A/B/C/D/E）
        "width": float,               // 宽度
        "height": float,              // 高度
        "power": int/float            // 功耗
      }
    ],
    "connections": [                  // 数组，chiplet 之间的连接关系
      {
        "node1": "str",               // 连接的第一个 chiplet
        "node2": "str",               // 连接的第二个 chiplet
        "wireCount": int              // 连线数量（作为边权重）
      }
    ],
    "placement": [                    // 数组，chiplet 布局坐标信息
      {
        "name": "str",                // chiplet 名称
        "x-position": float,           // 布局 x 坐标
        "y-position": float,           // 布局 y 坐标
        "width": float,               // 宽度
        "height": float,              // 高度
        "rotation": int,              // 旋转角度 0/1
        "power": float                // 功耗
      }
    ]
  }
}

'''