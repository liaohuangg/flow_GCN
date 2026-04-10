import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_chiplets_json(
    json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load the raw chiplet description JSON.

    Parameters
    ----------
    json_path:
        Path to the `chiplets.json` file. If None, uses relative path from src directory.

    Returns
    -------
    dict
        The parsed JSON object. Top-level keys are chiplet names.
    """
    if json_path is None:
        # 从 src 目录的相对路径: ../../benchmark/dummy-chiplet-input/chiplet_input/chiplets.json
        current_file = Path(__file__)
        src_dir = current_file.parent
        project_root = src_dir.parent
        json_path = project_root / "benchmark" / "dummy-chiplet-input" / "chiplet_input" / "chiplets.json"
        # 如果默认路径不存在，抛出错误提示用户提供路径
        if not json_path.exists():
            raise FileNotFoundError(
                f"默认的 chiplets.json 文件不存在: {json_path}\n"
                f"请使用 input_json_path 参数指定正确的 JSON 文件路径。"
            )
    
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    return data


def build_chiplet_table(chiplets: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a simple table from the chiplet JSON.

    Each row in the table is a dict with the following keys:

    - ``name``: chiplet name (top-level key in the JSON)
    - ``dimensions``: the raw ``dimensions`` object for this chiplet
    - ``phys``: the raw ``phys`` list for this chiplet
    - ``power``: the ``power`` value for this chiplet

    Parameters
    ----------
    chiplets:
        Parsed JSON dict returned by :func:`load_chiplets_json`.

    Returns
    -------
    list of dict
        A list of rows; each row has keys ``name``, ``dimensions``, ``phys``,
        and ``power``.
    """

    table: List[Dict[str, Any]] = []

    for name, info in chiplets.items():
        row = {
            "name": name,
            "dimensions": info.get("dimensions", {}),
            "phys": info.get("phys", []),
            "power": info.get("power", None),
        }
        table.append(row)

    return table


def pretty_print_table(table: List[Dict[str, Any]]) -> None:
    """
    Pretty-print the chiplet table to the console.

    This is just for quick inspection / debug.
    """

    from pprint import pprint

    for row in table:
        pprint(row)


if __name__ == "__main__":
    # Quick manual test: load the default JSON and print the table.
    data = load_chiplets_json()
    tbl = build_chiplet_table(data)
    pretty_print_table(tbl)


