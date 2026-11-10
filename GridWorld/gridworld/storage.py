"""File format and persistence shared by the three independent applications."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MAPS = DATA / "maps"
SETTINGS = DATA / "settings.json"  # 以前の設定ファイル。読み込みのみ互換対応する。
TRAINING_SETTINGS = DATA / "training_settings.json"
WORLD = DATA / "world.json"
RESULT = DATA / "latest_result.json"


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSONの最上位はオブジェクトにしてください。")
    return value


def validate_map(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("マップの形式が正しくありません。")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 100:
        raise ValueError("マップ名は1〜100文字で指定してください。")
    rows, cols = value.get("rows"), value.get("cols")
    if type(rows) is not int or type(cols) is not int or not (2 <= rows <= 50 and 2 <= cols <= 50):
        raise ValueError("縦・横はそれぞれ2〜50にしてください。")

    def cell(raw: Any, label: str) -> list[int]:
        if not isinstance(raw, list) or len(raw) != 2 or any(type(n) is not int for n in raw):
            raise ValueError(f"{label}は[行, 列]で指定してください。")
        if not (0 <= raw[0] < rows and 0 <= raw[1] < cols):
            raise ValueError(f"{label}がマップの範囲外です。")
        return raw

    start = cell(value.get("start"), "スタート")
    goal = cell(value.get("goal"), "ゴール")
    if start == goal:
        raise ValueError("スタートとゴールは別のマスにしてください。")
    raw_obstacles = value.get("obstacles", [])
    if not isinstance(raw_obstacles, list):
        raise ValueError("障害物の形式が正しくありません。")
    obstacles = [cell(item, "障害物") for item in raw_obstacles]
    unique = {(r, c) for r, c in obstacles}
    if len(unique) != len(obstacles) or tuple(start) in unique or tuple(goal) in unique:
        raise ValueError("障害物に重複、スタートまたはゴールとの重なりがあります。")
    return {"name": name.strip(), "rows": rows, "cols": cols, "start": start,
            "goal": goal, "obstacles": [list(p) for p in sorted(unique)]}


def map_path(name: str) -> Path:
    key = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return MAPS / f"{key}.json"


def save_map(value: dict[str, Any]) -> dict[str, Any]:
    clean = validate_map(value)
    write_json(map_path(clean["name"]), clean)
    return clean


def list_maps() -> list[dict[str, Any]]:
    if not MAPS.exists():
        return []
    maps = []
    for path in MAPS.glob("*.json"):
        try:
            item = validate_map(read_json(path))
            maps.append({"name": item["name"], "rows": item["rows"], "cols": item["cols"]})
        except (ValueError, OSError, json.JSONDecodeError):
            continue
    return sorted(maps, key=lambda item: item["name"].casefold())


DEFAULT_SETTINGS = {
    "episodes": 1000, "max_steps": 200, "learning_rate": 0.1,
    "discount_factor": 0.95, "epsilon": 1.0, "epsilon_decay": 0.995,
    "min_epsilon": 0.05, "seed": 42,
}


def validate_settings(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("設定の形式が正しくありません。")
    clean = {}
    for key in ("episodes", "max_steps"):
        raw = value.get(key)
        if type(raw) is not int or not (1 <= raw <= 1_000_000):
            raise ValueError(f"{key}は1〜1,000,000の整数にしてください。")
        clean[key] = raw
    for key in ("learning_rate", "discount_factor", "epsilon", "epsilon_decay", "min_epsilon"):
        raw = value.get(key)
        if type(raw) not in (int, float) or not (0 <= raw <= 1):
            raise ValueError(f"{key}は0〜1にしてください。")
        clean[key] = float(raw)
    if clean["min_epsilon"] > clean["epsilon"]:
        raise ValueError("探索率の下限は初期探索率以下にしてください。")
    raw_seed = value.get("seed")
    if raw_seed is not None and (type(raw_seed) is not int or not (0 <= raw_seed <= 2**32 - 1)):
        raise ValueError("乱数シードは空欄または0〜2^32-1の整数にしてください。")
    clean["seed"] = raw_seed
    return clean


def load_settings() -> dict[str, Any]:
    training_path = SETTINGS.parent / "training_settings.json"
    source = training_path if training_path.exists() else SETTINGS
    raw = read_json(source) if source.exists() else {}
    clean = validate_settings({**DEFAULT_SETTINGS, **raw})
    clean.update({key: value for key, value in raw.items() if key.startswith(("ppo_", "crl_"))})
    return clean
