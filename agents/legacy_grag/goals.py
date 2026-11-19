"""自然言語目標と、環境で検証できる達成条件。"""

from dataclasses import dataclass


def _coordinate(value, label: str) -> tuple[int, int]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(item) is not int or item < 0 for item in value)):
        raise ValueError(f"{label}は非負の整数の[行, 列]で指定してください。")
    return tuple(value)


@dataclass(frozen=True)
class LanguageGoal:
    """目標文と判定用座標の組。文の意味解析は行わない。"""

    text: str
    target: tuple[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("目標文は空でない文字列で指定してください。")
        # 元の観測のリストが変更されても、目標座標は変わらない。
        object.__setattr__(self, "target", _coordinate(self.target, "目標座標"))

    @classmethod
    def from_observation(cls, observation: dict,
                         text: str | None = None) -> "LanguageGoal":
        """既存グリッドワールドの goal を判定用座標として使う。"""
        target = _coordinate(observation.get("goal"), "目標座標")
        rows, cols = observation.get("rows"), observation.get("cols")
        if (type(rows) is not int or rows <= 0
                or type(cols) is not int or cols <= 0):
            raise ValueError("rowsとcolsは正の整数で指定してください。")
        if not (target[0] < rows and target[1] < cols):
            raise ValueError("目標座標がマップの範囲外です。")
        if text is None:
            text = f"マス [{target[0]}, {target[1]}] に到達する"
        return cls(text=text, target=target)

    def is_reached(self, observation: dict) -> bool:
        """現在位置と目標座標の一致を判定する。観測を変更しない。"""
        position = _coordinate(observation.get("position"), "現在位置")
        return position == self.target
