"""GridWorld の観測と GRAG のネットワークを接続する。"""

import torch


class GridWorldAdapter:
    """マップごとの入力次元と観測のテンソル化を提供する。"""

    def __init__(self, map_data: dict):
        self.rows = map_data["rows"]
        self.cols = map_data["cols"]
        if any(type(n) is not int or n <= 0 for n in (self.rows, self.cols)):
            raise ValueError("rowsとcolsは正の整数にしてください。")

    def network_dimensions(self, action_dim: int) -> tuple[int, int, int]:
        """状態・ゴール・行動の次元をこの順に返す。"""
        return 3 * self.rows * self.cols, 2, action_dim

    def encode_observation(self, observation: dict) -> torch.Tensor:
        """観測を障害物・ゴール・現在位置の3チャネルに変換する。"""
        if (observation["rows"], observation["cols"]) != (self.rows, self.cols):
            raise ValueError("観測のマップサイズがアダプターと一致しません。")
        state = torch.zeros((3, self.rows, self.cols), dtype=torch.float32)
        for row, col in observation["obstacles"]:
            state[0, row, col] = 1
        state[1, *observation["goal"]] = 1
        state[2, *observation["position"]] = 1
        return state

    def encode_achieved_goal(self, observation: dict) -> torch.Tensor:
        """到達した位置を対照学習用のゴールとして返す。"""
        return torch.tensor(observation["position"], dtype=torch.float32)

    def encode_goal(self, observation: dict) -> torch.Tensor:
        """観測で指定された目標位置を返す。"""
        return torch.tensor(observation["goal"], dtype=torch.float32)

    def relabel_goal(self, observation: dict, goal: torch.Tensor) -> dict:
        """Actorに渡す観測の目標をHERで選んだ位置に差し替える。"""
        return {**observation, "goal": [int(coordinate) for coordinate in goal]}
