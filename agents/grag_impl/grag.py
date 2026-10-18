"""開発中の強化学習システムGRAG。

実装後は既存の学習処理に次のように渡せます。

    from agents.grag import GRAG
    from gridworld.training import train

    result = train(map_data, settings, agent_type=GRAG)
"""

import math

import torch
from torch import nn


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

ACTIONS = ("up", "down", "left", "right")

class GRAG:
    """GRAGの観測変換・行動選択・学習を扱うエージェント。"""

    HIDDEN_SIZE = 64

    def __init__(self, settings: dict):
        """学習パラメータを受け取る。モデルや内部状態の初期化はここに追加する。"""
        self.settings = settings
        self.actor = None
        self.critic = None

    def initialize_networks(self, observation:dict):
        if (self.actor is not None) or (self.critic is not None):
            return
        
        state = self.observation_to_tensor(observation)
        observation_shape = state.shape

        self.actor = nn.Sequential(
            nn.Flatten(start_dim=-3),
            layer_init(nn.Linear(math.prod(observation_shape), self.HIDDEN_SIZE)),
            nn.Tanh(),
            layer_init(nn.Linear(self.HIDDEN_SIZE, self.HIDDEN_SIZE)),
            nn.Tanh(),
            layer_init(nn.Linear(self.HIDDEN_SIZE, len(ACTIONS)), std=0.01)
        )

        self.critic = nn.Sequential(
            nn.Flatten(start_dim=-3),
            layer_init(nn.Linear(math.prod(observation_shape), self.HIDDEN_SIZE)),
            nn.Tanh(),
            layer_init(nn.Linear(self.HIDDEN_SIZE, self.HIDDEN_SIZE)),
            nn.Tanh(),
            layer_init(nn.Linear(self.HIDDEN_SIZE, 1), std=1.0)
        )
        

    def choose_action(self, observation: dict, greedy: bool = False) -> str:
        """現在の観測から行動を1つ返す。

        observation は name, rows, cols, start, goal, obstacles, position
        を含む辞書。座標は [行, 列] で、左上が [0, 0]。
        戻り値は ACTIONS の値（up, down, left, right）のいずれか。
        greedy=True は結果再生時に渡されるため、探索や内部状態の更新を
        行わず、現在の方策で行動を選ぶ。
        """
        # TODO: 観測をもとにAIが選んだ行動を返す。

        self.initialize_networks(observation)
        state = self.observation_to_tensor(observation)

        with torch.no_grad():
            logits = self.actor(state)

            if greedy:
                action = torch.argmax(logits).item()
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()

            return ACTIONS[action.item()]


    def action_probabilities(self, observation: dict) -> dict[str, float]:
        """任意の公開機能。観測に対する4方向の確率（合計1）を返す。

        学習状態、乱数、入力観測を変更しない。環境や画面を呼び出さない。
        表示に対応しないAIはこのメソッドを省略するかNotImplementedErrorを返す。
        """        
        if self.actor is None:
            raise NotImplementedError("ネットワークが未初期化です。")

        state = self.observation_to_tensor(observation)

        with torch.no_grad():
            logits = self.actor(state)
            probabilities = torch.softmax(logits, dim=-1)

        return dict(zip(ACTIONS, probabilities.tolist()))

    def learn(self, observation: dict, action: str, reward: float,
              next_observation: dict, reached_goal: bool) -> None:
        """1回の行動の結果を受け取り、必要に応じて学習する。

        reward はゴール到達時に1、それ以外は0。
        reached_goal はゴール到達を表し、最大行動回数による終了は含まない。
        学習しないAIなら、このメソッドは空のままでよい。
        """
        # TODO: 必要なら学習や経験の記録を実装する。
        pass
    

    def end_episode(self) -> None:
        """ゴール到達または最大行動回数で試行が終わるたびに呼ばれる。"""
        # TODO: 必要なら試行ごとの後処理や内部状態のリセットを実装する。
        pass
    
    def observation_to_tensor(self, observation: dict):
        """観測を[障害物・ゴール・現在位置, 行, 列]のfloat32テンソルに変換する。"""
        import torch

        tensor = torch.zeros((3, observation["rows"], observation["cols"]),
                             dtype=torch.float32)
        for row, col in observation["obstacles"]:
            tensor[0, row, col] = 1.0
        goal_row, goal_col = observation["goal"]
        tensor[1, goal_row, goal_col] = 1.0
        row, col = observation["position"]
        tensor[2, row, col] = 1.0
        
        return tensor

    def export(self) -> dict:
        """保存用のJSONに変換可能な辞書を返す。

        現在の学習処理では結果JSONの q_table に格納される。
        保存する学習データがなければ空の辞書を返す。
        """
        # TODO: 必要なら保存する学習データを返す。
        return {}
