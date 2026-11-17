"""GRAGのActorと対照エンコーダーの構築。"""

import math

import torch
from torch import nn


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def build_networks(
    observation_shape: tuple[int, ...], hidden_size: int, num_actions: int, *, seed: int
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """外部の乱数状態を保ち、指定seedで3つのネットワークを構築する。"""
    # 初期化にseedを反映し、外部のグローバル乱数状態は元に戻す。
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        actor = nn.Sequential(
            nn.Flatten(start_dim=-3),
            layer_init(nn.Linear(math.prod(observation_shape), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, num_actions), std=0.01)
        )

        # CPPOのMLP構成: 512幅×4層、LayerNorm、Swish、64次元出力。
        # PRTSの phi(s, a): 目標チャネルを除いた障害物・現在位置を
        # 平坦化し、行動one-hotを連結したベクトルを入力する。
        sa_encoder = nn.Sequential(
            nn.Linear(2 * math.prod(observation_shape[-2:]) + num_actions, 512),
            nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 64),
        )

        # PRTSの psi(l): 外部で抽出した言語特徴ベクトルを入力する。
        # Ruri v3 70mの384次元の言語特徴を入力する。
        goal_encoder = nn.Sequential(
            nn.Linear(384, 512),
            nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 64),
        )

    return actor, sa_encoder, goal_encoder
