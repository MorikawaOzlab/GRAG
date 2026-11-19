"""CleanRL型Actor-Critic。環境オブジェクトを受け取らず、入力・行動の次元だけを使う。"""

import math

import torch
from torch import nn
from torch.distributions import Categorical


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        if any(type(size) is not int or size < 1
               for size in (observation_size, action_size, hidden_size)):
            raise ValueError("入力・行動・隠れ層の次元は1以上の整数にしてください。")
        # 初期化順は元のCleanRLコードと同じ。登録順は既存の保存モデルと合わせる。
        critic = nn.Sequential(
            layer_init(nn.Linear(observation_size, hidden_size)), nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0))
        actor = nn.Sequential(
            layer_init(nn.Linear(observation_size, hidden_size)), nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_size), std=0.01))
        self.actor = actor
        self.critic = critic

    def forward(self, states):
        return self.actor(states), self.critic(states).squeeze(-1)

    def get_value(self, states):
        return self.critic(states)

    def get_action_and_value(self, states, action=None, *, generator=None):
        distribution = Categorical(logits=self.actor(states))
        if action is None:
            action = (distribution.sample() if generator is None else
                      torch.multinomial(distribution.probs, 1, generator=generator).squeeze(-1))
        return action, distribution.log_prob(action), distribution.entropy(), self.critic(states)
