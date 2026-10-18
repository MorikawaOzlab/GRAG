"""開発中の強化学習システムGRAG。

実装後は既存の学習処理に次のように渡せます。

    from agents.grag import GRAG
    from gridworld.training import train

    result = train(map_data, settings, agent_type=GRAG)

    Goal-Reachabilioty-Aware Gating
"""

import math

import torch
from torch import nn

from dataclasses import dataclass

@dataclass
class Transition:
    state: torch.Tensor
    action: int
    old_log_prob: float
    old_value: float
    reward: float
    next_value: float
    reached_goal: bool
    episode_end: bool = False

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
        self.optimizer = None

        # memory
        self.memory: list[Transition] = []

        # hyper parameters
        self.gamma = settings.get("ppo_gamma", settings.get(
            "gamma", settings.get("discount_factor", 0.99)))
        self.gae_lambda = settings.get("ppo_gae_lambda", settings.get("gae_lambda", 0.95))
        self.clip_coef = settings.get("ppo_clip_range", settings.get("clip_coef", 0.2))
        self.ent_coef = settings.get("ppo_entropy_coef", settings.get("ent_coef", 0.01))
        self.vf_coef = settings.get("ppo_value_coef", settings.get("vf_coef", 0.5))

        # learning_rateは画面のQ学習用設定なので、PPOには使わない。
        self.learning_rate = settings.get("ppo_learning_rate", 2.5e-4)
        self.hidden_size = settings.get("ppo_hidden_size", self.HIDDEN_SIZE)
        self.update_epochs = settings.get("ppo_epochs", settings.get("update_epochs", 4))
        self.minibatch_size = settings.get("ppo_minibatch_size", settings.get("minibatch_size", 64))
        self.rollout_steps = settings.get("ppo_rollout_steps", settings.get("rollout_steps", 128))

        seed = settings.get("seed")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("seedは0〜2^32-1の整数またはNoneにしてください。")
        self._generator = torch.Generator(device="cpu")
        self.seed = self._generator.seed() if seed is None else seed
        self._generator.manual_seed(self.seed)

    def initialize_networks(self, observation:dict):
        if (self.actor is not None) or (self.critic is not None):
            return
        
        state = self.observation_to_tensor(observation)
        observation_shape = state.shape

        # 初期化にseedを反映し、外部のグローバル乱数状態は元に戻す。
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.actor = nn.Sequential(
                nn.Flatten(start_dim=-3),
                layer_init(nn.Linear(math.prod(observation_shape), self.hidden_size)),
                nn.Tanh(),
                layer_init(nn.Linear(self.hidden_size, self.hidden_size)),
                nn.Tanh(),
                layer_init(nn.Linear(self.hidden_size, len(ACTIONS)), std=0.01)
            )

            self.critic = nn.Sequential(
                nn.Flatten(start_dim=-3),
                layer_init(nn.Linear(math.prod(observation_shape), self.hidden_size)),
                nn.Tanh(),
                layer_init(nn.Linear(self.hidden_size, self.hidden_size)),
                nn.Tanh(),
                layer_init(nn.Linear(self.hidden_size, 1), std=1.0)
            )

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) +
            list(self.critic.parameters()),
            lr=self.learning_rate,
            eps=1e-5
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
                action = torch.multinomial(
                    dist.probs, 1, generator=self._generator).squeeze(-1)

            return ACTIONS[action]


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
        # 実行結果を記録
        state = self.observation_to_tensor(observation)
        next_state = self.observation_to_tensor(next_observation)
        action_index = ACTIONS.index(action)

        with torch.no_grad():
            # Actorの出力
            logits = self.actor(state)
            dist = torch.distributions.Categorical(logits=logits)

            # 行動の対数確率
            old_log_prob = dist.log_prob(
                torch.tensor(action_index)
            ).item()

            # 現在の状態価値
            old_value = self.critic(state).item()

            # 次の状態価値
            if reached_goal:
                next_value = 0.0
            else:
                next_value = self.critic(next_state).item()

        transition = Transition(
            state=state,
            action=action_index,
            old_log_prob=old_log_prob,
            old_value=old_value,
            reward=reward,
            next_value=next_value,
            reached_goal=reached_goal
        )

        self.memory.append(transition)

        # rollout_stepごとに学習
        if len(self.memory) >= self.rollout_steps:
            self._update_ppo()

    def end_episode(self) -> None:
        """ゴール到達または最大行動回数で試行が終わるたびに呼ばれる。"""
        if self.memory:
            self.memory[-1].episode_end = True
            self._update_ppo()
    
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

    def _update_ppo(self):
        if not self.memory:
            return

        advantages, returns = self._compute_gae()

        states = torch.stack(
            [tr.state for tr in self.memory]
        )

        actions = torch.tensor(
            [tr.action for tr in self.memory],
            dtype=torch.long
        )

        old_log_probs = torch.tensor(
            [tr.old_log_prob for tr in self.memory],
            dtype=torch.float32
        )

        # Advantageの正規化
        if len(advantages) > 1:
            advantages = (
                (advantages - advantages.mean())
                / (advantages.std(unbiased=False) + 1e-8)
            )

        batch_size = len(self.memory)

        # 同じ経験を複数回使って学習
        for epoch in range(self.update_epochs):

            indices = torch.randperm(batch_size, generator=self._generator)

            # 全データからミニバッチを作成．startは，batchの中にある，mini batchのスタート位置
            for start in range(0, batch_size, self.minibatch_size):
                batch_indices = indices[
                    start:start + self.minibatch_size
                ]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                # 新しい方策で確率を再計算
                logits = self.actor(batch_states) # ミニバッチをNNに入力
                dist = torch.distributions.Categorical(logits=logits)

                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # 新旧方策の確率比
                # 方策を実行したときの対数確立のe乗で割り算
                #  -> これが重要度サンプリングの式
                ratio = torch.exp(
                    new_log_probs - batch_old_log_probs
                )

                # Actor損失
                pg_loss1 = -batch_advantages * ratio
                pg_loss2 = -batch_advantages * torch.clamp(
                    ratio,
                    1.0 - self.clip_coef,
                    1.0 + self.clip_coef
                )

                actor_loss = torch.max(
                    pg_loss1, pg_loss2
                ).mean()

                # Critic損失
                new_values = self.critic(
                    batch_states
                ).squeeze(-1)

                critic_loss = 0.5 * (
                    (new_values - batch_returns) ** 2
                ).mean()

                # 最終損失
                loss = (
                    actor_loss
                    + self.vf_coef * critic_loss
                    - self.ent_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) +
                    list(self.critic.parameters()),
                    max_norm=0.5
                )

                self.optimizer.step()

        self.memory.clear()

    def _compute_gae(self):
        advantages = torch.zeros(len(self.memory))

        gae = 0.0

        for t in reversed(range(len(self.memory))):
            tr = self.memory[t]

            bootstrap = 0.0 if tr.reached_goal else 1.0
            continuation = 0.0 if tr.episode_end or tr.reached_goal else 1.0

            delta = (
                tr.reward
                + self.gamma * bootstrap * tr.next_value
                - tr.old_value
            )

            gae = (
                delta
                + self.gamma * self.gae_lambda * continuation * gae
            )

            advantages[t] = gae

        old_values = torch.tensor(
            [tr.old_value for tr in self.memory],
            dtype=torch.float32
        )

        returns = advantages + old_values

        return advantages, returns
