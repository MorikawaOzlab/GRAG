"""開発中の強化学習システムGRAG。

実装後は既存の学習処理に次のように渡せます。

    from agents.grag import GRAG
    from gridworld.training import train

    result = train(map_data, settings, agent_type=GRAG)

    Goal-Reachabilioty-Aware Gating
"""

import base64
import math
import zlib

import torch
from torch import nn

from dataclasses import dataclass

ACTIONS = ("up", "down", "left", "right")


def _pack_tensor(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return {"tensor_dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "data": base64.b64encode(zlib.compress(tensor.numpy().tobytes())).decode("ascii")}


def _unpack_tensor(data):
    raw = bytearray(zlib.decompress(base64.b64decode(data["data"], validate=True)))
    dtype = getattr(torch, data["tensor_dtype"])
    if not raw:
        return torch.empty(data["shape"], dtype=dtype)
    return torch.frombuffer(raw, dtype=dtype).clone().reshape(data["shape"])

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
    goal: tuple[int, int] | None = None

def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def state_action_encoder(
    state: torch.Tensor, action: int | torch.Tensor, *,
    num_actions: int = len(ACTIONS), state_features: torch.Tensor | None = None
) -> torch.Tensor:
    """ゴール情報を除いた状態特徴と行動one-hotを結合したテンソルを返す。"""
    # sのテンソルを準備
    features = (state[..., [0, 2], :, :].flatten(start_dim=-3)
                if state_features is None else state_features) # 状態sのテンソル = ゴール情報を取り除いたstate (ゴール情報は自然言語が保持しているため)
    action = torch.as_tensor(action, device=features.device).expand(features.shape[:-1]) # actionを(feature -1)次元のテンソルに変換

    if ((action < 0) | (action >= num_actions)).any():
        raise ValueError(f"actionは0以上{num_actions}未満にしてください。")

    # (s, a)のテンソルを準備
    onehot = nn.functional.one_hot(action, num_classes=num_actions) # actionをonehotにする
    dtype = features.dtype if features.is_floating_point() else torch.float32 # featuresの型を保持
    sa_input = torch.cat((features.to(dtype), onehot.to(dtype)), dim=-1) # s, aのテンソルを結合
    return sa_input


class GRAG:
    """GRAGの観測変換・行動選択・学習を扱うエージェント。"""

    HIDDEN_SIZE = 64
    CRL_NUM_ACTIONS = len(ACTIONS)

    def __init__(self, settings: dict):
        """学習パラメータを受け取る。モデルや内部状態の初期化はここに追加する。"""
        self.settings = settings
        self.actor = None
        self.critic = None
        self.optimizer = None

        # memory
        self.memory: list[Transition] = []
        self._crl_updates = 0
        self._crl_episode = []
        self._crl_replay = []
        self._last_advantage_source = "gae"
        for key, default, minimum in (("crl_replay_capacity", 2048, 2),
                                      ("crl_reference_size", 128, 1),
                                      ("crl_warmup_updates", 10, 0)):
            value = settings.get(key, default)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{key}は{minimum}以上の整数にしてください。")

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
        self._map_shape = tuple(observation_shape[-2:])

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

            # CPPOのMLP構成: 512幅×4層、LayerNorm、Swish、64次元出力。
            # PRTSの phi(s, a): 目標チャネルを除いた障害物・現在位置を
            # 平坦化し、行動one-hotを連結したベクトルを入力する。
            self.sa_encoder = nn.Sequential(
                nn.Linear(2 * math.prod(observation_shape[-2:]) + self.CRL_NUM_ACTIONS, 512),
                nn.LayerNorm(512), nn.SiLU(),
                nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU(),
                nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU(),
                nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU(),
                nn.Linear(512, 64),
            )

            # Encode the goal coordinates (row, column).
            # Input dimension is two.
            self.goal_encoder = nn.Sequential(
                nn.Linear(2, 512),
                nn.LayerNorm(512), nn.SiLU(),
                nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU(),
                nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU(),
                nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU(),
                nn.Linear(512, 64),
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

    @torch.no_grad()
    def crl_action_values(self, observation: dict) -> dict[str, float]:
        """保存された参照経験で補正した4行動のQ推定値を返す。"""
        if self.actor is None or not self._crl_replay:
            raise ValueError("対照学習の参照経験がまだありません。")
        key = self._goal_key(observation["goal"])
        refs = self._crl_replay[-self.settings.get("crl_reference_size", 128):]
        mask = torch.tensor([key in item["goals"] for item in refs], dtype=torch.bool)
        if not mask.any():
            raise ValueError("No positive reference for this goal.")
        goal = torch.tensor(observation["goal"], dtype=torch.float32)
        ref_states = torch.stack([item["state"] for item in refs])
        ref_actions = torch.tensor([item["action"] for item in refs])
        ref_scores = self.sa_encoder(state_action_encoder(ref_states, ref_actions)) @ self.goal_encoder(goal)
        steps = torch.tensor([item["goals"].get(key, 0) for item in refs])
        state = self.observation_to_tensor(observation).unsqueeze(0).expand(len(ACTIONS), -1, -1, -1)
        q = self._compute_crl_q(state_action_encoder(state, torch.arange(len(ACTIONS))), goal,
                                reference_log_scores=ref_scores,
                                reference_steps_to_goal=steps, reference_positive_mask=mask)
        return dict(zip(ACTIONS, q.tolist()))

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
            reached_goal=reached_goal,
            goal=tuple(observation["goal"]),
        )

        self.memory.append(transition)
        self._crl_episode.append({"state": state.clone(), "action": action_index,
                                  "next_position": list(next_observation["position"])})

        if reached_goal:
            self._finish_crl_episode()

        # rollout_stepごとに学習
        if len(self.memory) >= self.rollout_steps:
            self._update_grag()

    def end_episode(self) -> None:
        """ゴール到達または最大行動回数で試行が終わるたびに呼ばれる。"""
        if self._crl_episode:
            self._finish_crl_episode()
        if self.memory:
            self.memory[-1].episode_end = True
            self._update_grag()
    
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
        networks = None if self.actor is None else {
            name: {key: _pack_tensor(tensor)
                   for key, tensor in getattr(self, name).state_dict().items()}
            for name in ("actor", "critic", "sa_encoder", "goal_encoder")}
        return {"schema_version": 2, "algorithm": "GRAG",
                "settings": self.settings.copy(),
                "map_shape": None if self.actor is None else list(self._map_shape),
                "networks": networks, "crl_updates": self._crl_updates}

    @classmethod
    def from_export(cls, data: dict):
        if data.get("schema_version") != 2 or data.get("algorithm") != "GRAG":
            raise ValueError("対応していないGRAGモデル形式です。")
        agent = cls(data["settings"])
        shape = data.get("map_shape")
        if shape is not None:
            rows, cols = shape
            agent.initialize_networks({"rows": rows, "cols": cols, "position": [0, 0],
                                       "goal": [0, 0], "obstacles": []})
            for name, weights in data["networks"].items():
                getattr(agent, name).load_state_dict({key: _unpack_tensor(value) if isinstance(value, dict)
                                                     else torch.tensor(value, dtype=torch.float32)
                                                     for key, value in weights.items()})
        agent._crl_updates = data.get("crl_updates", 0)
        return agent

    def evaluation_copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def checkpoint(self) -> dict:
        def encode(value):
            if isinstance(value, torch.Tensor):
                return _pack_tensor(value)
            if isinstance(value, dict):
                return {str(key): encode(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [encode(item) for item in value]
            return value
        from dataclasses import asdict
        return {"schema_version": 1, "model": self.export(),
                "generator": self._generator.get_state().tolist(),
                "memory": [encode(asdict(item)) for item in self.memory],
                "crl_episode": encode(self._crl_episode), "crl_replay": encode(self._crl_replay),
                "last_advantage_source": self._last_advantage_source,
                "optimizers": {name: encode(getattr(self, name).state_dict())
                               for name in ("_grag_actor_optimizer", "_grag_encoder_optimizer",
                                            "_grag_critic_optimizer") if hasattr(self, name)}}

    def load_checkpoint(self, state: dict) -> None:
        if state.get("schema_version") != 1:
            raise ValueError("対応していないGRAG学習状態の形式です。")
        def decode(value):
            if isinstance(value, dict) and set(value) == {"tensor_dtype", "shape", "data"}:
                return _unpack_tensor(value)
            if isinstance(value, dict) and set(value) == {"dtype", "values"}:
                return torch.tensor(value["values"], dtype=getattr(torch, value["dtype"]))
            if isinstance(value, dict):
                return {key: decode(item) for key, item in value.items()}
            if isinstance(value, list):
                return [decode(item) for item in value]
            return value
        restored = type(self).from_export(state["model"])
        restored._generator.set_state(torch.tensor(state["generator"], dtype=torch.uint8))
        restored.memory = [Transition(**{**decode(item), "goal": tuple(item["goal"]) if item.get("goal") is not None else None}) for item in state.get("memory", [])]
        restored._crl_episode = decode(state.get("crl_episode", []))
        restored._crl_replay = decode(state.get("crl_replay", []))
        restored._last_advantage_source = state.get("last_advantage_source", "gae")
        for name, saved in state.get("optimizers", {}).items():
            if name == "_grag_actor_optimizer":
                parameters = restored.actor.parameters()
            elif name == "_grag_critic_optimizer":
                parameters = restored.critic.parameters()
            elif name == "_grag_encoder_optimizer":
                parameters = list(restored.sa_encoder.parameters()) + list(restored.goal_encoder.parameters())
            else:
                raise ValueError("不明なoptimizerです。")
            optimizer = torch.optim.Adam(parameters, eps=1e-5)
            decoded = decode(saved)
            decoded["state"] = {int(key): value for key, value in decoded["state"].items()}
            optimizer.load_state_dict(decoded)
            setattr(restored, name, optimizer)
        self.__dict__.update(restored.__dict__)

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

    @staticmethod
    def _goal_key(goal):
        return f"{goal[0]},{goal[1]}"

    def _finish_crl_episode(self):
        """実際に訪れた将来位置を目標に再ラベルし、試行全体を保持する。"""
        episode, self._crl_episode = self._crl_episode, []
        if not episode or float(self.settings.get("crl_coef", 1.0)) <= 0:
            return
        # 元タスクを含む各到達位置の目標文を、一試行内で一度だけ構築する。
        future = {}
        examples = []
        for index in reversed(range(len(episode))):
            step = episode[index]
            position = tuple(step["next_position"])
            future[position] = index
            current = tuple((step["state"][2] == 1).nonzero()[0].tolist())
            # その場にとどまっただけの目標を除く。再訪先は最初の到達時刻を使う。
            reached = {self._goal_key(pos): target - index for pos, target in future.items()
                       if pos != current}
            if reached:
                examples.append({"state": step["state"], "action": step["action"],
                                 "goals": reached})
        self._crl_replay.extend(reversed(examples))
        capacity = int(self.settings.get("crl_replay_capacity", 2048))
        if capacity < 2:
            raise ValueError("crl_replay_capacityは2以上にしてください。")
        self._crl_replay = self._crl_replay[-capacity:]
        self._update_contrastive()

    def _update_contrastive(self):
        """到達済み目標の複数正例で、試行をまたいだ対照更新を行う。"""
        if len(self._crl_replay) < 2:
            return
        gamma = float(self.settings.get("crl_gamma", 0.995))
        coef = float(self.settings.get("crl_coef", 1.0))
        if not 0 < gamma < 1 or not math.isfinite(coef) or coef < 0:
            raise ValueError("crl_gammaは0より大きく1未満、crl_coefは0以上の有限値にしてください。")
        if not hasattr(self, "_grag_encoder_optimizer"):
            self._grag_encoder_optimizer = torch.optim.Adam(
                list(self.sa_encoder.parameters()) + list(self.goal_encoder.parameters()),
                lr=self.settings.get("crl_learning_rate", self.learning_rate), eps=1e-5)
        for _ in range(self.update_epochs):
            indices = torch.randperm(len(self._crl_replay), generator=self._generator)
            batch = [self._crl_replay[i] for i in indices[:max(2, self.minibatch_size)].tolist()]
            # 同一(s,a)で別試行から判明した到達先も正例に含める。
            merged = {}
            keys = []
            for item in batch:
                state = item["state"]
                key = (tuple(state.shape), tuple(state[0].flatten().tolist()),
                       tuple(state[2].flatten().tolist()), item["action"])
                keys.append(key)
                goals = merged.setdefault(key, {})
                for prompt, distance in item["goals"].items():
                    goals[prompt] = min(distance, goals.get(prompt, distance))
            batch = [{**item, "goals": merged[key]} for item, key in zip(batch, keys)]
            prompts = list(dict.fromkeys(goal for item in batch for goal in item["goals"]))
            states = torch.stack([item["state"] for item in batch])
            actions = torch.tensor([item["action"] for item in batch])
            sa = self.sa_encoder(state_action_encoder(states, actions))
            goals = torch.tensor([[int(part) for part in key.split(",")] for key in prompts], dtype=torch.float32)
            scores = sa @ self.goal_encoder(goals).T
            log_weights = torch.full_like(scores, -torch.inf)
            for i, item in enumerate(batch):
                for j, prompt in enumerate(prompts):
                    if prompt in item["goals"]:
                        log_weights[i, j] = item["goals"][prompt] * math.log(gamma)
            # 各状態と目標の複数の正例を保持し、同じ目標を負例にしない。
            state_targets = log_weights.softmax(dim=1)
            goal_targets = log_weights.softmax(dim=0)
            loss = coef * (-(state_targets * scores.log_softmax(dim=1)).sum(dim=1).mean()
                           -(goal_targets * scores.log_softmax(dim=0)).sum(dim=0).mean())
            if not torch.isfinite(loss):
                raise FloatingPointError("対照損失にNaN/Infがあります。")
            self._grag_encoder_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.sa_encoder.parameters()) +
                                     list(self.goal_encoder.parameters()), 0.5)
            self._grag_encoder_optimizer.step()
            self._crl_updates += 1

    @torch.no_grad()
    def _q_advantages(self, states, actions, fallback):
        """凍結した対照Qから A(s,a)=Q(s,a)-E_{a'~pi}Q(s,a') を得る。"""
        advantages = fallback.clone()
        warmup = int(self.settings.get("crl_warmup_updates", 10))
        if self._crl_updates < warmup or not self._crl_replay:
            self._last_advantage_source = "gae"
            return advantages
        refs = self._crl_replay[-int(self.settings.get("crl_reference_size", 128)):]
        ref_states = torch.stack([item["state"] for item in refs])
        ref_actions = torch.tensor([item["action"] for item in refs])
        ref_repr = self.sa_encoder(state_action_encoder(ref_states, ref_actions))
        goal_prompts = [self._goal_key(tr.goal) if tr.goal is not None else None for tr in self.memory]
        used = 0
        for prompt in dict.fromkeys(goal_prompts):
            if prompt is None:
                continue
            mask = torch.tensor([prompt in item["goals"] for item in refs], dtype=torch.bool)
            if not mask.any():
                continue
            steps = torch.tensor([item["goals"].get(prompt, 0) for item in refs])
            language = torch.tensor([int(part) for part in prompt.split(",")], dtype=torch.float32)
            goal_repr = self.goal_encoder(language)
            ref_scores = ref_repr @ goal_repr
            selected = torch.tensor([i for i, value in enumerate(goal_prompts) if value == prompt])
            group_states = states[selected]
            all_states = group_states[:, None].expand(-1, len(ACTIONS), -1, -1, -1)
            all_actions = torch.arange(len(ACTIONS)).expand(len(selected), -1)
            q = self._compute_crl_q(state_action_encoder(all_states, all_actions), language,
                                    reference_log_scores=ref_scores,
                                    reference_steps_to_goal=steps,
                                    reference_positive_mask=mask)
            probs = self.actor(group_states).softmax(dim=-1)
            advantages[selected] = q.gather(1, actions[selected, None]).squeeze(1) - (probs * q).sum(dim=-1)
            used += len(selected)
        self._last_advantage_source = "crl_q" if used == len(states) else "mixed" if used else "gae"
        return advantages

    def _update_grag(self):
        """PPOを更新する。対照Qが利用可能になるまではGAEを使う。"""
        if not self.memory:
            return

        crl_gamma = float(self.settings.get("crl_gamma", 0.995))
        crl_coef = float(self.settings.get("crl_coef", 1.0))
        if not 0 < crl_gamma < 1:
            raise ValueError("crl_gammaは0より大きく1未満にしてください。")
        if not math.isfinite(crl_coef) or crl_coef < 0:
            raise ValueError("crl_coefは0以上の有限値にしてください。")

        advantages, returns = self._compute_gae()
        states = torch.stack([tr.state for tr in self.memory])
        actions = torch.tensor([tr.action for tr in self.memory], dtype=torch.long)
        old_log_probs = torch.tensor([tr.old_log_prob for tr in self.memory], dtype=torch.float32)
        batch_size = len(self.memory)
        if crl_coef > 0:
            advantages = self._q_advantages(states, actions, advantages)

        # 既存optimizerはCriticも含むので、このメソッド専用に遅延生成する。
        if not hasattr(self, "_grag_actor_optimizer"):
            self._grag_actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=self.learning_rate, eps=1e-5)
        if not hasattr(self, "_grag_critic_optimizer"):
            self._grag_critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=self.learning_rate, eps=1e-5)
        if len(advantages) > 1:
            advantages = ((advantages - advantages.mean())
                          / (advantages.std(unbiased=False) + 1e-8))

        for epoch in range(self.update_epochs):
            indices = torch.randperm(batch_size, generator=self._generator)
            for start in range(0, batch_size, self.minibatch_size):
                batch_indices = indices[start:start + self.minibatch_size]
                batch_states = states[batch_indices.to(states.device)]
                batch_actions = actions[batch_indices]

                # コピー元のPPOクリップ損失とエントロピー正則化。
                logits = self._encode_crl_features(self.actor, batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions.to(logits.device))
                ratio = (new_log_probs - old_log_probs[batch_indices].to(logits)).exp()
                batch_advantages = advantages[batch_indices].to(logits)
                pg_loss1 = -batch_advantages * ratio
                pg_loss2 = -batch_advantages * ratio.clamp(1.0 - self.clip_coef, 1.0 + self.clip_coef)
                actor_loss = torch.maximum(pg_loss1, pg_loss2).mean() - self.ent_coef * dist.entropy().mean()

                self._grag_actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self._grag_actor_optimizer.step()

                values = self.critic(batch_states).squeeze(-1)
                value_loss = 0.5 * (values - returns[batch_indices].to(values)).square().mean()
                self._grag_critic_optimizer.zero_grad(set_to_none=True)
                (self.vf_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self._grag_critic_optimizer.step()

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

    def _crl_state_features(self, state: torch.Tensor) -> torch.Tensor:
        """stateテンソルからゴール位置情報を除いた状態特徴。他環境ではこの前処理を差し替える。"""
        return state[..., [0, 2], :, :].flatten(start_dim=-3)

    @staticmethod
    def _encode_crl_features(encoder: nn.Module, features: torch.Tensor) -> torch.Tensor:
        """
        引数のテンソルを, 引数のエンコーダーのdevice/dtypeに合わせ, エンコーダーに変換後のテンソルを入力
        エンコーダーからの出力を返す
        """
        reference = next(encoder.parameters(), None)
        if reference is None:
            reference = next(encoder.buffers(), None)
        if reference is not None:
            dtype = reference.dtype if reference.is_floating_point() else features.dtype
            features = features.to(device=reference.device, dtype=dtype)
        return encoder(features)

    def _compute_crl_q(
        self,
        state_action_embedding: torch.Tensor, # 評価対象の状態・行動の特徴ベクトル
        goal_embedding: torch.Tensor, # 評価対象の自然言語目標の特徴ベクトル
        *,
        goal_log_constant: float | torch.Tensor | None = None, # 事前に計算した補正定数C(l)
        reference_log_scores: torch.Tensor | None = None, # 参照データの状態・行動と目標の内積スコア
        reference_steps_to_goal: torch.Tensor | None = None, # 各参照データから目標達成までの残りステップ数
        reference_positive_mask: torch.Tensor | None = None, # 参照データのうち、どれが正例なのかを示すマスク
    ) -> torch.Tensor:
        """PRTS付録B・式(25)のQ = exp(phi(s,a)^T psi(l) - C(l))を計算する。

        state_action_embeddingはstate_action_encoder(state, action)の出力。
        goal_embeddingはencode_language()の出力で、1つの言語特徴ベクトル、またはstate_action_embeddingと同じバッチ形状の言語特徴テンソル。
        C(l)はgoal_log_constantで指定するか、参照内積・正例のT-t・正例マスクからこのメソッド内で計算する。

        通常のQ値テンソルを返す。勾配は保持する。

        
        参照内積は[..., B]、残り時間とマスクは[B]または同形状。
        マスク省略時は全件を正例とする。
        定数も参照データも未指定ならValueError。
        参照データから計算する場合は残り時間も必須。
        同じ目標の行動比較では、同じ参照バッチとC(l)を用いる。
        参照バッチの指定時は、goal_log_constantより参照バッチの計算を優先する。
        """
        if goal_log_constant is None and reference_log_scores is None:
            raise ValueError(
                "goal_log_constantを指定するか、reference_log_scoresと"
                "reference_steps_to_goalを指定してください。")
        
        if reference_log_scores is not None and reference_steps_to_goal is None:
            raise ValueError("補正定数の計算にはreference_steps_to_goalが必要です。")

        # エンコーダーへ入力
        sa_repr = self._encode_crl_features(self.sa_encoder, state_action_embedding)
        goal_repr = self._encode_crl_features(self.goal_encoder, goal_embedding)

        # sa_reprとgoal_reprの型を合わせる
        dtype = torch.promote_types(sa_repr.dtype, goal_repr.dtype)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        
        # 内積をとり，要素を合計
        log_q = (sa_repr.to(dtype) * goal_repr.to(dtype)).sum(dim=-1)


        # 必要時のみ定数部分の計算
        if reference_log_scores is not None:
            # 式(21),(25): C_B(l) = log Z - log sum_positive gamma^(T-t) - log(1-gamma)

            shape = reference_log_scores.shape # 参照スコアのテンソル形状を取得
            scores = reference_log_scores.to(
                torch.float64 if reference_log_scores.dtype == torch.float64 else torch.float32) # 参照スコアをfloat32以上の精度に変換
            mask = (torch.ones_like(scores, dtype=torch.bool) if reference_positive_mask is None else
                    torch.broadcast_to(reference_positive_mask.to(scores.device), shape)) # 正例マスクを作成。未指定なら全データを正例とし、指定時はscoresと同じ形状に拡張
            
            if not mask.any(dim=-1).all():
                raise ValueError("C_B(l)の計算には各目標の正例が1件以上必要です。")
            
            steps = torch.broadcast_to(reference_steps_to_goal.to(scores.device), shape) # ゴールまでのステップ数のテンソルの形をshapeに合わせる
            steps = torch.where(mask, steps, 0).to(scores.dtype) # マスクのところはstepを維持し，それ以外は0とする

            # 定数部分を計算
            log_z = torch.logsumexp(scores, dim=-1)
            # 式(21) C(l, B)第2講Σの中身を計算し，テンソルに保持
            # 式(21) r'は，正例のサンプル番号 (正例集合 S(l;B) =  {j1, ... , j_r', ... , j_m}) であることに注意
            # 負例は計算結果を -inf に置き換えることで e^(-inf) = 0 となり除外する 
            gamma = float(self.settings.get("crl_gamma", 0.995))
            if not 0 < gamma < 1:
                raise ValueError("crl_gammaは0より大きく1未満にしてください。")
            log_weights = (steps * math.log(gamma)).masked_fill(~mask, -torch.inf)
            goal_log_constant = log_z - torch.logsumexp(log_weights, dim=-1) - math.log1p(-gamma)

        # 定数部分をテンソル化して，log_qの計算
        constant = torch.as_tensor(goal_log_constant, device=log_q.device, dtype=log_q.dtype) 
        log_q = log_q - constant

        q = log_q.exp()

        if not torch.isfinite(q).all():
            raise FloatingPointError("Qの計算結果にNaN/Infがあります。内積や補正定数を確認してください。")
        
        return q

    def get_goal_coordinates(self, map_data: dict) -> tuple[int, int]:
        """マップまたは観測のゴール座標を、0始まりの(行, 列)で返す。"""
        goal_row, goal_col = map_data["goal"]
        return goal_row, goal_col

    def generate_goal_prompt(self, map_data: dict) -> str:
        
        # 学習開始時に呼び出す
        """座標の意味・移動条件・達成条件を含む、固定の目標指示文を生成する。

        LLMにもRuriにも渡せるタスク記述。現在位置は状態特徴が保持するため、
        positionは含めず、同じマップ・ゴールの文は時刻によって変化させない。
        """
        goal_row, goal_col = self.get_goal_coordinates(map_data)
        lines = [
            "あなたはグリッドマップ内のエージェントをゴールへ導きます。",
            f"目標: 座標({goal_row}, {goal_col})を目指してください。",
            "座標は(行, 列)の順で、左上を(0, 0)とする0始まりです。",
            "行は下向きに、列は右向きに増えます。",
        ]
        if "rows" in map_data and "cols" in map_data:
            rows, cols = map_data["rows"], map_data["cols"]
            lines.append(f"マップは{rows}行{cols}列で、有効範囲は行0〜{rows - 1}、列0〜{cols - 1}です。")
        if "start" in map_data:
            start_row, start_col = map_data["start"]
            lines.append(f"開始座標は({start_row}, {start_col})です。")
        if "obstacles" in map_data:
            obstacles = sorted({tuple(position) for position in map_data["obstacles"]})
            coordinates = ", ".join(f"({row}, {col})" for row, col in obstacles)
            lines.append(f"通行できない障害物の座標: {coordinates or 'なし'}。")
        lines.extend([
            "1回の行動で上下左右の隣接マスへ1マス移動します。斜め移動はできません。",
            "行動はup=(-1, 0)、down=(1, 0)、left=(0, -1)、right=(0, 1)です。",
            "マップの外側や障害物への移動では、その場にとどまります。",
            f"達成条件: エージェントの現在座標が({goal_row}, {goal_col})と一致することです。",
            "ゴールに近づくだけでは達成ではありません。移動可能なマスを通って到達してください。",
        ])
        return "\n".join(lines)
