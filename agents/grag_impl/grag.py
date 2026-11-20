"""離散行動を扱うGRAGエージェント。

既存の学習処理に次のように渡せます。

    from agents.grag import GRAG
    from gridworld.training import train

    result = train(map_data, settings, agent_type=GRAG)
"""

from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

ACTIONS = ("up", "down", "left", "right")


class GRAG:
    """Actorによる行動選択と、エピソード単位の経験収集を行う。"""

    UPDATE_INTERVAL = 128*10

    def __init__(self, settings: dict):
        """設定と経験収集用のメモリを初期化する。"""
        self.settings = settings
        self.UPDATE_INTERVAL = settings.get("grag_update_interval", self.UPDATE_INTERVAL)
        if type(self.UPDATE_INTERVAL) is not int or not 1 <= self.UPDATE_INTERVAL <= 1_000_000:
            raise ValueError("grag_update_intervalは1〜1,000,000の整数にしてください。")
        self.actions = tuple(settings.get("actions", ACTIONS))
        if not self.actions:
            raise ValueError("actionsには1つ以上の行動を指定してください。")
        self.observation_encoder = settings.get("observation_encoder")
        if self.observation_encoder is not None and not callable(self.observation_encoder):
            raise ValueError("observation_encoderは呼び出し可能にしてください。")
        self.achieved_goal_encoder = settings.get("achieved_goal_encoder")
        if self.achieved_goal_encoder is not None and not callable(self.achieved_goal_encoder):
            raise ValueError("achieved_goal_encoderは呼び出し可能にしてください。")
        self.target_goal_encoder = settings.get("target_goal_encoder")
        if self.target_goal_encoder is not None and not callable(self.target_goal_encoder):
            raise ValueError("target_goal_encoderは呼び出し可能にしてください。")
        self.relabel_goal = None
        self.contrastive_learning_rate = settings.get("contrastive_learning_rate", 3e-4)
        if (isinstance(self.contrastive_learning_rate, bool)
                or not isinstance(self.contrastive_learning_rate, (int, float))
                or not math.isfinite(self.contrastive_learning_rate)
                or self.contrastive_learning_rate <= 0):
            raise ValueError("contrastive_learning_rateは正の有限数にしてください。")
        self.hindsight_gamma = settings.get("hindsight_gamma", 0.99)
        if (isinstance(self.hindsight_gamma, bool)
                or not isinstance(self.hindsight_gamma, (int, float))
                or not math.isfinite(self.hindsight_gamma)
                or not 0 < self.hindsight_gamma < 1):
            raise ValueError("hindsight_gammaは0より大きく1未満にしてください。")

        self.sa_encoder = None
        self.goal_encoder = None
        self.actor = None
        self.contrastive_optimizer = None
        self.actor_optimizer = None
        self.policy_loss = None
        self.contrastive_loss = None
        self._contrastive_pending = []  # 目標ごとの負例が揃うまで保持
        self._network_spec = None
        self.episodes = []    # 終了したエピソードの軌跡
        self.trajectory = []  # 現在のエピソードの遷移
        self.pending = None   # 行動選択後、learn()を待っている情報
        self._update_batch = []  # 次の更新までに収集した遷移
        self._episode_id = 0
        self._generator = torch.Generator(device="cpu")
        seed = settings.get("seed")
        if seed is None:
            self._generator.seed()
        elif type(seed) is int and 0 <= seed < 2**32:
            self._generator.manual_seed(seed)
        else:
            raise ValueError("seedは0〜2^32-1の整数またはNoneにしてください。")

    def initialize_networks(self, state_dim: int | None = None, goal_dim: int | None = None,
                            action_dim: int | None = None, *, problem=None) -> None:
        """Actorと対照学習用ネットワークを初期化する。

        Args:
            state_dim: 状態ベクトルの次元数
            goal_dim: ゴールベクトルの次元数
            action_dim: 離散行動の数。省略時はsettingsのactionsの数。
            problem: network_dimensions(action_dim) と encode_observation(observation)
                を提供する問題アダプター。指定時は各次元を直接渡さない。

        状態・行動エンコーダー: φ(s,a)
        ゴールエンコーダー: ψ(g)
        """
        if problem is not None:
            if state_dim is not None or goal_dim is not None or action_dim is not None:
                raise ValueError("problemと次元を同時に指定できません。")
            state_dim, goal_dim, action_dim = problem.network_dimensions(len(self.actions))
            if not callable(getattr(problem, "encode_observation", None)):
                raise ValueError("problemにはencode_observationが必要です。")
            goal_encoder = getattr(problem, "encode_achieved_goal", self.achieved_goal_encoder)
            if goal_encoder is not None and not callable(goal_encoder):
                raise ValueError("encode_achieved_goalは呼び出し可能にしてください。")
            target_encoder = getattr(problem, "encode_goal", self.target_goal_encoder)
            if target_encoder is not None and not callable(target_encoder):
                raise ValueError("encode_goalは呼び出し可能にしてください。")
            relabel_goal = getattr(problem, "relabel_goal", None)
            if relabel_goal is not None and not callable(relabel_goal):
                raise ValueError("relabel_goalは呼び出し可能にしてください。")
        if action_dim is None:
            action_dim = len(self.actions)
        if any(type(n) is not int or n <= 0 for n in (state_dim, goal_dim, action_dim)):
            raise ValueError("state_dim、goal_dim、action_dimは正の整数にしてください。")
        if action_dim != len(self.actions):
            raise ValueError("action_dimはactionsの数と一致させてください。")
        spec = (state_dim, goal_dim, action_dim)
        if self._network_spec is not None:
            if self._network_spec != spec:
                raise ValueError("初期化済みネットワークの次元は変更できません。")
            if problem is not None:
                self.observation_encoder = problem.encode_observation
                self.achieved_goal_encoder = goal_encoder
                self.target_goal_encoder = target_encoder
                self.relabel_goal = relabel_goal
            return
        repr_dim = 64

        def build_encoder(input_dim: int) -> nn.Sequential:
            layers = []

            for _ in range(4):
                layers.extend([
                    nn.Linear(input_dim, 512),
                    nn.LayerNorm(512),
                    nn.SiLU(),
                ])
                input_dim = 512

            layers.append(nn.Linear(512, repr_dim))

            return nn.Sequential(*layers)

        # ネットワークの初期化によって外部の乱数状態を変えない。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._generator.initial_seed())

            self.actor = nn.Sequential(
                nn.Flatten(), nn.Linear(state_dim, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, action_dim),
            )

            self.sa_encoder = build_encoder(state_dim + action_dim)
            self.goal_encoder = build_encoder(goal_dim)
        self.contrastive_optimizer = torch.optim.Adam(
            list(self.sa_encoder.parameters()) + list(self.goal_encoder.parameters()),
            lr=self.contrastive_learning_rate,
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4, eps=1e-5)
        self._network_spec = spec
        if problem is not None:
            self.observation_encoder = problem.encode_observation
            self.achieved_goal_encoder = goal_encoder
            self.target_goal_encoder = target_encoder
            self.relabel_goal = relabel_goal

    def initialize_for_gridworld(self, map_data: dict) -> None:
        """GridWorldのマップから特徴次元を決める。"""
        from agents.grag_impl.gridworld_adapter import GridWorldAdapter

        self.initialize_networks(problem=GridWorldAdapter(map_data))

    def _prepare_state(self, observation) -> torch.Tensor:
        """アダプターで観測を変換し、共通の検証とコピーを行う。"""
        if self.observation_encoder is not None:
            features = self.observation_encoder(observation)
        else:
            features = observation
        state = torch.as_tensor(features, dtype=torch.float32).detach().cpu().clone()
        if state.numel() == 0 or not torch.isfinite(state).all():
            raise ValueError("観測の特徴は空でない有限の数値にしてください。")
        if self.actor is None:
            raise RuntimeError("initialize_networks()でネットワークを初期化してください。")
        if state.numel() != self._network_spec[0]:
            raise ValueError("観測の特徴の要素数がstate_dimと一致しません。")

        return state

    def _action_logits(self, state: torch.Tensor) -> torch.Tensor:
        """変換済みの状態に対するActorの行動スコアを返す。"""
        with torch.no_grad():
            reference = next(self.actor.parameters())
            logits = self.actor(state.reshape(1, -1).to(reference)).squeeze(0)
            if not torch.isfinite(logits).all():
                raise ValueError("Actorの出力は有限の数値にしてください。")
            return logits

    def choose_action(self, observation, greedy: bool = False):
        """観測から行動を選ぶ。greedy=Trueでは最大確率の行動を返す。"""
        
        if not greedy and self.pending is not None:
            raise RuntimeError("次の行動選択前にlearn()を呼んでください。")
        
        state = self._prepare_state(observation)
        logits = self._action_logits(state)

        with torch.no_grad():
            if greedy:
                action = logits.argmax().item()
            else:
                probabilities = torch.softmax(logits, dim=-1).cpu()
                action = torch.multinomial(probabilities, 1, generator=self._generator).item()

                # learnメソッドで学習用にメモリに記録するための一時保存
                self.pending = {
                    "observation": deepcopy(observation),
                    "state": state,
                    "action": action,  # actions内のインデックス
                    "log_prob": float(torch.log_softmax(logits, dim=-1)[action].item()),
                }

        return self.actions[action]

    def action_probabilities(self, observation) -> dict:
        """各行動の選択確率を返す。Actorの初期化と乱数の消費は行わない。"""
        try:
            if len(set(self.actions)) != len(self.actions):
                raise ValueError("行動確率の取得には重複しないactionsが必要です。")
        except TypeError as exc:
            raise ValueError("行動確率の取得には辞書のキーとして使えるactionsが必要です。") from exc
        
        state = self._prepare_state(observation)
        probabilities = torch.softmax(self._action_logits(state), dim=-1).cpu().tolist()

        return dict(zip(self.actions, probabilities))

    def learn(self, observation, action, reward: float,
              next_observation, reached_goal: bool) -> None:

        if self.pending is None or self.actions[self.pending["action"]] != action:
            raise RuntimeError("learn()には直前に選択した行動を渡してください。")
        
        next_state = self._prepare_state(next_observation)
        transition = {
            **self.pending,
            "reward": reward,
            "next_observation": deepcopy(next_observation),
            "next_state": next_state,
            "reached_goal": reached_goal,
            "episode_id": self._episode_id,
        }
        self.trajectory.append(transition)
        self._update_batch.append(transition)

        self.pending = None

        # データがたまったら重みを更新
        if len(self._update_batch) == self.UPDATE_INTERVAL:
            self._update_networks(self._update_batch)
            self._update_batch = []

    def end_episode(self) -> None:
        """ゴール到達または最大行動回数で試行が終わるたびに呼ばれる。"""
        if self.pending is not None:
            raise RuntimeError("learn()の後にend_episode()を呼んでください。")
        if self.trajectory:
            self.episodes.append(self.trajectory)
            self.trajectory = []
            self._episode_id += 1


    #===============
    # 学習用メソッド
    #===============

    def _update_networks(self, transitions: list) -> None:
        """同じ経験とHER目標を対照学習とPPOに使う。"""
        batch = self._contrastive_pending + transitions
        if not batch:
            return
        hindsight_goals = self._sample_hindsight_goals(batch)
        if self._update_contrastive(transitions, hindsight_goals=hindsight_goals):
            self._update_ppo(batch, hindsight_goals=hindsight_goals)

    def _sample_hindsight_goals(self, transitions: list) -> torch.Tensor:
        """各遷移から、同一エピソード内の将来の到達ゴールを選択する。"""

        future_goals = []

        for index, transition in enumerate(transitions):

            # 1. 同一エピソードの現在以降の経験を取得
            episode_id = transition["episode_id"]

            candidates = [
                i for i in range(index, len(transitions))
                if transitions[i]["episode_id"] == episode_id
            ]

            # 2. 時間的に近い経験ほど高い確率で選択
            distances = torch.arange(
                len(candidates), dtype=torch.float32
            )
            weights = self.hindsight_gamma ** distances

            sampled_index = torch.multinomial(
                weights, num_samples=1, generator=self._generator
            ).item()

            future_transition = transitions[candidates[sampled_index]]

            # 3. 選択した経験から到達ゴールを取得
            if self.achieved_goal_encoder is not None:
                goal = self.achieved_goal_encoder(
                    future_transition["next_observation"]
                )
            else:
                goal = future_transition["next_state"]

            goal = torch.as_tensor(
                goal, dtype=torch.float32
            ).detach().reshape(-1)

            # 4. ゴールの次元数と値を検証
            goal_dim = self._network_spec[1]

            if goal.numel() != goal_dim or not torch.isfinite(goal).all():
                raise ValueError(
                    "到達ゴールはgoal_dimと一致する有限の値にしてください。"
                )

            # 5. ゴールを保存
            future_goals.append(goal)

        return torch.stack(future_goals)

    @staticmethod
    def _goal_infonce_loss(logits: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        """各行の目標を固定したInfoNCEを計算し、目標間で平均する。"""
        log_probs = F.log_softmax(logits, dim=1)
        positive_log_probs = log_probs.masked_fill(~positives, 0)
        goal_losses = -positive_log_probs.sum(dim=1) / positives.sum(dim=1)
        return goal_losses.mean()

    def _update_contrastive(self, transitions: list, *, hindsight_goals: torch.Tensor | None = None) -> bool:
        """InfoNCEで更新できた場合にTrueを返す。"""

        # 1. 経験の蓄積
        self._contrastive_pending.extend(transitions)
        batch = self._contrastive_pending

        if not batch:
            self.contrastive_loss = None
            return False

        # 2. 学習データの準備
        reference = next(self.sa_encoder.parameters()) # エンコーダの１番目の学習対象パラメータ
        device = reference.device

        # バッチから各遷移の状態・行動を取り出す。
        # エンコーダのパラメータと同じデータ型，デバイスにそろえる
        states = torch.stack([
            t["state"].reshape(-1) for t in batch
        ]).to(reference)

        actions = torch.tensor(
            [t["action"] for t in batch],
            device=device,
        )

        actions = F.one_hot(
            actions, num_classes=len(self.actions)
        ).to(reference)

        # 3. HERの目標ごとに正例の状態・行動をまとめる。
        if hindsight_goals is None:
            hindsight_goals = self._sample_hindsight_goals(batch)
        goals = hindsight_goals.to(reference)
        unique_goals, goal_indices = torch.unique(
            goals, dim=0, return_inverse=True
        )

        sa_inputs = torch.cat([states, actions], dim=-1)
        unique_sa, sa_indices = torch.unique(
            sa_inputs, dim=0, return_inverse=True
        )

        # 目標と状態・行動の正例関係を作成
        positives = torch.zeros(
            (len(unique_goals), len(unique_sa)),
            dtype=torch.bool,
            device=device
        )
        positives[goal_indices, sa_indices] = True

        # 正例と負例の両方が存在する目標だけを使用
        has_positives = positives.any(dim=1)
        has_negatives = (~positives).any(dim=1)
        valid_goals = has_positives & has_negatives

        if not valid_goals.any():
            self.contrastive_loss = None
            return False

        # 4. 目標ごとに複数正例InfoNCEを計算する。
        sa_repr = self.sa_encoder(unique_sa)
        goal_repr = self.goal_encoder(unique_goals[valid_goals])

        logits = -torch.cdist(goal_repr, sa_repr)

        loss = self._goal_infonce_loss(
            logits,
            positives[valid_goals]
        )

        # 5. 目標間の平均損失でエンコーダを更新する。
        self.contrastive_optimizer.zero_grad()
        loss.backward()

        parameters = (
            list(self.sa_encoder.parameters())
            + list(self.goal_encoder.parameters())
        )

        nn.utils.clip_grad_norm_(parameters, max_norm=0.5)
        self.contrastive_optimizer.step()



        self.contrastive_loss = loss.detach().item()
        self._contrastive_pending = []
        return True

    def q_values(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        """各状態・ゴールに対する全行動の対照Q値を返す。"""
        reference = next(self.actor.parameters())
        states = states.reshape(states.shape[0], -1).to(reference)
        goals = goals.reshape(goals.shape[0], -1).to(reference)
        action_count = len(self.actions)
        actions = torch.eye(action_count, device=reference.device, dtype=reference.dtype)
        all_states = states.repeat_interleave(action_count, dim=0)
        state_actions = torch.cat((all_states, actions.repeat(len(states), 1)), dim=-1)
        state_action_repr = self.sa_encoder(state_actions)
        goal_repr = self.goal_encoder(goals).repeat_interleave(action_count, dim=0)
        return -torch.linalg.vector_norm(state_action_repr - goal_repr, dim=-1).reshape(-1, action_count)

    def _update_ppo(self, transitions: list, *, hindsight_goals: torch.Tensor | None = None) -> None:
        """元の目標とHER目標ごとのPPO損失を平均してActorを更新する。"""
        reference = next(self.actor.parameters())
        states = torch.stack([row["state"].reshape(-1) for row in transitions]).to(reference)
        actions = torch.tensor([row["action"] for row in transitions], device=reference.device)
        old_log_probs = torch.tensor([row["log_prob"] for row in transitions],
                                     dtype=reference.dtype, device=reference.device)
        goals = []
        for row in transitions:
            goal = (self.target_goal_encoder(row["observation"])
                    if self.target_goal_encoder is not None else row["state"])
            goal = torch.as_tensor(goal, dtype=torch.float32).detach().reshape(-1)
            if goal.numel() != self._network_spec[1] or not torch.isfinite(goal).all():
                raise ValueError("目標ゴールはgoal_dimと一致する有限の値にしてください。")
            goals.append(goal)
        goals = torch.stack(goals).to(reference)

        actor_states = states
        if hindsight_goals is not None:
            hindsight_goals = hindsight_goals.to(reference)
            relabeled = (hindsight_goals != goals).any(dim=1).nonzero(as_tuple=True)[0]
            if relabeled.numel():
                relabeled_goals = hindsight_goals[relabeled]
                relabeled_states = states[relabeled]
                if self.relabel_goal is not None:
                    relabeled_states = torch.stack([
                        self._prepare_state(self.relabel_goal(transitions[index]["observation"], goal)).reshape(-1)
                        for index, goal in zip(relabeled.tolist(), relabeled_goals.cpu())
                    ]).to(reference)
                actor_states = torch.cat([states, relabeled_states])
                # Criticは収集時の状態表現とHER目標の組を学習する。
                states = torch.cat([states, states[relabeled]])
                goals = torch.cat([goals, relabeled_goals])
                actions = torch.cat([actions, actions[relabeled]])
                old_log_probs = torch.cat([old_log_probs, old_log_probs[relabeled]])

        # 各目標内で平均し、遷移数によらず各目標を等しく扱う。
        _, goal_ids, goal_counts = torch.unique(goals, dim=0, return_inverse=True, return_counts=True)
        sample_weights = (goal_counts[goal_ids].to(reference) * len(goal_counts)).reciprocal()

        with torch.no_grad():
            q_values = self.q_values(states, goals)
            probabilities = torch.softmax(self.actor(actor_states), dim=-1)
            baseline = (probabilities * q_values).sum(dim=-1)
            advantages = q_values.gather(1, actions[:, None]).squeeze(1) - baseline
            if len(goals) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        for _ in range(4):
            distribution = torch.distributions.Categorical(logits=self.actor(actor_states))
            ratio = (distribution.log_prob(actions) - old_log_probs).exp()
            clipped_ratio = ratio.clamp(0.8, 1.2)
            policy_loss = -(sample_weights * torch.minimum(ratio * advantages, clipped_ratio * advantages)).sum()
            loss = policy_loss - 0.01 * (sample_weights * distribution.entropy()).sum()
            self.actor_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
        self.policy_loss = float(policy_loss.detach())

    def export(self) -> dict:
        """保存用のJSONに変換可能な辞書を返す。

        現在の学習処理では結果JSONの q_table に格納される。
        保存する学習データがなければ空の辞書を返す。
        """
        # TODO: 必要なら保存する学習データを返す。
        return {}
