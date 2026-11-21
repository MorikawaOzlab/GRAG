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
        self.contrastive_epochs = settings.get("grag_contrastive_epochs", 1)
        self.actions = tuple(settings.get("actions", ACTIONS))
        if not self.actions:
            raise ValueError("actionsには1つ以上の行動を指定してください。")
        self.epsilon = settings.get("epsilon", 0.0)
        self.epsilon_decay = settings.get("epsilon_decay", 1.0)
        self.min_epsilon = settings.get("min_epsilon", 0.0)
        for name in ("epsilon", "epsilon_decay", "min_epsilon"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError(f"{name}は0〜1の数値にしてください。")
        if self.min_epsilon > self.epsilon:
            raise ValueError("min_epsilonはepsilon以下にしてください。")
        self.observation_encoder = settings.get("observation_encoder")
        if self.observation_encoder is not None and not callable(self.observation_encoder):
            raise ValueError("observation_encoderは呼び出し可能にしてください。")
        self.achieved_goal_encoder = settings.get("achieved_goal_encoder")
        if self.achieved_goal_encoder is not None and not callable(self.achieved_goal_encoder):
            raise ValueError("achieved_goal_encoderは呼び出し可能にしてください。")
        self.target_goal_encoder = settings.get("target_goal_encoder")
        if self.target_goal_encoder is not None and not callable(self.target_goal_encoder):
            raise ValueError("target_goal_encoderは呼び出し可能にしてください。")
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
        self.cumulative_positive_goal_counts: dict[tuple[float, ...], int] = {}
        self._contrastive_pending = []  # バッチ内に負例が揃うまで保持
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

    @staticmethod
    def _exploration_log_probs(logits: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        """Actorと一様探索を混合した行動分布の対数確率。"""
        actor = F.log_softmax(logits, dim=-1) + torch.log1p(-epsilon)
        uniform = epsilon.log() - math.log(logits.shape[-1])
        return torch.logaddexp(actor, uniform)

    def choose_action(self, observation, greedy: bool = False):
        """学習時はε探索を混ぜ、評価時はActorの最大確率の行動を返す。"""
        
        if not greedy and self.pending is not None:
            raise RuntimeError("次の行動選択前にlearn()を呼んでください。")
        
        state = self._prepare_state(observation)
        logits = self._action_logits(state)

        with torch.no_grad():
            if greedy:
                action = logits.argmax().item()
            else:
                log_probs = self._exploration_log_probs(logits, logits.new_tensor(self.epsilon))
                action = torch.multinomial(log_probs.exp().cpu(), 1, generator=self._generator).item()

                # learnメソッドで学習用にメモリに記録するための一時保存
                self.pending = {
                    "observation": deepcopy(observation),
                    "state": state,
                    "action": action,  # actions内のインデックス
                    "log_prob": float(log_probs[action].item()),
                    "epsilon": self.epsilon,
                }

        return self.actions[action]

    def action_probabilities(self, observation) -> dict:
        """評価用にActorの行動確率を返す。乱数は消費しない。"""
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
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


    #===============
    # 学習用メソッド
    #===============

    def _update_networks(self, transitions: list) -> None:
        """収集した経験を対照学習とPPOの更新に渡す。"""
        # CPPOと同様、更新前のCriticでadvantageを計算してActor更新中は固定する。
        if self._update_contrastive(transitions):
            self._update_ppo(transitions)

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

    def _update_contrastive(self, transitions: list) -> bool:
        """InfoNCEで更新できた場合にTrueを返す。"""

        # 1. 経験の蓄積
        self._contrastive_pending.extend(transitions)
        batch = self._contrastive_pending

        if len(batch) < 2:
            self.contrastive_loss = None
            return False

        # 2. 学習データの準備
        reference = next(self.sa_encoder.parameters()) # エンコーダの１番目の学習対象パラメータ
        device = reference.device

        # バッチからデータを取り出す．各エピソードを1次元で管理する
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

        # HERを適用
        hindsight_goals = self._sample_hindsight_goals(batch)
        goals = hindsight_goals.to(reference)

        # 各行の正例は、その遷移の (s, a) と HER で割り当てたゴールの組。
        batch_size = len(batch)
        negative_count = min(255, batch_size - 1)
        positive_indices = torch.arange(batch_size).unsqueeze(1)
        negative_indices = torch.stack([
            torch.randperm(batch_size - 1, generator=self._generator)[:negative_count].clone()
            for _ in batch
        ])
        # 正例のインデックスを飛ばして、残りの遷移から重複なしで抽選する。
        negative_indices += negative_indices >= positive_indices
        candidate_indices = torch.cat((positive_indices, negative_indices), dim=1).to(device)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        sa_inputs = torch.cat((states, actions), dim=-1)
        parameters = list(self.sa_encoder.parameters()) + list(self.goal_encoder.parameters())

        # ゴールを行に固定し、候補の (s, a) に対する InfoNCE をバッチで平均する。
        for _ in range(self.contrastive_epochs):
            sa_repr = self.sa_encoder(sa_inputs)
            goal_repr = self.goal_encoder(goals)
            logits = -torch.cdist(goal_repr, sa_repr).gather(1, candidate_indices)
            loss = F.cross_entropy(logits, labels)

            self.contrastive_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=0.5)
            self.contrastive_optimizer.step()

        # 学習結果の保存と経験の初期化
        goal_values, counts = torch.unique(hindsight_goals.cpu(), dim=0, return_counts=True)
        for goal, count in zip(goal_values.tolist(), counts.tolist()):
            key = tuple(goal)
            self.cumulative_positive_goal_counts[key] = self.cumulative_positive_goal_counts.get(key, 0) + count
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

    def _update_ppo(self, transitions: list) -> None:
        """対照学習のQ値からadvantageを計算し、PPOでActorを更新する。"""
        reference = next(self.actor.parameters())
        states = torch.stack([row["state"].reshape(-1) for row in transitions]).to(reference)
        actions = torch.tensor([row["action"] for row in transitions], device=reference.device)
        old_log_probs = torch.tensor([row["log_prob"] for row in transitions],
                                     dtype=reference.dtype, device=reference.device)
        epsilons = reference.new_tensor([row.get("epsilon", 0.0) for row in transitions]).unsqueeze(1)
        goals = []
        for row in transitions:
            goal = (self.target_goal_encoder(row["observation"])
                    if self.target_goal_encoder is not None else row["state"])
            goal = torch.as_tensor(goal, dtype=torch.float32).detach().reshape(-1)
            if goal.numel() != self._network_spec[1] or not torch.isfinite(goal).all():
                raise ValueError("目標ゴールはgoal_dimと一致する有限の値にしてください。")
            goals.append(goal)
        goals = torch.stack(goals).to(reference)

        with torch.no_grad():
            q_values = self.q_values(states, goals)
            probabilities = self._exploration_log_probs(self.actor(states), epsilons).exp()
            baseline = (probabilities * q_values).sum(dim=-1)
            advantages = q_values.gather(1, actions[:, None]).squeeze(1) - baseline
            if len(transitions) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        for _ in range(2):
            log_probs = self._exploration_log_probs(self.actor(states), epsilons)
            distribution = torch.distributions.Categorical(logits=log_probs)
            ratio = (distribution.log_prob(actions) - old_log_probs).exp()
            clipped_ratio = ratio.clamp(0.8, 1.2)
            policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
            loss = policy_loss - 0.01 * distribution.entropy().mean()
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
