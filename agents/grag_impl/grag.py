"""離散行動を扱うGRAGエージェントの実装ひな形。

実装後は既存の学習処理に次のように渡せます。

    from agents.grag import GRAG
    from gridworld.training import train

    result = train(map_data, settings, agent_type=GRAG)
"""

import torch
from torch import nn

ACTIONS = ("up", "down", "left", "right")


class GRAG:
    """行動選択や学習のロジックをこれから実装するためのインターフェース。"""

    def __init__(self, settings: dict):
        """学習パラメータを受け取る。モデルや内部状態の初期化はここに追加する。"""
        self.settings = settings
        self.actions = tuple(settings.get("actions", ACTIONS))
        if not self.actions:
            raise ValueError("actionsには1つ以上の行動を指定してください。")
        self.observation_encoder = settings.get("observation_encoder")
        if self.observation_encoder is not None and not callable(self.observation_encoder):
            raise ValueError("observation_encoderは呼び出し可能にしてください。")

        self.sa_encoder = None
        self.goal_encoder = None
        self.actor = None
        self._network_spec = None
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
                raise ValueError("problem???????????????")
            state_dim, goal_dim, action_dim = problem.network_dimensions(len(self.actions))
            if not callable(getattr(problem, "encode_observation", None)):
                raise ValueError("problem??encode_observation??????")
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

        # ネットワークを初期化
        # start_dim : 状態sの次元
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._generator.initial_seed())

            self.actor = nn.Sequential(
                nn.Flatten(), nn.Linear(state_dim, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, action_dim),
            )
            
            self.sa_encoder = build_encoder(state_dim + action_dim)
            self.goal_encoder = build_encoder(goal_dim)
        self._network_spec = spec
        if problem is not None:
            self.observation_encoder = problem.encode_observation

    def initialize_for_gridworld(self, map_data: dict) -> None:
        """GridWorldのマップから特徴次元を決める。"""
        from agents.grag_impl.gridworld_adapter import GridWorldAdapter

        self.initialize_networks(problem=GridWorldAdapter(map_data))

    def _action_logits(self, observation) -> torch.Tensor:
        """観測を変換し、現在のActorの行動スコアを返す。"""
        if self.observation_encoder is not None:
            features = self.observation_encoder(observation)
        else:
            features = observation
        state = torch.as_tensor(features, dtype=torch.float32).detach().cpu()
        if state.numel() == 0 or not torch.isfinite(state).all():
            raise ValueError("観測の特徴は空でない有限の数値にしてください。")
        if self.actor is None:
            raise RuntimeError("initialize_networks()でネットワークを初期化してください。")
        if state.numel() != self._network_spec[0]:
            raise ValueError("観測の特徴の要素数がstate_dimと一致しません。")

        with torch.no_grad():
            reference = next(self.actor.parameters())
            logits = self.actor(state.reshape(1, -1).to(reference)).squeeze(0)
            if not torch.isfinite(logits).all():
                raise ValueError("Actorの出力は有限の数値にしてください。")
            return logits

    def choose_action(self, observation, greedy: bool = False):
        """観測から行動を選ぶ。greedy=Trueでは最大確率の行動を返す。"""
        logits = self._action_logits(observation)
        with torch.no_grad():
            if greedy:
                action = logits.argmax().item()
            else:
                probabilities = torch.softmax(logits, dim=-1).cpu()
                action = torch.multinomial(probabilities, 1, generator=self._generator).item()
        return self.actions[action]

    def action_probabilities(self, observation) -> dict:
        """各行動の選択確率を返す。Actorの初期化と乱数の消費は行わない。"""
        try:
            if len(set(self.actions)) != len(self.actions):
                raise ValueError("行動確率の取得には重複しないactionsが必要です。")
        except TypeError as exc:
            raise ValueError("行動確率の取得には辞書のキーとして使えるactionsが必要です。") from exc
        probabilities = torch.softmax(self._action_logits(observation), dim=-1).cpu().tolist()
        return dict(zip(self.actions, probabilities))

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

    def export(self) -> dict:
        """保存用のJSONに変換可能な辞書を返す。

        現在の学習処理では結果JSONの q_table に格納される。
        保存する学習データがなければ空の辞書を返す。
        """
        # TODO: 必要なら保存する学習データを返す。
        return {}
