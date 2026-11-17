"""開発中の強化学習システムGRAG。

実装後は既存の学習処理に次のように渡せます。

    from agents.grag import GRAG
    from gridworld.training import train

    result = train(map_data, settings, agent_type=GRAG)

    Goal-Reachabilioty-Aware Gating
"""

import math
from functools import lru_cache

import torch
from torch import nn

from dataclasses import dataclass

from .components.losses import compute_crl_loss, compute_ppo_loss
from .components.networks import build_networks
ACTIONS = ("up", "down", "left", "right")

@dataclass
class Transition:
    state: torch.Tensor
    action: int
    old_log_prob: float
    reward: float
    reached_goal: bool
    episode_end: bool = False
    goal_prompt: str | None = None

def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


@lru_cache(maxsize=1)
def _load_language_encoder(device: str | None):
    """初回のエンコード時にRuriを読み込み、同じデバイスで再利用する。"""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("cl-nagoya/ruri-v3-70m", device=device)

def encode_language(
    texts: str | list[str], *, device: str | None = None
) -> torch.Tensor:
    """日本語の自然言語をRuri v3 70mでエンコードする。

    単文は[384]、文のリストは[文数, 384]のL2正規化済みテンソルを返す。
    意味を表す埋め込みのため、検索用などの接頭辞は付けない。
    device=NoneではSentence TransformersがGPU/CPUを自動選択する。 (cpu or cuda)
    """
    sentences = [texts] if isinstance(texts, str) else texts
    if not isinstance(sentences, list) or not sentences:
        raise ValueError("textsは文字列または空でない文字列のリストにしてください。")
    if any(not isinstance(text, str) or not text.strip() for text in sentences):
        raise ValueError("空でない自然言語の文字列を指定してください。")

    model = _load_language_encoder(device)
    return model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


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
        self.optimizer = None

        # memory
        self.memory: list[Transition] = []
        self._successful_trajectories = 0
        self._goal_embedding_cache: dict[str, torch.Tensor] = {}

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
        self.positive_update_count = settings.get("grag_positive_update_count", 8)
        if (type(self.positive_update_count) is not int
                or not 1 <= self.positive_update_count <= 1_000_000):
            raise ValueError("grag_positive_update_countは1〜1,000,000の整数にしてください。")

        seed = settings.get("seed")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("seedは0〜2^32-1の整数またはNoneにしてください。")
        self._generator = torch.Generator(device="cpu")
        self.seed = self._generator.seed() if seed is None else seed
        self._generator.manual_seed(self.seed)

    def initialize_networks(self, observation:dict):
        # ネットワークが未初期化のときは初期化をする
        if self.actor is not None:
            return
        
        state = self.observation_to_tensor(observation)
        observation_shape = state.shape

        self.actor, self.sa_encoder, self.goal_encoder = build_networks(
            observation_shape, self.hidden_size, self.CRL_NUM_ACTIONS, seed=self.seed
        )

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()),
            lr=self.learning_rate,
            eps=1e-5
        )

        self.crl_optimizer = torch.optim.Adam(
            list(self.sa_encoder.parameters())
            + list(self.goal_encoder.parameters()),
            lr=self.settings.get("crl_learning_rate", 2.5e-4),
            eps=1e-5,
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
        action_index = ACTIONS.index(action)

        with torch.no_grad():
            # Actorの出力
            logits = self.actor(state)
            dist = torch.distributions.Categorical(logits=logits)

            # 行動の対数確率
            old_log_prob = dist.log_prob(
                torch.tensor(action_index)
            ).item()

        transition = Transition(
            state=state,
            action=action_index,
            old_log_prob=old_log_prob,
            reward=reward,
            reached_goal=reached_goal,
            goal_prompt=self.generate_goal_prompt(observation),
        )

        self.memory.append(transition)
        if reached_goal:
            self._successful_trajectories += 1

        # 成功した軌道が指定数そろうまで方策は固定する。
        if self._successful_trajectories >= self.positive_update_count:
            if self._update_grag():
                self._update_ppo()
                self._successful_trajectories = 0

    def end_episode(self) -> None:
        """試行終了時、未更新の最後の遷移に境界を記録する。"""
        if self.memory:
            self.memory[-1].episode_end = True
            if not self.memory[-1].reached_goal:
                # 成功を待つ間も失敗軌道を無制限に保持しない。
                episodes, start = [], 0
                for index, transition in enumerate(self.memory):
                    if transition.episode_end or transition.reached_goal:
                        episodes.append(self.memory[start:index + 1])
                        start = index + 1
                failed_steps = sum(len(episode) for episode in episodes
                                   if not episode[-1].reached_goal)
                failed_count = sum(not episode[-1].reached_goal for episode in episodes)
                kept = []
                for episode in episodes:
                    if (not episode[-1].reached_goal and failed_count > 1
                            and failed_steps > self.rollout_steps):
                        failed_steps -= len(episode)
                        failed_count -= 1
                    else:
                        kept.extend(episode)
                self.memory = kept
    
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

    # ===========
    # ここから
    # ===========

    def _update_grag(self) -> bool:
        """成功軌道から対照学習し、更新したかどうかを返す。"""
        if not self.memory:
            return False
        
        transitions = self.memory

        # 1. 対照損失を計算し、2. 両エンコーダーを更新する。
        crl_loss = compute_crl_loss(self, transitions, state_action_encoder, encode_language)
        if crl_loss is not None:
            self._update_crl_encoders(crl_loss)
            return True
        return False


    def _update_crl_encoders(self, crl_loss: torch.Tensor) -> None:
        """対照損失を用いて両エンコーダーを更新する。"""

        self.crl_optimizer.zero_grad()

        crl_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.sa_encoder.parameters())
            + list(self.goal_encoder.parameters()),
            max_norm=0.5,
        )

        self.crl_optimizer.step()

    def _update_ppo(self) -> None:
        """CRLのQ値を用いてActorをPPOで更新する。"""
        if not self.memory:
            return

        transitions = self.memory
        device = next(self.actor.parameters()).device

        # 1. 学習データを準備
        states = torch.stack(
            [t.state for t in transitions]
        ).to(device)

        actions = torch.tensor(
            [t.action for t in transitions],
            dtype=torch.long,
            device=device,
        )

        old_log_probs = torch.tensor(
            [t.old_log_prob for t in transitions],
            dtype=torch.float32,
            device=device,
        )

        # 2. 更新済みCRLからQ値を計算
        with torch.no_grad():
            q_values = self._compute_crl_q(transitions).to(device)

            # 更新前方策による行動確率
            logits = self.actor(states)
            probabilities = torch.softmax(logits, dim=-1)

            # 状態価値 V(s,l) = Σ π(a|s)Q(s,a,l)
            values = (probabilities * q_values).sum(dim=-1)

            # 実際に選択した行動のQ値
            selected_q = q_values.gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)

            # アドバンテージ A = Q - V
            advantages = selected_q - values

            if not torch.isfinite(advantages).all():
                raise ValueError(
                    "CRLのQ値から計算したアドバンテージが不正です。"
                )

            # 全経験で一度だけ正規化
            if advantages.numel() > 1:
                advantages = (
                    advantages - advantages.mean()
                ) / (
                    advantages.std(unbiased=False) + 1e-8
                )

        # 3. PPO更新
        n = len(transitions)

        for _ in range(self.update_epochs):
            indices = torch.randperm(
                n, generator=self._generator
            )

            for start in range(0, n, self.minibatch_size):
                batch_idx = indices[
                    start:start + self.minibatch_size
                ].to(device)

                # 4. 損失計算（外部関数）
                loss = compute_ppo_loss(
                    actor=self.actor,
                    states=states[batch_idx],
                    actions=actions[batch_idx],
                    old_log_probs=old_log_probs[batch_idx],
                    advantages=advantages[batch_idx],
                    clip_coef=self.clip_coef,
                    ent_coef=self.ent_coef,
                )

                # 5. Actor更新
                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    max_norm=0.5,
                )

                self.optimizer.step()

        # 6. 学習済み経験を削除
        self.memory.clear()


    def _compute_crl_q(
        self, transitions: list[Transition]
    ) -> torch.Tensor:
        """各経験について全行動のQ値を計算する。

        Q(s,a,l) = exp(phi(s,a)^T psi(l))

        Returns:
            Q値テンソル [経験数, 行動数]
        """
        if not transitions:
            return torch.empty((0, self.CRL_NUM_ACTIONS))

        # 1. 状態と目標文を取得
        states = torch.stack([t.state for t in transitions])
        goals = [t.goal_prompt for t in transitions]

        # 2. 全行動に対応する(s,a)を生成
        n = len(transitions)
        num_actions = self.CRL_NUM_ACTIONS

        states = states.repeat_interleave(num_actions, dim=0)

        actions = torch.arange(
            num_actions, device=states.device
        ).repeat(n)

        sa_features = state_action_encoder(
            states,
            actions,
            num_actions=num_actions,
        )

        # 3. 状態・行動を特徴ベクトルに変換
        phi = self._encode_crl_features(
            self.sa_encoder, sa_features
        )

        # 4. 言語を特徴ベクトルに変換
        unique_goals = list(dict.fromkeys(goals))

        language_features = []

        for goal in unique_goals:
            if goal not in self._goal_embedding_cache:
                embedding = encode_language(goal)
                self._goal_embedding_cache[goal] = (
                    embedding.detach().flatten().cpu()
                )

            language_features.append(
                self._goal_embedding_cache[goal]
            )

        language_features = torch.stack(language_features)

        psi_unique = self._encode_crl_features(
            self.goal_encoder, language_features
        )

        goal_to_index = {
            goal: i for i, goal in enumerate(unique_goals)
        }

        indices = torch.tensor(
            [goal_to_index[goal] for goal in goals],
            device=psi_unique.device,
        )

        psi = psi_unique[indices]
        psi = psi.repeat_interleave(num_actions, dim=0)

        # 5. 内積からQ値を計算
        scores = (phi * psi).sum(dim=-1)

        q_values = torch.exp(scores)

        return q_values.reshape(n, num_actions)


    # ===========
    # ここまで
    # ===========
    
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
