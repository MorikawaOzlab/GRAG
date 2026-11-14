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

ACTIONS = ("up", "down", "left", "right")

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
        self.critic = None
        self.optimizer = None

        # memory
        self.memory: list[Transition] = []
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

        seed = settings.get("seed")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("seedは0〜2^32-1の整数またはNoneにしてください。")
        self._generator = torch.Generator(device="cpu")
        self.seed = self._generator.seed() if seed is None else seed
        self._generator.manual_seed(self.seed)

    def initialize_networks(self, observation:dict):
        # ネットワークが未初期化のときは初期化をする
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

            # PRTSの psi(l): 外部で抽出した言語特徴ベクトルを入力する。
            # 入力次元は goal_embedding_dim（既定値: self.hidden_size）。
            self.goal_encoder = nn.Sequential(
                nn.Linear(384, 512),
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

        # Qスコアの取得方法
        
        # prompt = "こんにちは"
        # self._compute_crl_q(
        #     goal_embedding=encode_language(prompt),
        #     state_action_embedding=state_action_encoder(state, action),
        #     goal_log_constant=0.0,).item()

        # Qスコアで分岐

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
            reached_goal=reached_goal,
            goal_prompt=self.generate_goal_prompt(observation),
        )

        self.memory.append(transition)

        # rollout_stepごとに学習
        if len(self.memory) >= self.rollout_steps:
            self._update_grag()

    def end_episode(self) -> None:
        """ゴール到達または最大行動回数で試行が終わるたびに呼ばれる。"""
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
        # TODO: 必要なら保存する学習データを返す。
        return {}

    # ===========
    # ここから
    # ===========

    def _update_grag(self) -> None:
        """各目標の参照正例が揃ってから、対照学習とPPO更新を行う。"""
        if not self.memory:
            return

        transitions = list(self.memory)
        if not self._has_crl_positive_references(transitions):
            # 経験を保持したまま戻り、次の経験追加後に再確認する。
            return

        # 1. 対照損失を計算し、2. 両エンコーダーを更新する。
        crl_loss = self._compute_crl_loss(transitions)
        if crl_loss is not None:
            self._update_crl_encoders(crl_loss)

        # 3. 成功経験を参照バッチとして、補正済みQを全行動について計算する。
        with torch.no_grad():
            q_inputs = self._prepare_crl_q_inputs(transitions)
            q_values = self._compute_crl_q(**q_inputs)

        # 4. 補正済みQから選択行動のアドバンテージを計算する。
        advantages = self._compute_grag_advantages(transitions, q_values)
        if advantages is None:
            return

        # 5. PPOの更新に成功した場合だけ経験を消去する。
        updated = self._update_ppo_actor(transitions, advantages)
        if updated:
            self.memory.clear()

    def _compute_crl_loss(
        self, transitions: list[Transition],
    ) -> torch.Tensor | None:
        """PRTS式(10)--(12)の双方向対照損失を返す。

        入力は時系列順の経験。PRTSの成功デモに対応する、同じ目標の
        到達成功が確認できた区間だけを使う。失敗・未完了の区間は除く。
        最後の成功遷移をT-t=0とし、時間重みはgamma**(T-t)。
        goal_promptをタスクIDとして、同じ目標文を重複除去する。

        スコアはphi(s,a)^T psi(l)。返り値はcrl_coefを掛けた
        L_sa_to_l + L_l_to_saで、両エンコーダーへの勾配を保持する。
        crl_gammaの省略時はself.gamma、crl_coefの省略時は1.0を使う。
        対象経験が2件未満、またはcrl_coef=0の場合はNoneを返す。
        重み更新・逆伝播・memoryの消去は行わない。

        参照: https://arxiv.org/html/2604.27472v1#S3.SS2
        """
        gamma = float(self.settings.get("crl_gamma", self.gamma))
        coef = float(self.settings.get("crl_coef", 1.0))
        if not 0 < gamma < 1:
            raise ValueError("crl_gammaは0より大きく1未満にしてください。")
        if not math.isfinite(coef) or coef < 0:
            raise ValueError("crl_coefは0以上の有限値にしてください。")
        if len(transitions) < 2 or coef == 0:
            return None

        prompts, remaining = self._crl_transition_metadata(transitions)
        valid = [index for index, steps in enumerate(remaining) if steps >= 0]
        if len(valid) < 2:
            return None
        batch_prompts = [prompts[index] for index in valid]
        unique_prompts = list(dict.fromkeys(batch_prompts))
        task_ids = {prompt: index for index, prompt in enumerate(unique_prompts)}

        # 言語特徴は固定。学習対象のgoal_encoderの出力はキャッシュしない。
        missing = [prompt for prompt in unique_prompts if prompt not in self._goal_embedding_cache]
        if missing:
            with torch.no_grad():
                embeddings = encode_language(missing).detach().to(device="cpu", dtype=torch.float32)
            for prompt, embedding in zip(missing, embeddings):
                self._goal_embedding_cache[prompt] = embedding.clone()
        goal_inputs = torch.stack([self._goal_embedding_cache[prompt] for prompt in unique_prompts])
        states = torch.stack([transitions[index].state for index in valid])
        actions = torch.tensor([transitions[index].action for index in valid], device=states.device)
        sa_inputs = state_action_encoder(
            states, actions, num_actions=self.CRL_NUM_ACTIONS,
            state_features=self._crl_state_features(states))
        sa_repr = self._encode_crl_features(self.sa_encoder, sa_inputs)
        goal_repr = self._encode_crl_features(self.goal_encoder, goal_inputs)
        dtype = torch.promote_types(sa_repr.dtype, goal_repr.dtype)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        scores = sa_repr.to(dtype) @ goal_repr.to(device=sa_repr.device, dtype=dtype).T
        labels = torch.tensor([task_ids[prompt] for prompt in batch_prompts], device=scores.device)
        log_weights = scores.new_tensor([remaining[index] for index in valid]) * math.log(gamma)

        # 式(11): 分母は重複を除いた全目標。各(s,a)の正例は1つ。
        sa_to_goal = (log_weights.exp() * nn.functional.cross_entropy(
            scores, labels, reduction="none")).mean()

        # 式(10),(12): 同じタスクの全(s,a)に、時間重みを正規化して割り当てる。
        positive_mask = (torch.arange(len(unique_prompts), device=scores.device)[:, None]
                         == labels[None, :])
        targets = log_weights[None, :].expand(len(unique_prompts), -1)
        targets = targets.masked_fill(~positive_mask, -torch.inf).softmax(dim=-1)
        per_goal_loss = -(targets * scores.T.log_softmax(dim=-1)).sum(dim=-1)
        # E_{i~B}なので、目標ごとの一様平均ではなく経験数に応じて平均する。
        goal_to_sa = per_goal_loss[labels].mean()
        loss = coef * (sa_to_goal + goal_to_sa)
        if not torch.isfinite(loss):
            raise FloatingPointError("PRTSの対照損失にNaN/Infがあります。")
        return loss

    def _update_crl_encoders(self, crl_loss: torch.Tensor) -> None:
        """対照損失を逆伝播し、sa_encoderとgoal_encoderを更新する。

        入力: _compute_crl_lossが返したスカラー損失。
        エンコーダー用optimizerの初期化・勾配クリップ・stepを担当する。
        Adamは初回に作成して再利用する。学習率はcrl_learning_rate
        （省略時はself.learning_rate）、勾配の最大ノルムは0.5。
        ActorとスカラーCriticは更新しない。戻り値はなし。
        """
        if not isinstance(crl_loss, torch.Tensor) or crl_loss.ndim != 0:
            raise ValueError("crl_lossにはスカラーTensorを指定してください。")
        if not crl_loss.requires_grad:
            raise ValueError("crl_lossには勾配を持つTensorを指定してください。")
        if not torch.isfinite(crl_loss):
            raise FloatingPointError("対照損失にNaN/Infがあります。")

        parameters = list(self.sa_encoder.parameters()) + list(self.goal_encoder.parameters())
        if not hasattr(self, "_grag_encoder_optimizer"):
            learning_rate = float(self.settings.get("crl_learning_rate", self.learning_rate))
            if not math.isfinite(learning_rate) or learning_rate < 0:
                raise ValueError("crl_learning_rateは0以上の有限値にしてください。")
            self._grag_encoder_optimizer = torch.optim.Adam(
                parameters, lr=learning_rate, eps=1e-5)

        self._grag_encoder_optimizer.zero_grad(set_to_none=True)
        crl_loss.backward()
        nn.utils.clip_grad_norm_(parameters, max_norm=0.5, error_if_nonfinite=True)
        self._grag_encoder_optimizer.step()

    def _has_crl_positive_references(self, transitions: list[Transition]) -> bool:
        """すべての対象目標に、成功が確認できた参照正例があればTrue。

        正例が不足する場合や経験が空の場合はFalseを返す。
        _update_gragはFalseなら学習せずに戻り、memoryを保持して待機する。
        このメソッド自体は経験や重みを変更せず、処理をブロックしない。
        """
        if not transitions:
            return False
        prompts, remaining = self._crl_transition_metadata(transitions)
        reference_prompts = {prompt for prompt, steps in zip(prompts, remaining) if steps >= 0}
        return all(prompt in reference_prompts for prompt in prompts)

    @torch.no_grad()
    def _prepare_crl_q_inputs(
        self, transitions: list[Transition],
    ) -> dict[str, torch.Tensor]:
        """経験から_compute_crl_qに渡す入力辞書を準備する。

        空でない時系列順の経験を受け取り、各目標の参照正例は存在すると仮定する。
        Nは経験数、Aは行動数、Rは成功区間内の参照経験数。
        全行動の状態・行動特徴[N,A,Dsa]、目標特徴[N,1,Dgoal]、
        参照内積[N,1,R]、残りステップ数[R]、正例マスク[N,1,R]を返す。
        返り値は_compute_crl_q(**inputs)として渡せる。勾配は保持しない。
        補正定数とQ値の計算は_compute_crl_qが担当する。
        """
        prompts, remaining = self._crl_transition_metadata(transitions)
        valid = [index for index, steps in enumerate(remaining) if steps >= 0]
        unique_prompts = list(dict.fromkeys(prompts))
        task_ids = {prompt: index for index, prompt in enumerate(unique_prompts)}
        missing = [prompt for prompt in unique_prompts if prompt not in self._goal_embedding_cache]
        if missing:
            embeddings = encode_language(missing).detach().to(device="cpu", dtype=torch.float32)
            for prompt, embedding in zip(missing, embeddings):
                self._goal_embedding_cache[prompt] = embedding.clone()
        goal_inputs = torch.stack([self._goal_embedding_cache[prompt] for prompt in unique_prompts])
        labels = torch.tensor([task_ids[prompt] for prompt in prompts], dtype=torch.long)

        states = torch.stack([transition.state for transition in transitions])
        features = self._crl_state_features(states)
        reference_indices = torch.tensor(valid, device=states.device)
        reference_actions = torch.tensor([transitions[index].action for index in valid],
                                         device=states.device)
        reference_inputs = state_action_encoder(
            states[reference_indices], reference_actions, num_actions=self.CRL_NUM_ACTIONS,
            state_features=features[reference_indices])
        sa_repr = self._encode_crl_features(self.sa_encoder, reference_inputs)
        goal_repr = self._encode_crl_features(self.goal_encoder, goal_inputs)
        dtype = torch.promote_types(sa_repr.dtype, goal_repr.dtype)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        reference_scores = (goal_repr.to(device=sa_repr.device, dtype=dtype)
                            @ sa_repr.to(dtype).T)
        reference_labels = labels[valid].to(reference_scores.device)
        positive_mask = (torch.arange(len(unique_prompts), device=reference_scores.device)[:, None]
                         == reference_labels[None, :])

        one_hot = torch.eye(self.CRL_NUM_ACTIONS, device=features.device, dtype=features.dtype)
        sa_inputs = torch.cat((features[:, None, :].expand(-1, self.CRL_NUM_ACTIONS, -1),
                               one_hot.expand(len(transitions), -1, -1)), dim=-1)
        score_labels = labels.to(reference_scores.device)
        return {
            "state_action_embedding": sa_inputs,
            "goal_embedding": goal_inputs[labels.to(goal_inputs.device)][:, None, :],
            "reference_log_scores": reference_scores[score_labels][:, None, :],
            "reference_steps_to_goal": reference_scores.new_tensor([remaining[index] for index in valid]),
            "reference_positive_mask": positive_mask[score_labels][:, None, :],
        }

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
        割引率はcrl_gamma（省略時はself.gamma）を使う。
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
            gamma = float(self.settings.get("crl_gamma", self.gamma))
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

    @torch.no_grad()
    def _compute_grag_advantages(
        self,
        transitions: list[Transition],
        q_values: torch.Tensor,
    ) -> torch.Tensor | None:
        """各経験で実際に選んだ行動のアドバンテージを返す。

        入力: N件の経験と、同じ順序の補正済みQ値[N, CRL_NUM_ACTIONS]。
        現在のActorの方策確率からV = sum_a pi(a|s) * Q(s,a)を計算し、
        経験のactionに対応するQからVを引く。Q値は再計算しない。
        戻り値: 勾配を持たない未標準化のアドバンテージ[N]。経験が空ならNone。
        Actorの評価はself.minibatch_size件ずつ行う。
        """
        if not transitions:
            return None
        if not isinstance(q_values, torch.Tensor) or q_values.shape != (len(transitions), self.CRL_NUM_ACTIONS):
            raise ValueError("q_valuesの形状は[経験数, 行動数]にしてください。")
        if not q_values.is_floating_point():
            raise ValueError("q_valuesには浮動小数点Tensorを指定してください。")
        if not torch.isfinite(q_values).all():
            raise FloatingPointError("Q値にNaN/Infがあります。")
        if self.minibatch_size < 1:
            raise ValueError("ppo_minibatch_sizeは1以上にしてください。")

        advantages = q_values.new_empty(len(transitions))
        for start in range(0, len(transitions), self.minibatch_size):
            batch = transitions[start:start + self.minibatch_size]
            states = torch.stack([transition.state for transition in batch])
            logits = self._encode_crl_features(self.actor, states)
            scores = q_values[start:start + len(batch)]
            action_probs = logits.softmax(dim=-1).to(scores)
            values = (action_probs * scores).sum(dim=-1)
            actions = torch.tensor([transition.action for transition in batch],
                                   dtype=torch.long, device=scores.device)
            q_taken = scores.gather(1, actions[:, None]).squeeze(-1)
            advantages[start:start + len(batch)] = q_taken - values

        if not torch.isfinite(advantages).all():
            raise FloatingPointError("アドバンテージにNaN/Infがあります。")
        return advantages

    def _update_ppo_actor(
        self,
        transitions: list[Transition],
        advantages: torch.Tensor,
    ) -> bool:
        """固定したアドバンテージで、PPOのActor更新を行う。

        入力: N件の経験と、同じ順序のアドバンテージ[N]。
        アドバンテージの標準化後、経験のold_log_probを使い、
        複数epoch・ミニバッチのclipped lossとエントロピー項で更新する。
        _update_ppoのActor更新ループを使い、Actor専用Adamを再利用する。
        学習率はself.learning_rate、勾配の最大ノルムは0.5。
        GAEの再計算やCritic・エンコーダーの更新、memoryの消去は行わない。
        戻り値: 更新を実行したらTrue、実行しなかったらFalse。
        """
        if not transitions or self.update_epochs < 1:
            return False
        if self.minibatch_size < 1:
            raise ValueError("ppo_minibatch_sizeは1以上にしてください。")
        if not isinstance(advantages, torch.Tensor) or advantages.shape != (len(transitions),):
            raise ValueError("advantagesの形状は[経験数]にしてください。")
        if not advantages.is_floating_point():
            raise ValueError("advantagesには浮動小数点Tensorを指定してください。")
        if not torch.isfinite(advantages).all():
            raise FloatingPointError("アドバンテージにNaN/Infがあります。")

        # バッチ全体で一度だけ標準化し、全epochで同じ値を使う。
        fixed_advantages = advantages.detach()
        if len(transitions) > 1:
            fixed_advantages = ((fixed_advantages - fixed_advantages.mean())
                                / (fixed_advantages.std(unbiased=False) + 1e-8))
        states = torch.stack([transition.state.detach() for transition in transitions])
        actions = torch.tensor([transition.action for transition in transitions],
                               dtype=torch.long, device=states.device)
        parameters = list(self.actor.parameters())
        old_log_probs = torch.tensor([transition.old_log_prob for transition in transitions],
                                     dtype=parameters[0].dtype, device=states.device)
        if not torch.isfinite(old_log_probs).all():
            raise FloatingPointError("更新前の行動対数確率にNaN/Infがあります。")
        if not hasattr(self, "_grag_actor_optimizer"):
            learning_rate = float(self.learning_rate)
            if not math.isfinite(learning_rate) or learning_rate < 0:
                raise ValueError("ppo_learning_rateは0以上の有限値にしてください。")
            self._grag_actor_optimizer = torch.optim.Adam(parameters, lr=learning_rate, eps=1e-5)

        for _ in range(self.update_epochs):
            indices = torch.randperm(len(transitions), generator=self._generator)
            for start in range(0, len(transitions), self.minibatch_size):
                batch_indices = indices[start:start + self.minibatch_size]
                state_indices = batch_indices.to(states.device)
                logits = self._encode_crl_features(self.actor, states[state_indices])
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[state_indices].to(logits.device))
                entropy = dist.entropy().mean()
                ratio = (new_log_probs - old_log_probs[state_indices].to(logits)).exp()
                batch_advantages = fixed_advantages[batch_indices.to(fixed_advantages.device)].to(logits)

                pg_loss1 = -batch_advantages * ratio
                pg_loss2 = -batch_advantages * ratio.clamp(1.0 - self.clip_coef, 1.0 + self.clip_coef)
                actor_loss = torch.maximum(pg_loss1, pg_loss2).mean()
                loss = actor_loss - self.ent_coef * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("PPOのActor損失にNaN/Infがあります。")

                self._grag_actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, max_norm=0.5, error_if_nonfinite=True)
                self._grag_actor_optimizer.step()
        return True

    def _crl_transition_metadata(
        self, transitions: list[Transition],
    ) -> tuple[list[str], list[int]]:
        """目標文と成功到達までの残りステップ数を返す。対象外の経験は-1。"""
        prompts = []
        for transition in transitions:
            prompt = transition.goal_prompt
            if prompt is None:
                state = transition.state
                goal_positions = (state[1] == 1).nonzero()
                if len(goal_positions) != 1:
                    raise ValueError("stateのゴールチャネルにはゴールを1つ指定してください。")
                prompt = self.generate_goal_prompt({
                    "rows": state.shape[-2], "cols": state.shape[-1],
                    "goal": goal_positions[0].tolist(),
                    "obstacles": (state[0] == 1).nonzero().tolist(),
                })
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("goal_promptは空でない文字列にしてください。")
            prompts.append(prompt)

        # ロールアウトの末尾や失敗した試行を、到達成功と取り違えない。
        remaining = [-1] * len(transitions)
        success_index = None
        success_prompt = None
        for index in reversed(range(len(transitions))):
            transition = transitions[index]
            if transition.reached_goal or transition.episode_end:
                success_index = index if transition.reached_goal else None
                success_prompt = prompts[index] if transition.reached_goal else None
            if success_index is not None:
                if prompts[index] == success_prompt:
                    remaining[index] = success_index - index
                else:
                    success_index = None
        return prompts, remaining


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
