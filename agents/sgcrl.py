"""環境に依存しない、状態ゴール版SGCRLのネットワークと学習インターフェース。"""

import base64
from collections import deque
import io
import math
import random
from time import perf_counter

import torch
from torch import nn


class _Policy(nn.Module):
    """特徴ベクトルから離散方策または[-1, 1]の連続方策を返す。"""

    def __init__(self, network: nn.Module, discrete: bool):
        super().__init__()
        self.network = network
        self.discrete = discrete
        self.graph_capture = False

    def forward(self, state_goal: torch.Tensor) -> torch.distributions.Distribution:
        output = self.network(state_goal)
        if self.discrete:
            return torch.distributions.Categorical(logits=output, validate_args=False if self.graph_capture else None)
        mean, raw_scale = output.chunk(2, dim=-1)
        scale = nn.functional.softplus(raw_scale) + 1e-6
        base = torch.distributions.Independent(
            torch.distributions.Normal(mean, scale), 1)
        return torch.distributions.TransformedDistribution(
            base, [torch.distributions.TanhTransform(cache_size=1)])


class SGCRL:
    """状態・ゴール・行動の特徴次元を指定して使うSGCRLの基盤。"""

    def __init__(self, settings: dict):
        """学習パラメータを受け取る。モデルや内部状態の初期化はここに追加する。"""
        self.settings = settings
        self.actor = None
        self.sa_encoder = None
        self.goal_encoder = None
        self._network_spec = None
        seed = settings.get("seed")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("seedは0〜2^32-1の整数またはNoneにしてください。")
        self.seed = torch.Generator().seed() if seed is None else seed
        self._random = random.Random(self.seed)
        requested = settings.get("crl_device", "cuda")
        self.device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                                   else "cpu" if requested == "auto" else requested)
        self._action_generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self._replay_generator = torch.Generator(device=self.device).manual_seed(self.seed ^ 0x5A17)
        self.replay_cache_bytes = int(float(settings.get("crl_replay_cache_mb", 8192)) * 1024**2)
        if self.replay_cache_bytes < 0:
            raise ValueError("crl_replay_cache_mbは0以上にしてください。")
        self._replay_cache = None
        self._replay_cache_key = None
        self.gamma = float(settings.get("crl_gamma", 0.99))
        self.batch_size = settings.get("crl_batch_size", 256)
        self.min_replay_size = settings.get("crl_min_replay_size", 10_000)
        self.max_replay_size = settings.get("crl_max_replay_size", 1_000_000)
        self.updates_per_step = float(settings.get("crl_updates_per_step", 1.0))
        self.learning_rate = float(settings.get("crl_learning_rate", 3e-4))
        self.target_entropy = float(settings.get("crl_target_entropy", 0.0))
        self.random_goals = float(settings.get("crl_random_goals", 0.5))
        self.share_actor_features = settings.get("crl_share_actor_features", True)
        if type(self.share_actor_features) is not bool:
            raise ValueError("crl_share_actor_featuresはboolにしてください。")
        self.hidden_size = settings.get("crl_hidden_size", 256)
        self.repr_dim = settings.get("crl_repr_dim", 64)
        for name, value in (("crl_batch_size", self.batch_size),
                            ("crl_hidden_size", self.hidden_size),
                            ("crl_repr_dim", self.repr_dim),
                            ("crl_min_replay_size", self.min_replay_size),
                            ("crl_max_replay_size", self.max_replay_size)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name}は1以上の整数にしてください。")
        if self.batch_size < 2 or self.max_replay_size < self.min_replay_size:
            raise ValueError("batch_sizeは2以上、max_replay_sizeはmin_replay_size以上が必要です。")
        if not 0 <= self.gamma < 1 or self.random_goals not in (0.0, 0.5, 1.0):
            raise ValueError("crl_gammaは[0, 1)、crl_random_goalsは0、0.5、1にしてください。")
        if (not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not math.isfinite(self.updates_per_step) or self.updates_per_step < 0
                or not math.isfinite(self.target_entropy)):
            raise ValueError("学習率・更新比率・目標エントロピーの設定が不正です。")
        self._episodes = deque()
        self._episode = []
        self._replay_size = 0
        self._steps = 0
        self._updates = 0
        self.metrics = {}
        self._update_progress = {"phase": "collecting", "completed": 0, "total": 0}
        self._update_rate = 0.0
        self._cuda_graph = None
        self._cuda_graph_disabled = False
        self._cuda_graph_error = None
        self._graph_inline_sampling = False

    def initialize_networks(
        self, state_dim: int, goal_dim: int, action_dim: int, *, discrete: bool = False,
    ) -> None:
        """SGCRLの方策と対照Criticのphi(s,a)、psi(g)を一度だけ構築する。

        既定値は論文Appendix Bの隠れ層(256, 256)、表現次元64。
        ReLUとエンコーダーのXavier uniform初期化は著者実装に従う。
        https://arxiv.org/html/2408.05804v1#A2
        https://github.com/graliuce/sgcrl/blob/main/contrastive/networks.py

        state_dimとgoal_dimは外部で前処理した状態・ゴールの特徴次元。
        action_dimは連続行動の次元数、discrete=Trueでは離散行動数。
        actorは結合ベクトル[s, g]から分布を返す。連続方策はtanh変換した
        対角Gaussian、離散方策はCategorical。行動範囲の変換は環境側で行う。
        phiは[s, a]、psiはgを受け取る。離散行動aは外部でone-hotにする。
        各ネットワークは末尾を特徴次元とし、任意のバッチ次元を保持する。
        観測の抽出・平坦化・画像特徴抽出などは環境側で行う。
        Criticは未正規化の内積phi(s,a)^T psi(g)であり、独立したV網は作らない。
        Adamと適応エントロピー係数も初期化する。外部のPyTorch乱数状態は保持する。
        """
        if any(type(size) is not int or size < 1
               for size in (state_dim, goal_dim, action_dim)):
            raise ValueError("state_dim、goal_dim、action_dimは1以上の整数にしてください。")
        if type(discrete) is not bool:
            raise ValueError("discreteはboolにしてください。")
        spec = (state_dim, goal_dim, action_dim, discrete)
        if self._network_spec is not None:
            if spec != self._network_spec:
                raise ValueError("初期化済みネットワークと異なる次元・行動種別は使えません。")
            return

        def mlp(input_dim: int, output_dim: int, *, actor: bool = False) -> nn.Sequential:
            layers = []
            dimensions = (input_dim, self.hidden_size, self.hidden_size, output_dim)
            for index, (in_features, out_features) in enumerate(zip(dimensions, dimensions[1:])):
                layer = nn.Linear(in_features, out_features)
                if actor:
                    # 著者実装の方策MLP: VarianceScaling(1, fan_in, uniform)。
                    nn.init.uniform_(layer.weight, -(3 / in_features) ** 0.5,
                                     (3 / in_features) ** 0.5)
                else:
                    nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
                layers.append(layer)
                if index < 2:
                    layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            actor = _Policy(mlp(state_dim + goal_dim,
                                action_dim if discrete else 2 * action_dim, actor=True),
                            discrete)
            sa_encoder = mlp(state_dim + action_dim, self.repr_dim)
            goal_encoder = mlp(goal_dim, self.repr_dim)

        self.actor = actor.to(self.device)
        self.sa_encoder = sa_encoder.to(self.device)
        self.goal_encoder = goal_encoder.to(self.device)
        self._network_spec = spec
        fused = self.device.type == "cuda"
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate, eps=1e-7, fused=fused, capturable=fused)
        self.encoder_optimizer = torch.optim.Adam(self._encoder_parameters(), lr=self.learning_rate, eps=1e-7, fused=fused, capturable=fused)
        self.log_alpha = nn.Parameter(torch.zeros((), device=self.device))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate, eps=1e-7, fused=fused, capturable=fused)

    def _sample_action(self, distribution, *, reparameterize=False):
        """エージェント専用乱数を使い、プロセス全体の乱数を変更しない。"""
        if isinstance(distribution, torch.distributions.Categorical):
            probs = distribution.probs
            return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1,
                                     generator=self._action_generator).reshape(probs.shape[:-1])
        normal = distribution.base_dist.base_dist
        noise = torch.randn(normal.loc.shape, device=normal.loc.device,
                            dtype=normal.loc.dtype, generator=self._action_generator)
        action = normal.loc + normal.scale * noise
        for transform in distribution.transforms:
            action = transform(action)
        return action if reparameterize else action.detach()

    def choose_action(
        self, state: torch.Tensor, goal: torch.Tensor, greedy: bool = False,
    ) -> torch.Tensor:
        """Actorの分布から行動を返す。greedy=Trueでは乱数を使わない。

        離散方策は最大確率の行動インデックス、連続方策はGaussianの
        平均をtanh変換した行動を返す。greedy=Falseでは初期収集中は一様ランダム、
        その後はActorの分布からサンプルする。
        入力の末尾は特徴次元で、状態とゴールのバッチ次元は揃える。
        返り値はActorと同じdevice上の、勾配を保持しないTensor。
        """
        if self.actor is None:
            raise RuntimeError("initialize_networks()でネットワークを初期化してください。")
        reference = next(self.actor.parameters())
        with torch.no_grad():
            if not greedy and self._steps < self.min_replay_size:
                # 初期ランダム収集で使わないActor推論や特徴のGPU転送を省く。
                if self.actor.discrete:
                    return torch.randint(self._network_spec[2], state.shape[:-1],
                                         device=self.device, generator=self._action_generator)
                return 2 * torch.rand((*state.shape[:-1], self._network_spec[2]),
                                      device=self.device, dtype=reference.dtype,
                                      generator=self._action_generator) - 1
            state = state.to(device=reference.device, dtype=reference.dtype)
            goal = goal.to(device=reference.device, dtype=reference.dtype)
            distribution = self.actor(torch.cat((state, goal), dim=-1))
            if not greedy:
                return self._sample_action(distribution)
            if self.actor.discrete:
                return distribution.logits.argmax(dim=-1)
            return distribution.base_dist.mean.tanh()

    def action_probabilities(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """乱数・学習状態を変更せず、離散Actorの確率を返す。"""
        if self.actor is None or not self.actor.discrete:
            raise NotImplementedError("初期化済みの離散方策が必要です。")
        reference = next(self.actor.parameters())
        with torch.no_grad():
            features = torch.cat((state.to(reference), goal.to(reference)), -1)
            return self.actor(features).probs

    def learn(self, state: torch.Tensor, action: torch.Tensor,
              next_state: torch.Tensor, goal: torch.Tensor, *,
              terminated: bool = False, truncated: bool = False,
              next_goal: torch.Tensor | None = None) -> None:
        """1遷移を記録する。next_goalは次状態の達成ゴール表現（目標指示とは別）。

        同一次元ならnext_stateを達成ゴールとして使える。異なる表現では
        環境側で射影したnext_goalを渡す。報酬・成功ラベルは損失に使わない。
        終端または時間制限でエピソードを確定して学習する。
        """
        if self._network_spec is None:
            raise RuntimeError("initialize_networks()を先に呼んでください。")
        state_dim, goal_dim, action_dim, discrete = self._network_spec
        if next_goal is None:
            if state_dim != goal_dim:
                raise ValueError("状態とゴールの次元が異なる場合はnext_goalを指定してください。")
            next_goal = next_state
        tensors = [torch.as_tensor(x).detach().cpu().float().clone()
                   for x in (state, next_state, goal, next_goal)]
        for tensor, size in zip(tensors, (state_dim, state_dim, goal_dim, goal_dim)):
            if tensor.shape != (size,) or not torch.isfinite(tensor).all():
                raise ValueError("learn()には有限な1次元の特徴ベクトルを渡してください。")
        state, next_state, goal, achieved = tensors
        action = torch.as_tensor(action).detach().cpu().clone()
        if discrete:
            if action.ndim != 0 or action.is_floating_point() or not 0 <= action.item() < action_dim:
                raise ValueError("離散行動は範囲内の整数スカラーにしてください。")
            action = action.long()
        elif action.shape != (action_dim,) or not torch.isfinite(action).all() or (action.abs() > 1).any():
            raise ValueError("連続行動は[-1, 1]の行動ベクトルにしてください。")
        self._episode.append((state, action, achieved))
        self._steps += 1
        if terminated or truncated:
            self.end_episode()

    def end_episode(self) -> None:
        """成功・失敗とも保存し、収集遷移数に比例した回数だけ学習する。二重呼出しは無操作。"""
        if not self._episode:
            return
        collected = len(self._episode)
        episode = tuple(torch.stack(items) for items in zip(*self._episode[-self.max_replay_size:]))
        self._episode.clear()
        self._episodes.append(episode)
        if self._graph_inline_sampling:
            # Captured sampling points directly at the previous replay cache.
            self._cuda_graph = None
            self._graph_metrics = None
            self._graph_inline_sampling = False
        self._replay_cache_key = None
        self._replay_cache = None
        self._replay_size += len(episode[0])
        while self._replay_size > self.max_replay_size:
            oldest = self._episodes.popleft()
            removed = min(self._replay_size - self.max_replay_size, len(oldest[0]))
            self._replay_size -= removed
            if removed < len(oldest[0]):
                self._episodes.appendleft(tuple(tensor[removed:] for tensor in oldest))
        if self._replay_size < self.min_replay_size:
            return
        count = collected * self.updates_per_step
        updates = int(count) + (self._random.random() < count % 1)
        started = perf_counter()
        self._update_progress = {"phase": "updating", "completed": 0, "total": updates}
        try:
            for index in range(updates):
                self.update()
                self._update_progress = {"phase": "updating", "completed": index + 1, "total": updates}
        finally:
            elapsed = perf_counter() - started
            if updates:
                self._update_rate = self._update_progress["completed"] / max(elapsed, 1e-9)
            self._update_progress = {**self._update_progress, "phase": "collecting"}

    def runtime_status(self):
        return {**self._update_progress, "device": str(self.device), "updates": self._updates,
                "updates_per_second": self._update_rate, "batch_size": self.batch_size,
                "replay_size": self._replay_size, "replay_cached": self._replay_cache is not None,
                "cuda_graph": self._cuda_graph is not None,
                "cuda_graph_error": self._cuda_graph_error}

    def sample_batch(self):
        """エピソードと時点を一様抽出し、将来ゴールをgamma**(delta-1)で抽出。

        エピソード境界は越えない。有限軌跡内で正規化した幾何分布を使い、
        最も近い将来状態はs[t+1]。対角正例以外のバッチ要素が負例となる。
        """
        if not self._episodes:
            raise RuntimeError("完了したエピソードがありません。")
        key = (len(self._episodes), self._replay_size, id(self._episodes[0]), id(self._episodes[-1]))
        if key != self._replay_cache_key:
            self._replay_cache_key = key
            required = sum(tensor.numel() * tensor.element_size()
                           for episode in self._episodes for tensor in episode)
            self._replay_cache = None
            enough_memory = True
            if self.device.type == "cuda":
                free_bytes, _ = torch.cuda.mem_get_info(self.device)
                enough_memory = required <= max(0, free_bytes - 1024**3)
            if required <= self.replay_cache_bytes and enough_memory:
                episodes = tuple(self._episodes)
                features = []
                try:
                    for i in range(3):
                        features.append(torch.cat([episode[i] for episode in episodes]).to(self.device))
                    lengths = torch.tensor([len(episode[0]) for episode in episodes], device=self.device)
                    starts = lengths.cumsum(0) - lengths
                    self._replay_cache = (*features, lengths, starts)
                except torch.cuda.OutOfMemoryError:
                    features.clear()
                    if self.device.type != "cuda":
                        raise
                    torch.cuda.empty_cache()
        if self._replay_cache is not None:
            states, actions, goals, lengths, starts = self._replay_cache
            generator = self._replay_generator
            episode_ids = torch.randint(len(lengths), (self.batch_size,), device=self.device, generator=generator)
            lengths = lengths[episode_ids]
            times = (torch.rand(self.batch_size, device=self.device, generator=generator,
                                dtype=torch.float64) * lengths).long()
            remaining = lengths - times
            if self.gamma == 0:
                offsets = torch.zeros_like(times)
            else:
                uniform = torch.rand(self.batch_size, device=self.device, generator=generator,
                                     dtype=torch.float64)
                log_gamma = math.log(self.gamma)
                mass = -torch.expm1(remaining * log_gamma)
                offsets = (torch.log1p(-uniform * mass) / log_gamma).long()
                offsets = torch.minimum(offsets, remaining - 1)
            indices = starts[episode_ids] + times
            return states[indices], actions[indices], goals[indices + offsets]
        # 上限を越える巨大なバッファーは複製せず、従来のCPU抽出を使う。
        states, actions, goals = [], [], []
        for _ in range(self.batch_size):
            episode = self._episodes[self._random.randrange(len(self._episodes))]
            t = self._random.randrange(len(episode[0]))
            remaining = len(episode[0]) - t
            offset = (0 if self.gamma == 0 else min(remaining - 1, int(
                math.log1p(-self._random.random() * (1 - self.gamma ** remaining))
                / math.log(self.gamma))))
            states.append(episode[0][t])
            actions.append(episode[1][t])
            goals.append(episode[2][t + offset])
        return tuple(torch.stack(items).to(self.device) for items in (states, actions, goals))

    def _encoder_parameters(self):
        return list(self.sa_encoder.parameters()) + list(self.goal_encoder.parameters())

    def encode_state_actions(self, states, actions):
        if self.actor.discrete:
            actions = nn.functional.one_hot(actions.long(), self._network_spec[2]).to(states)
        return self.sa_encoder(torch.cat((states, actions), -1))

    def encode_goals(self, goals):
        return self.goal_encoder(goals)

    def state_goal_scores(self, states, actions, goals):
        """全組合せの未正規化内積[B, B]。"""
        return self.encode_state_actions(states, actions) @ self.encode_goals(goals).T

    def _additional_encoder_losses(self, states, actions, goals):
        """言語対応などの派生クラス用フック。"""
        return states.new_zeros(())

    def critic_loss(self, states, actions, goals):
        """論文式(3): 行方向InfoNCE + 0.01 * LogSumExp**2。"""
        logits = self.state_goal_scores(states, actions, goals)
        labels = torch.arange(len(states), device=states.device)
        return (nn.functional.cross_entropy(logits, labels)
                + 0.01 * logits.logsumexp(-1).square().mean()
                + self._additional_encoder_losses(states, actions, goals))

    def actor_loss(self, states, goals):
        """論文式(4)。離散行動は全行動で期待値を厳密に計算する。"""
        original_states, original_goals = states, goals
        if self.random_goals == 0.5:
            states = torch.cat((states, states))
            goals = torch.cat((goals, goals.roll(1, 0)))
        elif self.random_goals == 1.0:
            goals = goals.roll(1, 0)
        distribution = self.actor(torch.cat((states, goals), -1))
        alpha = self.log_alpha.exp().detach()
        if self.actor.discrete:
            with torch.no_grad():
                if self.random_goals == 0.5 and self._can_share_actor_features():
                    phi = self._all_action_representations(original_states)
                    psi = self.encode_goals(original_goals)
                    q = torch.cat(((phi * psi[:, None, :]).sum(-1),
                                   (phi * psi.roll(1, 0)[:, None, :]).sum(-1)))
                else:
                    q = (self._all_action_representations(states)
                         * self.encode_goals(goals)[:, None, :]).sum(-1)
            entropy = distribution.entropy()
            loss = -(distribution.probs * q).sum(-1) - alpha * entropy
        else:
            parameters = self._encoder_parameters()
            flags = [parameter.requires_grad for parameter in parameters]
            try:
                for parameter in parameters:
                    parameter.requires_grad_(False)
                actions = self._sample_action(distribution, reparameterize=True)
                q = (self.encode_state_actions(states, actions) * self.encode_goals(goals)).sum(-1)
                entropy = -distribution.log_prob(actions)
                loss = -q - alpha * entropy
            finally:
                for parameter, flag in zip(parameters, flags):
                    parameter.requires_grad_(flag)
        return loss.mean(), entropy.mean()

    def _can_share_actor_features(self):
        return (self.share_actor_features
                and type(self).encode_state_actions is SGCRL.encode_state_actions
                and type(self).encode_goals is SGCRL.encode_goals
                and all(isinstance(network, nn.Sequential) and len(network) > 0 and
                        all(type(layer) in (nn.Linear, nn.ReLU) for layer in network)
                        for network in (self.sa_encoder, self.goal_encoder)))

    def _all_action_representations(self, states):
        count = self._network_spec[2]
        if self._can_share_actor_features() and type(self.sa_encoder[0]) is nn.Linear:
            first = self.sa_encoder[0]
            # W_s*s + W_a*one_hot(a) + b: 状態側の大きな行列積を行動ごとに繰り返さない。
            hidden = nn.functional.linear(states, first.weight[:, :-count], first.bias)
            hidden = hidden[:, None, :] + first.weight[:, -count:].T[None, :, :]
            for index, layer in enumerate(self.sa_encoder):
                if index:
                    hidden = layer(hidden)
            return hidden
        all_states = states[:, None, :].expand(-1, count, -1)
        all_actions = torch.arange(count, device=self.device).expand(len(states), -1)
        return self.encode_state_actions(all_states, all_actions)

    def _capture_cuda_update(self, batch, *, inline_sampling=False):
        """Capture one fixed-shape discrete update after optimizer state is initialized."""
        self._graph_batch = None if inline_sampling else tuple(x.clone() for x in batch)
        graph = torch.cuda.CUDAGraph()
        self.actor.graph_capture = True
        try:
            with torch.cuda.graph(graph):
                states, actions, goals = self.sample_batch() if inline_sampling else self._graph_batch
                critic_loss = self.critic_loss(states, actions, goals)
                self.encoder_optimizer.zero_grad(set_to_none=False)
                critic_loss.backward()
                self.encoder_optimizer.step()
                self.encoder_optimizer.zero_grad(set_to_none=False)
                actor_loss, entropy = self.actor_loss(states, goals)
                self.actor_optimizer.zero_grad(set_to_none=False)
                actor_loss.backward()
                self.actor_optimizer.step()
                alpha_loss = self.log_alpha.exp() * (entropy.detach() - self.target_entropy)
                self.alpha_optimizer.zero_grad(set_to_none=False)
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self._graph_metrics = torch.stack((critic_loss.detach(), actor_loss.detach(),
                                                   entropy.detach(), self.log_alpha.detach().exp()))
        finally:
            self.actor.graph_capture = False
        self._cuda_graph = graph
        self._graph_inline_sampling = inline_sampling

    def update(self):
        # Graph mode is limited to the built-in discrete implementation; extensions
        # and continuous policies keep their existing execution and RNG semantics.
        eligible = (self.device.type == "cuda" and self.actor.discrete
                    and type(self) is SGCRL and self.settings.get("crl_cuda_graph", True)
                    and self._updates >= 3 and not self._cuda_graph_disabled)
        generator_before = (self._replay_generator.get_state() if eligible and self._cuda_graph is None else None)
        batch = None if eligible and self._cuda_graph is not None and self._graph_inline_sampling else self.sample_batch()
        if eligible:
            if self._cuda_graph is None:
                generator_after = self._replay_generator.get_state()
                inline_sampling = self._replay_cache is not None
                try:
                    if inline_sampling:
                        self._replay_generator.set_state(generator_before)
                    self._capture_cuda_update(batch, inline_sampling=inline_sampling)
                except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
                    if inline_sampling:
                        self._replay_generator.set_state(generator_after)
                    self._cuda_graph_disabled = True
                    self._cuda_graph_error = str(error)
                    self._graph_batch = None
            if self._cuda_graph is not None:
                if not self._graph_inline_sampling:
                    for target, source in zip(self._graph_batch, batch):
                        target.copy_(source)
                self._cuda_graph.replay()
                values = self._graph_metrics.tolist()
                if not all(math.isfinite(value) for value in values):
                    raise FloatingPointError("CUDA Graph更新の損失または統計値が有限ではありません。")
                self._updates += 1
                self.metrics = dict(zip(("critic_loss", "actor_loss", "entropy", "alpha"), values))
                self.metrics["replay_size"] = self._replay_size
                return self.metrics
        states, actions, goals = batch
        critic_loss = self.critic_loss(states, actions, goals)
        self.encoder_optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("Critic損失が有限ではありません。")
        critic_loss.backward()
        self.encoder_optimizer.step()
        self.encoder_optimizer.zero_grad(set_to_none=True)
        actor_loss, entropy = self.actor_loss(states, goals)
        if not torch.isfinite(actor_loss):
            raise FloatingPointError("Actor損失が有限ではありません。")
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        alpha_loss = self.log_alpha.exp() * (entropy.detach() - self.target_entropy)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self._updates += 1
        # GPUの小さな値を4回同期転送せず、一度に取得する。
        values = torch.stack((critic_loss.detach(), actor_loss.detach(), entropy.detach(),
                              self.log_alpha.detach().exp())).tolist()
        self.metrics = dict(zip(("critic_loss", "actor_loss", "entropy", "alpha"), values))
        self.metrics["replay_size"] = self._replay_size
        return self.metrics

    def export(self) -> dict:
        """保存用のJSONに変換可能な辞書を返す。

        保存する学習データがなければ空の辞書を返す。
        """
        return self._serialize(include_replay=False)

    def checkpoint(self):
        return self._serialize(include_replay=True)

    def _serialize(self, *, include_replay):
        payload = {"settings": self.settings, "spec": self._network_spec,
                   "steps": self._steps, "updates": self._updates, "metrics": self.metrics,
                   "seed": self.seed, "random": self._random.getstate(),
                   "action_random": self._action_generator.get_state(),
                   "action_random_device": self._action_generator.device.type,
                   "replay_random": self._replay_generator.get_state()}
        if self.actor is not None:
            payload.update({name: getattr(self, name).state_dict() for name in
                            ("actor", "sa_encoder", "goal_encoder", "actor_optimizer",
                             "encoder_optimizer", "alpha_optimizer")})
            payload["log_alpha"] = self.log_alpha.detach()
        if include_replay:
            payload.update(episodes=list(self._episodes), episode=self._episode,
                           replay_size=self._replay_size)
        stream = io.BytesIO()
        torch.save(payload, stream)
        return {"format": "sgcrl-v1", "data": base64.b64encode(stream.getvalue()).decode("ascii")}

    @classmethod
    def from_export(cls, value):
        agent = cls({})
        agent.load_checkpoint(value)
        return agent

    def load_checkpoint(self, value):
        if value.get("format") != "sgcrl-v1":
            raise ValueError("SGCRLチェックポイント形式が異なります。")
        payload = torch.load(io.BytesIO(base64.b64decode(value["data"])),
                             map_location="cpu", weights_only=True)
        self.__init__(payload["settings"])
        self.seed = payload["seed"]
        if payload["spec"] is not None:
            state_dim, goal_dim, action_dim, discrete = payload["spec"]
            self.initialize_networks(state_dim, goal_dim, action_dim, discrete=discrete)
            for name in ("actor", "sa_encoder", "goal_encoder", "actor_optimizer",
                         "encoder_optimizer", "alpha_optimizer"):
                getattr(self, name).load_state_dict(payload[name])
            for optimizer in (self.actor_optimizer, self.encoder_optimizer, self.alpha_optimizer):
                for group in optimizer.param_groups:
                    group["fused"] = self.device.type == "cuda"
                    group["capturable"] = self.device.type == "cuda"
                    group["foreach"] = None
                for state in optimizer.state.values():
                    if "step" in state:
                        state["step"] = state["step"].to(self.device)
            with torch.no_grad():
                self.log_alpha.copy_(payload["log_alpha"])
        self._steps, self._updates = payload["steps"], payload["updates"]
        self.metrics = payload["metrics"]
        self._random.setstate(payload["random"])
        source_device = payload.get("action_random_device")
        if source_device is None:
            source_device = ("cpu" if payload["action_random"].numel() ==
                             torch.Generator().get_state().numel() else "cuda")
        if source_device == self.device.type:
            self._action_generator.set_state(payload["action_random"])
            if "replay_random" in payload:
                self._replay_generator.set_state(payload["replay_random"])
        else:
            self._action_generator.manual_seed(self.seed)
            self._replay_generator.manual_seed(self.seed ^ 0x5A17)
        self._episodes = deque(payload.get("episodes", []))
        self._episode = payload.get("episode", [])
        self._replay_size = payload.get("replay_size", 0)

    def evaluation_copy(self):
        return type(self).from_export(self.export())


# 既存のimportとの互換名。環境との接続は別のアダプターで扱う。
GRAG = SGCRL


class GridWorldSGCRL:
    """既存TrainingSession用アダプター。環境固有処理をSGCRL本体から分離する。

    単一マップ・単一目標で収集する。衝突行動をマスクせず、報酬は捨てる。
    状態と達成ゴールは[障害物one-hot, 位置one-hot]の同じ表現を使う。
    """

    display_name = "SGCRL"
    CHECKPOINT_INCLUDES_MODEL = True
    ACTIONS = ("up", "down", "left", "right")

    def __init__(self, settings):
        self.core = SGCRL(settings)
        self._map_spec = None
        self._feature_vectors = {}
        self._obstacle_features = None
        self._policy_cache = {}
        self._policy_cache_updates = -1
        self._grid_action_generator = torch.Generator(device="cpu").manual_seed(self.core.seed ^ 0x6B31)

    @property
    def device(self):
        return self.core.device

    @property
    def metrics(self):
        return self.core.metrics

    @property
    def _updates(self):
        return self.core._updates

    def _features(self, observation, *, initialize=True):
        rows, cols = observation["rows"], observation["cols"]
        spec = (rows, cols, tuple(sorted(map(tuple, observation["obstacles"]))),
                tuple(observation["goal"]))
        if self._map_spec is None:
            if not initialize:
                raise NotImplementedError("ネットワークが未初期化です。")
            self.core.initialize_networks(2 * rows * cols, 2 * rows * cols,
                                          len(self.ACTIONS), discrete=True)
            self._map_spec = spec
        elif spec != self._map_spec:
            raise ValueError("SGCRLのデータ収集中はマップと目標を固定してください。")
        if self._obstacle_features is None:
            self._obstacle_features = torch.zeros(rows * cols)
            for row, col in spec[2]:
                self._obstacle_features[row * cols + col] = 1

        def encode(position):
            key = tuple(position)
            if key not in self._feature_vectors:
                location = torch.zeros(rows * cols)
                location[position[0] * cols + position[1]] = 1
                self._feature_vectors[key] = torch.cat((self._obstacle_features, location))
            return self._feature_vectors[key]

        return encode(observation["position"]), encode(observation["goal"])

    def choose_action(self, observation, greedy=False):
        state, goal = self._features(observation)
        if not greedy and self.core._steps < self.core.min_replay_size:
            action = torch.randint(len(self.ACTIONS), (), generator=self._grid_action_generator).item()
        else:
            probabilities = self._cached_probabilities(observation, state, goal)
            action = (probabilities.argmax().item() if greedy else
                      torch.multinomial(probabilities, 1, generator=self._grid_action_generator).item())
        return self.ACTIONS[action]

    def _cached_probabilities(self, observation, state, goal):
        if self._policy_cache_updates != self.core._updates:
            self._policy_cache.clear()
            self._policy_cache_updates = self.core._updates
        position = tuple(observation["position"])
        if position not in self._policy_cache:
            self._policy_cache[position] = self.core.action_probabilities(state, goal).cpu()
        return self._policy_cache[position]

    def action_probabilities(self, observation):
        state, goal = self._features(observation, initialize=False)
        probabilities = self._cached_probabilities(observation, state, goal).tolist()
        return dict(zip(self.ACTIONS, probabilities))

    def learn(self, observation, action, reward, next_observation, reached_goal):
        state, goal = self._features(observation)
        next_state, _ = self._features(next_observation)
        self.core.learn(state, torch.tensor(self.ACTIONS.index(action)), next_state, goal,
                        terminated=reached_goal)

    def end_episode(self):
        self.core.end_episode()

    def runtime_status(self):
        return self.core.runtime_status()

    def export(self):
        return {"format": "gridworld-sgcrl-v1", "map_spec": self._map_spec,
                "core": self.core.export(), "grid_action_random": self._grid_action_generator.get_state().tolist()}

    def checkpoint(self):
        return {"format": "gridworld-sgcrl-v1", "map_spec": self._map_spec,
                "core": self.core.checkpoint(), "grid_action_random": self._grid_action_generator.get_state().tolist()}

    def load_checkpoint(self, value):
        if value.get("format") != "gridworld-sgcrl-v1":
            raise ValueError("GridWorld SGCRLの保存形式が異なります。")
        self.core.load_checkpoint(value["core"])
        self._grid_action_generator = torch.Generator(device="cpu").manual_seed(self.core.seed ^ 0x6B31)
        if "grid_action_random" in value:
            self._grid_action_generator.set_state(torch.tensor(value["grid_action_random"], dtype=torch.uint8))
        self._policy_cache = {}
        self._policy_cache_updates = -1
        self._feature_vectors = {}
        self._obstacle_features = None
        spec = value["map_spec"]
        self._map_spec = (None if spec is None else
                          (spec[0], spec[1], tuple(map(tuple, spec[2])), tuple(spec[3])))

    @classmethod
    def from_export(cls, value):
        agent = cls({})
        agent.load_checkpoint(value)
        return agent

    def evaluation_copy(self):
        return type(self).from_export(self.export())
