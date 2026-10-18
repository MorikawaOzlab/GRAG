"""CleanRL型PPOを、観測辞書・4方向の行動を使うGRAGとして提供する。

環境の生成・遷移・試行ループ・HTTP・ログ保存は実行側が担当する。
"""

from dataclasses import dataclass
import math
import random

from agents.ppo_checkpoint import PPOCheckpointMixin
from .goals import LanguageGoal


ACTIONS = ("up", "down", "left", "right")


@dataclass
class _Transition:
    state: tuple[float, ...]
    action: int
    log_probability: float
    value: float
    reward: float
    next_value: float
    terminated: bool
    episode_end: bool = False


#=================
# 変更対象のメソッド
#=================
def _compute_gae(transitions: list[_Transition], gamma: float,
                 gae_lambda: float) -> tuple[list[float], list[float]]:
    """ゴールではbootstrapを止め、時間制限では最終状態の価値を使う。"""
    advantages = [0.0] * len(transitions)
    following_advantage = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        item = transitions[index]
        next_value = 0.0 if item.terminated else item.next_value
        delta = item.reward + gamma * next_value - item.value
        continuation = not (item.terminated or item.episode_end)
        following_advantage = (delta + gamma * gae_lambda
                               * continuation * following_advantage)
        advantages[index] = following_advantage
    returns = [advantage + item.value
               for advantage, item in zip(advantages, transitions)]
    return advantages, returns


def _number(settings: dict, name: str, default: float, minimum: float = 0.0,
            maximum: float | None = None, positive: bool = False) -> float:
    value = settings.get(name, default)
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < minimum or (positive and value <= 0)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{name}の値が不正です。")
    return float(value)


def _integer(settings: dict, name: str, default: int) -> int:
    value = settings.get(name, default)
    if type(value) is not int or value < 1:
        raise ValueError(f"{name}は1以上の整数にしてください。")
    return value


def _normalize_advantages(advantages):
    """元のCleanRLと同じミニバッチ内の正規化。1サンプルではNaNを避ける。"""
    if advantages.numel() < 2:
        return advantages
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


def _value_loss(values, returns, old_values, clip_range: float, clip_value: bool):
    """CleanRLの1/2付き価値損失。旧価値からの変化をクリップする。"""
    errors = (values - returns).square()
    if clip_value:
        clipped_values = old_values + (values - old_values).clamp(-clip_range, clip_range)
        errors = errors.maximum((clipped_values - returns).square())
    return 0.5 * errors.mean()


def _actor_critic(torch, input_size: int, hidden_size: int):
    from .networks import Agent
    return Agent(input_size, len(ACTIONS), hidden_size)


class GRAG(PPOCheckpointMixin):
    """離散4行動のニューラルPPO。既存のtrain(..., agent_type=GRAG)で使う。"""

    _transition_type = _Transition

    def __init__(self, settings: dict):
        self.settings = settings.copy()
        self._goal_text = settings.get("goal_text")
        if self._goal_text is not None and (
                not isinstance(self._goal_text, str) or not self._goal_text.strip()):
            raise ValueError("goal_textは空でない文字列またはNoneにしてください。")
        seed = settings.get("seed")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("seedは0〜2^32-1の整数またはNoneにしてください。")
        self._seed = seed if seed is not None else random.SystemRandom().randrange(2**32)
        gamma_default = settings.get("discount_factor", 0.99)
        self._config = {
            "ppo_learning_rate": _number(settings, "ppo_learning_rate", 2.5e-4,
                                         positive=True),
            "ppo_gamma": _number(settings, "ppo_gamma", gamma_default, maximum=1.0),
            "ppo_gae_lambda": _number(settings, "ppo_gae_lambda", 0.95, maximum=1.0),
            "ppo_clip_range": _number(settings, "ppo_clip_range", 0.2,
                                      maximum=0.99, positive=True),
            "ppo_epochs": _integer(settings, "ppo_epochs", 4),
            "ppo_minibatch_size": _integer(settings, "ppo_minibatch_size", 64),
            "ppo_rollout_steps": _integer(settings, "ppo_rollout_steps", 128),
            "ppo_hidden_size": _integer(settings, "ppo_hidden_size", 64),
            "ppo_value_coef": _number(settings, "ppo_value_coef", 0.5),
            "ppo_entropy_coef": _number(settings, "ppo_entropy_coef", 0.01),
            "ppo_max_grad_norm": _number(settings, "ppo_max_grad_norm", 0.5,
                                         positive=True),
        }
        normalize = settings.get("ppo_normalize_advantage", True)
        if type(normalize) is not bool:
            raise ValueError("ppo_normalize_advantageはboolにしてください。")
        self._config["ppo_normalize_advantage"] = normalize
        target_kl = settings.get("ppo_target_kl", None)
        self._config["ppo_target_kl"] = (
            None if target_kl is None else _number(
                settings, "ppo_target_kl", 0.03, positive=True))
        for name, default in (("ppo_clip_value_loss", True), ("ppo_anneal_lr", False)):
            value = settings.get(name, default)
            if type(value) is not bool:
                raise ValueError(f"{name}はboolにしてください。")
            self._config[name] = value
        budget = settings.get("ppo_total_timesteps")
        if budget is not None and (type(budget) is not int or budget < 1):
            raise ValueError("ppo_total_timestepsは1以上の整数またはNoneにしてください。")
        if self._config["ppo_anneal_lr"] and budget is None:
            raise ValueError("学習率減衰にはppo_total_timestepsを指定してください。")
        self._config["ppo_total_timesteps"] = budget
        self._torch = None
        self._model = None
        self._optimizer = None
        self._generator = None
        self._map_shape = None
        self._pending = None
        self._rollout: list[_Transition] = []
        self._updates = 0
        self._episodes = 0
        self._steps = 0
        self._last_update: dict = {}

    def goal_for(self, observation: dict) -> LanguageGoal:
        """設定の目標文と、観測のgoal座標を組にして返す。"""
        return LanguageGoal.from_observation(observation, self._goal_text)

    def is_goal_reached(self, observation: dict) -> bool:
        return self.goal_for(observation).is_reached(observation)

    @staticmethod
    def _features(observation: dict) -> tuple[tuple[int, int], tuple[float, ...]]:
        """障害物・ゴール・現在位置の3チャネルと、正規化座標を入力する。"""
        rows, cols = observation.get("rows"), observation.get("cols")
        if (type(rows) is not int or type(cols) is not int
                or not 2 <= rows <= 50 or not 2 <= cols <= 50):
            raise ValueError("rowsとcolsは2〜50の整数にしてください。")

        def coordinate(value, label):
            if (not isinstance(value, (list, tuple)) or len(value) != 2
                    or any(type(item) is not int for item in value)
                    or not 0 <= value[0] < rows or not 0 <= value[1] < cols):
                raise ValueError(f"{label}は範囲内の[行, 列]にしてください。")
            return tuple(value)

        position = coordinate(observation.get("position"), "現在位置")
        goal = coordinate(observation.get("goal"), "目標座標")
        obstacles = observation.get("obstacles", [])
        if not isinstance(obstacles, (list, tuple)):
            raise ValueError("obstaclesは座標配列にしてください。")
        walls = {coordinate(cell, "障害物") for cell in obstacles}
        if position in walls or goal in walls:
            raise ValueError("現在位置とゴールには障害物のないマスを指定してください。")
        cells = [(r, c) for r in range(rows) for c in range(cols)]
        features = ([float(cell in walls) for cell in cells]
                    + [float(cell == goal) for cell in cells]
                    + [float(cell == position) for cell in cells]
                    + [position[0] / (rows - 1), position[1] / (cols - 1),
                       goal[0] / (rows - 1), goal[1] / (cols - 1)])
        return (rows, cols), tuple(features)

    def _initialize_model(self, shape: tuple[int, int]) -> None:
        if self._map_shape is not None:
            if shape != self._map_shape:
                raise ValueError("同じGRAGには同じ行数・列数のマップを渡してください。")
            return
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "PPOにはPyTorchが必要です。使用するPythonで "
                "python -m pip install -r agents/grag_impl/requirements.txt "
                "を実行してください。") from error
        # CPUだけを使い、他の利用者のグローバル乱数系列を変更しない。
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self._seed)
            model = _actor_critic(torch, 3 * shape[0] * shape[1] + 4,
                                  self._config["ppo_hidden_size"]).to(device="cpu", dtype=torch.float32)
        self._torch = torch
        self._model = model
        self._optimizer = torch.optim.Adam(
            model.parameters(), lr=self._config["ppo_learning_rate"], eps=1e-5)
        self._generator = torch.Generator(device="cpu").manual_seed(self._seed)
        self._map_shape = shape

    def set_episode_limit(self, max_steps: int) -> None:
        """実行側から次の試行の上限を受け取る。環境そのものは参照しない。"""
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_stepsは1以上の整数にしてください。")
        self.settings["max_steps"] = max_steps

    def action_probabilities(self, observation: dict) -> dict[str, float]:
        """現在の方策の確率を返す。学習状態・行動乱数を変更しない。"""
        shape, features = self._features(observation)
        # 学習0回では同じ初期シードの一時モデルを使い、本体を初期化しない。
        policy = self if self._model is not None else type(self).from_export(self.export())
        policy._initialize_model(shape)
        torch = policy._torch
        with torch.no_grad():
            logits = policy._model.actor(torch.tensor(features, dtype=torch.float32).unsqueeze(0))
            probabilities = torch.distributions.Categorical(logits=logits).probs[0].tolist()
        return dict(zip(ACTIONS, probabilities))

    def choose_action(self, observation: dict, greedy: bool = False) -> str:
        """学習時は方策からサンプル、評価時は最大確率の行動を返す。"""
        if not greedy and self._pending is not None:
            raise RuntimeError("次の行動選択の前にlearn()を呼んでください。")
        shape, features = self._features(observation)
        self._initialize_model(shape)
        torch = self._torch
        with torch.no_grad():
            state = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            if greedy:
                logits = self._model.actor(state)
                return ACTIONS[int(logits.argmax(dim=-1).item())]
            action, log_probability, _, value = self._model.get_action_and_value(
                state, generator=self._generator)
            index = int(action.item())
            self._pending = (features, index, float(log_probability.item()), float(value.item()))
        return ACTIONS[index]

    def learn(self, observation: dict, action: str, reward: float,
              next_observation: dict, reached_goal: bool) -> None:
        """行動時の旧方策・価値を固定して、on-policyの経験を収集する。"""
        if self._pending is None:
            raise RuntimeError("learn()の前にchoose_action(..., greedy=False)を呼んでください。")
        shape, features = self._features(observation)
        next_shape, next_features = self._features(next_observation)
        if shape != self._map_shape or next_shape != self._map_shape:
            raise ValueError("行動前後でマップの行数・列数を変更できません。")
        old_features, index, log_probability, value = self._pending
        if features != old_features or action != ACTIONS[index]:
            raise ValueError("learn()には行動選択時と同じ観測・行動を渡してください。")
        if type(reward) not in (int, float) or not math.isfinite(reward):
            raise ValueError("rewardは有限の数値にしてください。")
        if type(reached_goal) is not bool:
            raise ValueError("reached_goalはboolにしてください。")
        next_value = 0.0
        if not reached_goal:
            torch = self._torch
            with torch.no_grad():
                _, prediction = self._model(
                    torch.tensor(next_features, dtype=torch.float32).unsqueeze(0))
                next_value = float(prediction.item())
        self._rollout.append(_Transition(
            features, index, log_probability, value, float(reward),
            next_value, reached_goal, episode_end=reached_goal))
        self._pending = None
        self._steps += 1
        if len(self._rollout) >= self._config["ppo_rollout_steps"]:
            self._update()

    def end_episode(self) -> None:
        """試行末尾の経験も更新する。時間制限では最終状態からbootstrapする。"""
        if self._pending is not None:
            raise RuntimeError("end_episode()の前に最後の行動のlearn()を呼んでください。")
        if self._rollout:
            self._rollout[-1].episode_end = True
            self._update()
        self._episodes += 1

    def _update(self) -> None:
        if not self._rollout:
            return
        torch, config = self._torch, self._config
        advantages, returns = _compute_gae(
            self._rollout, config["ppo_gamma"], config["ppo_gae_lambda"])
        states = torch.tensor([item.state for item in self._rollout], dtype=torch.float32)
        actions = torch.tensor([item.action for item in self._rollout], dtype=torch.long)
        old_log_probabilities = torch.tensor(
            [item.log_probability for item in self._rollout], dtype=torch.float32)
        old_values = torch.tensor([item.value for item in self._rollout], dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        count = len(self._rollout)
        if config["ppo_anneal_lr"]:
            # 収集したバッファーの先頭を進捗に使い、最初の更新は初期学習率で行う。
            fraction = max(0.0, 1.0 - (self._steps - count) / config["ppo_total_timesteps"])
            for group in self._optimizer.param_groups:
                group["lr"] = fraction * config["ppo_learning_rate"]
        totals = {name: 0.0 for name in (
            "policy_loss", "value_loss", "entropy", "old_approx_kl", "approx_kl",
            "clip_fraction", "gradient_norm")}
        optimized_samples = optimization_steps = epochs_used = 0
        stop = False
        for _ in range(config["ppo_epochs"]):
            indices = torch.randperm(count, generator=self._generator)
            for offset in range(0, count, config["ppo_minibatch_size"]):
                batch = indices[offset:offset + config["ppo_minibatch_size"]]
                _, log_probabilities, entropies, values = self._model.get_action_and_value(
                    states[batch], actions[batch])
                values = values.view(-1)
                log_ratio = log_probabilities - old_log_probabilities[batch]
                ratio = log_ratio.exp()
                batch_advantages = advantages[batch]
                if config["ppo_normalize_advantage"]:
                    batch_advantages = _normalize_advantages(batch_advantages)
                loss_unclipped = -batch_advantages * ratio
                loss_clipped = -batch_advantages * ratio.clamp(
                    1 - config["ppo_clip_range"], 1 + config["ppo_clip_range"])
                policy_loss = torch.maximum(loss_unclipped, loss_clipped).mean()
                value_loss = _value_loss(values, returns[batch], old_values[batch],
                                         config["ppo_clip_range"], config["ppo_clip_value_loss"])
                entropy = entropies.mean()
                loss = (policy_loss - config["ppo_entropy_coef"] * entropy
                        + config["ppo_value_coef"] * value_loss)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("PPOの損失が有限値ではありません。")
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    old_approx_kl = (-log_ratio).mean().item()
                    clip_fraction = ((ratio - 1).abs() > config["ppo_clip_range"]).float().mean().item()
                self._optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), config["ppo_max_grad_norm"], error_if_nonfinite=True)
                self._optimizer.step()
                batch_count = len(batch)
                metrics = {
                    "policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                    "entropy": entropy.item(), "old_approx_kl": old_approx_kl,
                    "approx_kl": approx_kl, "clip_fraction": clip_fraction,
                    "gradient_norm": gradient_norm.item()}
                for name, metric in metrics.items():
                    totals[name] += metric * batch_count
                optimized_samples += batch_count
                optimization_steps += 1
            epochs_used += 1
            # 元のCleanRLと同じく、エポック末尾のKLで次の反復を止める。
            if config["ppo_target_kl"] is not None and approx_kl > config["ppo_target_kl"]:
                stop = True
                break
        with torch.no_grad():
            variance = returns.var(unbiased=False).item()
            explained_variance = (None if variance == 0 else
                                  1.0 - (returns - old_values).var(unbiased=False).item() / variance)
        self._last_update = {
            **{name: value / max(1, optimized_samples) for name, value in totals.items()},
            "samples": count, "optimization_steps": optimization_steps, "epochs": epochs_used,
            "early_stopped": stop, "explained_variance": explained_variance,
            "learning_rate": self._optimizer.param_groups[0]["lr"]}
        self._updates += 1
        self._rollout.clear()

    def export(self) -> dict:
        """推論用モデルをJSONで保存する。optimizer・途中の軌道は含めない。"""
        weights = None if self._model is None else {
            name: tensor.detach().cpu().tolist()
            for name, tensor in self._model.state_dict().items()}
        return {
            "schema_version": 1, "algorithm": "PPO", "implementation": "cleanrl_grid_v1",
            "model_type": "mlp_actor_critic",
            "observation_encoding": "grid_channels_v1", "actions": list(ACTIONS),
            "map_shape": None if self._map_shape is None else list(self._map_shape),
            "settings": {**self._config, "seed": self._seed, "goal_text": self._goal_text},
            "state_dict": weights,
            "training": {"steps": self._steps, "episodes": self._episodes,
                         "updates": self._updates},
            "last_update": self._last_update.copy(),
        }

    @classmethod
    def from_export(cls, data: dict) -> "GRAG":
        """export()のモデルを復元する。再学習は新しいoptimizer・乱数系列で始める。"""
        if (data.get("schema_version") != 1 or data.get("algorithm") != "PPO"
                or data.get("model_type") != "mlp_actor_critic"
                or data.get("observation_encoding") != "grid_channels_v1"
                or data.get("actions") != list(ACTIONS)):
            raise ValueError("対応していないPPOモデル形式です。")
        agent = cls(data["settings"])
        shape = data.get("map_shape")
        weights = data.get("state_dict")
        if shape is not None:
            if (not isinstance(shape, (list, tuple)) or len(shape) != 2
                    or any(type(size) is not int or not 2 <= size <= 50 for size in shape)):
                raise ValueError("保存モデルのmap_shapeが不正です。")
            if not isinstance(weights, dict):
                raise ValueError("保存モデルのstate_dictが不正です。")
            agent._initialize_model(tuple(shape))
            tensors = {name: agent._torch.tensor(value, dtype=agent._torch.float32)
                       for name, value in weights.items()}
            if any(not agent._torch.isfinite(tensor).all().item()
                   for tensor in tensors.values()):
                raise ValueError("保存モデルの重みは有限値にしてください。")
            agent._model.load_state_dict(tensors, strict=True)
        elif weights is not None:
            raise ValueError("重みを復元するにはmap_shapeが必要です。")
        counters = data.get("training", {})
        for key, attribute in (("steps", "_steps"), ("episodes", "_episodes"),
                               ("updates", "_updates")):
            value = counters.get(key, 0)
            if type(value) is not int or value < 0:
                raise ValueError("保存モデルの学習回数が不正です。")
            setattr(agent, attribute, value)
        agent._last_update = dict(data.get("last_update", {}))
        return agent
