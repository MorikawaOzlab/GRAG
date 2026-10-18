"""Replaceable AI example: tabular Q learning."""

import random


ACTIONS = ("up", "down", "left", "right")


class QLearningAgent:
    def __init__(self, settings: dict):
        self.settings = settings
        self.rng = random.Random(settings["seed"])
        self.q: dict[tuple[int, int], list[float]] = {}
        self.epsilon = settings["epsilon"]

    @staticmethod
    def key(observation: dict) -> tuple[int, int]:
        return tuple(observation["position"])

    def values(self, observation: dict) -> list[float]:
        key = self.key(observation)
        values = self.q.get(key)
        if values is None:
            values = [0.0] * len(ACTIONS)
            self.q[key] = values
        return values

    def set_episode_limit(self, max_steps: int) -> None:
        """実行側から次の試行の上限を受け取る。環境そのものは参照しない。"""
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_stepsは1以上の整数にしてください。")
        self.settings["max_steps"] = max_steps

    def action_probabilities(self, observation: dict) -> dict[str, float]:
        """探索を含む学習時の選択確率。Q表や乱数を変更しない。"""
        values = self.q.get(self.key(observation), [0.0] * len(ACTIONS))
        probabilities = [self.epsilon / len(ACTIONS)] * len(ACTIONS)
        # choose_action()と同じく、最大Q値が同点なら最初の方向を選ぶ。
        probabilities[values.index(max(values))] += 1.0 - self.epsilon
        return dict(zip(ACTIONS, probabilities))

    def choose_action(self, observation: dict, greedy: bool = False) -> str:
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.choice(ACTIONS)
        values = (self.q.get(self.key(observation), [0.0] * len(ACTIONS))
                  if greedy else self.values(observation))
        best = max(values)
        # Stable tie breaking keeps replay reproducible.
        return ACTIONS[values.index(best)]

    def learn(self, observation: dict, action: str, reward: float,
              next_observation: dict, reached_goal: bool) -> None:
        current = self.values(observation)
        index = ACTIONS.index(action)
        future = 0.0 if reached_goal else max(self.values(next_observation))
        target = reward + self.settings["discount_factor"] * future
        current[index] += self.settings["learning_rate"] * (target - current[index])

    def end_episode(self) -> None:
        self.epsilon = max(self.settings["min_epsilon"],
                           self.epsilon * self.settings["epsilon_decay"])

    def export(self) -> dict[str, list[float]]:
        return {f"{r},{c}": values for (r, c), values in sorted(self.q.items())}

    def checkpoint(self) -> dict:
        """続行に必要なQ表・探索率・乱数状態を保存する。"""
        return {"q_table": {key: values.copy() for key, values in self.export().items()},
                "epsilon": self.epsilon, "rng_state": self.rng.getstate()}

    def load_checkpoint(self, state: dict) -> None:
        self.q = {tuple(map(int, key.split(","))): values.copy()
                  for key, values in state["q_table"].items()}
        self.epsilon = state["epsilon"]

        def as_tuple(value):
            return tuple(as_tuple(item) for item in value) if isinstance(value, (list, tuple)) else value

        self.rng.setstate(as_tuple(state["rng_state"]))
