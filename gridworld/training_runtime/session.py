"""同じ学習器を1行動ずつ進める。観察と高速学習は同じstep()を使う。"""

from collections import deque
from copy import deepcopy
import math
from threading import RLock

from agents.q_learning import ACTIONS, QLearningAgent
from gridworld.world import GridWorld


def evaluate_policy(map_data: dict, settings: dict, agent) -> list[dict]:
    """互換用の方策評価。学習中の状態・乱数は変更しない。"""
    clone = getattr(agent, "evaluation_copy", None)
    restore = getattr(type(agent), "from_export", None)
    policy = (clone() if callable(clone) else restore(agent.export())
              if callable(restore) else deepcopy(agent))
    world = GridWorld(map_data)
    observation = world.reset()
    frames = [{"position": observation["position"], "action": None,
               "collision": False, "reached_goal": False}]
    for _ in range(settings["max_steps"]):
        action = policy.choose_action(observation, greedy=True)
        observation, reached_goal, collision = world.step(action)
        frames.append({"position": observation["position"], "action": action,
                       "collision": collision, "reached_goal": reached_goal})
        if reached_goal:
            break
    return frames


class TrainingSession:
    """保持するのは学習器、現在の試行、集計値。試行の軌跡は記録しない。"""

    MAX_CURVE_POINTS = 2000

    def __init__(self, map_data: dict, settings: dict, agent_type=QLearningAgent):
        self.world = GridWorld(deepcopy(map_data))
        self.settings = deepcopy(settings)
        self.agent = agent_type(deepcopy(self.settings))
        self.history = []
        self.visits = [[0] * self.world.map["cols"] for _ in range(self.world.map["rows"])]
        self.successes = 0
        self.total_reward = 0
        self.completed_episodes = 0
        self.total_steps = 0
        self.frames = []  # 明示的な方策評価を呼んだ場合だけ使用する。
        self.last_batch_episodes = 0
        self._recent = deque(maxlen=100)
        self._recent_steps = deque(maxlen=100)
        self._observation = None
        self._episode_steps = 0
        self._episode_reward = 0
        self._episode_collisions = 0
        self._episode_limit = settings["max_steps"]
        self._lock = RLock()
        self.live = {"episode": 1, "step": 0, "position": list(self.world.map["start"]),
                     "action": None, "collision": False, "reached_goal": False,
                     "completed_episodes": 0, "total_steps": 0}

    def set_max_steps(self, value: int) -> None:
        if type(value) is not int or not 1 <= value <= 1_000_000:
            raise ValueError("最大行動回数は1〜1,000,000の整数にしてください。")
        with self._lock:
            self.settings["max_steps"] = value

    def step(self) -> dict:
        """実際の行動選択・環境遷移・学習を1回行う。途中で停止・再開できる。"""
        with self._lock:
            if self._observation is None:
                self._observation = self.world.reset()
                self._episode_steps = self._episode_reward = self._episode_collisions = 0
                self._episode_limit = self.settings["max_steps"]
                set_limit = getattr(self.agent, "set_episode_limit", None)
                if callable(set_limit):
                    set_limit(self._episode_limit)
            episode = self.completed_episodes + 1
            observation = self._observation
            action = self.agent.choose_action(observation)
            next_observation, reached_goal, collision = self.world.step(action)
            reward = 1 if reached_goal else 0
            self.agent.learn(observation, action, reward, next_observation, reached_goal)
            self._observation = next_observation
            self._episode_steps += 1
            self.total_steps += 1
            self._episode_reward += reward
            self._episode_collisions += int(collision)
            if not collision:
                r, c = next_observation["position"]
                self.visits[r][c] += 1
            finished = reached_goal or self._episode_steps >= self._episode_limit
            if finished:
                self.agent.end_episode()
                self.completed_episodes += 1
                self.successes += int(reached_goal)
                self.total_reward += self._episode_reward
                self._recent.append(self._episode_reward)
                self._recent_steps.append(self._episode_steps)
                self.history.append({
                    "episode": self.completed_episodes, "steps": self._episode_steps,
                    "reward": self._episode_reward, "cumulative_reward": self.total_reward,
                    "success": reached_goal, "collisions": self._episode_collisions,
                    "moving_average_reward": sum(self._recent) / len(self._recent)})
                if len(self.history) > self.MAX_CURVE_POINTS:
                    latest = self.history[-1]
                    self.history = self.history[::2]
                    if self.history[-1] is not latest:
                        self.history.append(latest)
                self._observation = None
            self.live = {
                "episode": episode, "step": self._episode_steps,
                "position": list(next_observation["position"]), "action": action,
                "collision": collision, "reached_goal": reached_goal,
                "completed_episodes": self.completed_episodes, "total_steps": self.total_steps,
                "finished": finished, "max_steps": self._episode_limit}
            return self.live.copy()

    def run_batch(self, episodes: int, progress=None) -> None:
        if type(episodes) is not int or not 1 <= episodes <= 1_000_000:
            raise ValueError("今回の学習回数は1〜1,000,000の整数にしてください。")
        self.last_batch_episodes = episodes
        self.frames = []
        before = self.completed_episodes
        target = before + episodes
        interval = max(1, episodes // 100)
        while self.completed_episodes < target:
            state = self.step()
            completed = self.completed_episodes - before
            if state["finished"] and progress and (completed % interval == 0 or completed == episodes):
                progress(completed)

    def inspect_policy(self, position: list[int]) -> dict:
        """指定マスにいると仮定した現在の方策。1行動の学習との間で取得する。"""
        if (not isinstance(position, list) or len(position) != 2
                or any(type(value) is not int for value in position)
                or not 0 <= position[0] < self.world.map["rows"]
                or not 0 <= position[1] < self.world.map["cols"]):
            raise ValueError("確認するマスは範囲内の[行, 列]にしてください。")
        with self._lock:
            r, c = position
            wall = position in self.world.map["obstacles"]
            inspect = getattr(self.agent, "action_probabilities", None)
            supported = callable(inspect) and not wall
            observation = self.world.observation()
            observation["position"] = position.copy()
            probabilities = None
            if supported:
                try:
                    probabilities = inspect(observation)
                except NotImplementedError:
                    supported = False
            if supported and (not isinstance(probabilities, dict)
                    or set(probabilities) != set(ACTIONS)
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           or not 0 <= value <= 1 for value in probabilities.values())
                    or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5)):
                raise ValueError("AIの行動確率は4方向の確率（合計1）で返してください。")
            return {"position": position.copy(), "probabilities": probabilities,
                    "supported": supported, "wall": wall,
                    "goal": position == self.world.map["goal"],
                    "visits": self.visits[r][c],
                    "completed_episodes": self.completed_episodes, "total_steps": self.total_steps,
                    "blocked_actions": [] if wall else [action for action in ACTIONS
                        if self.world.transition((r, c), action)[2]],
                    "note": "障害物のマスには移動できません。" if wall else
                            "このAIは行動確率の表示に対応していません。" if not supported else
                            "ゴール到達時には試行が終了します。表示はこの位置を入力した方策の出力です。"
                            if position == self.world.map["goal"] else
                            "衝突する方向も選択対象に含まれます。"}

    def evaluate(self) -> list[dict]:
        with self._lock:
            self.frames = evaluate_policy(self.world.map, self.settings, self.agent)
            return deepcopy(self.frames)

    def result(self, *, checkpoint: bool = False, compact: bool = False) -> dict:
        with self._lock:
            history = self.history
            if compact and history:
                stride = max(1, (len(history) + 699) // 700)
                history = history[::stride]
                if history[-1] is not self.history[-1]:
                    history.append(self.history[-1])
            result = {
                "schema_version": 3, "map": deepcopy(self.world.map),
                "settings": {**self.settings, "episodes": self.completed_episodes},
                "completed_episodes": self.completed_episodes, "total_steps": self.total_steps,
                "visits": deepcopy(self.visits), "history": deepcopy(history),
                "successes": self.successes, "total_reward": self.total_reward,
                "recent_success_rate": sum(self._recent) / len(self._recent) if self._recent else 0,
                "recent_mean_steps": sum(self._recent_steps) / len(self._recent_steps)
                                     if self._recent_steps else 0,
                "frames": deepcopy(self.frames), "live": deepcopy(self.live),
                "workers_used": 1, "batch_episodes": self.last_batch_episodes,
                "q_table": None if compact else self.agent.export()}
            if checkpoint:
                save = getattr(self.agent, "checkpoint", None)
                result["agent_checkpoint"] = save() if callable(save) else None
                result["recent_rewards"] = list(self._recent)
                result["recent_steps"] = list(self._recent_steps)
                result["episode_state"] = {
                    "active": self._observation is not None,
                    "position": list(self.world.position), "done": self.world.done,
                    "steps": self._episode_steps, "reward": self._episode_reward,
                    "collisions": self._episode_collisions, "limit": self._episode_limit}
            return result

    @classmethod
    def restore(cls, result: dict, agent_type=QLearningAgent):
        session = cls(result["map"], result["settings"], agent_type)
        load = getattr(session.agent, "load_checkpoint", None)
        if result.get("agent_checkpoint") is None or not callable(load):
            raise ValueError("この学習器はアプリ再起動後の続行に対応していません。")
        load(result["agent_checkpoint"])
        session.history = deepcopy(result["history"])
        session.completed_episodes = result.get("completed_episodes", result["settings"]["episodes"])
        session.total_steps = result.get("total_steps", sum(item["steps"] for item in session.history))
        session.visits = deepcopy(result["visits"])
        session.successes = result["successes"]
        session.total_reward = result["total_reward"]
        session.frames = deepcopy(result.get("frames", []))
        session.live = deepcopy(result.get("live", session.live))
        session.last_batch_episodes = result.get("batch_episodes", 0)
        session._recent.extend(result.get("recent_rewards", [item["reward"] for item in session.history[-100:]]))
        session._recent_steps.extend(result.get("recent_steps", [item["steps"] for item in session.history[-100:]]))
        state = result.get("episode_state", {})
        session._episode_steps = state.get("steps", 0)
        session._episode_reward = state.get("reward", 0)
        session._episode_collisions = state.get("collisions", 0)
        session._episode_limit = state.get("limit", session.settings["max_steps"])
        session.world.position = tuple(state.get("position", session.world.map["start"]))
        session.world.done = state.get("done", False)
        if state.get("active"):
            session._observation = session.world.observation()
        return session
