"""Batch training independent of the Web interface."""

from collections import deque

from agents.q_learning import QLearningAgent
from gridworld.training_runtime.session import TrainingSession, evaluate_policy


def add_moving_average(history: list[dict], window: int = 100) -> None:
    recent = deque()
    rolling_sum = 0
    cumulative = 0
    for episode, item in enumerate(history, 1):
        reward = item["reward"]
        recent.append(reward)
        rolling_sum += reward
        if len(recent) > window:
            rolling_sum -= recent.popleft()
        cumulative += reward
        item["episode"] = episode
        item["cumulative_reward"] = cumulative
        item["moving_average_reward"] = rolling_sum / len(recent)


def replay(map_data: dict, settings: dict, agent) -> list[dict]:
    """Read-only evaluation of the current policy from the start."""
    return evaluate_policy(map_data, settings, agent)


def train(map_data: dict, settings: dict, agent_type=QLearningAgent,
          progress=None, initial_q: dict | None = None,
          include_frames: bool = True) -> dict:
    session = TrainingSession(map_data, settings, agent_type)
    if initial_q is not None:
        if agent_type is not QLearningAgent:
            raise ValueError("初期Q表はQ学習AIにのみ指定できます。")
        session.agent.q = {tuple(map(int, key.split(","))): values.copy()
                           for key, values in initial_q.items()}
    session.run_batch(settings["episodes"], progress=progress)
    if include_frames:
        session.evaluate()
    return session.result()
