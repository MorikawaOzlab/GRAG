"""CPU-parallel Q learning using separate processes and periodic Q-table merging."""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from agents.q_learning import QLearningAgent
from gridworld.storage import validate_map
from gridworld.training import add_moving_average, replay, train


def _worker(map_data: dict, settings: dict, starting_q: dict) -> dict:
    return train(map_data, settings, initial_q=starting_q, include_frames=False)


def _merge_q_tables(starting_q: dict, results: list[dict], weights: list[int]) -> dict:
    keys = set(starting_q)
    for result in results:
        keys.update(result["q_table"])
    total = sum(weights)
    merged = {}
    for key in keys:
        base = starting_q.get(key, [0.0] * 4)
        vectors = [result["q_table"].get(key, base) for result in results]
        merged[key] = [sum(vector[action] * weight for vector, weight in zip(vectors, weights)) / total
                       for action in range(4)]
    return merged


def worker_count(episodes: int) -> int:
    """Leave capacity for two logical CPUs whenever the host has at least three."""
    return min(max(1, (os.cpu_count() or 1) - 2), episodes)


def train_parallel(map_data: dict, settings: dict, progress=None) -> dict:
    """Train independent workers in rounds; synchronize their Q tables between rounds.

    A worker count of one uses the original exact sequential algorithm. Multi-worker
    training is a different Q-learning schedule, so its Q table may differ.
    """
    clean_map = validate_map(map_data)
    episodes = settings["episodes"]
    workers = worker_count(episodes)
    if workers == 1:
        return train(clean_map, settings, progress=progress)

    rounds = max(1, min(4, episodes // (workers * 250)))
    round_base, extra_rounds = divmod(episodes, rounds)
    visits = [[0 for _ in range(clean_map["cols"])] for _ in range(clean_map["rows"])]
    history = []
    q_table = {}
    completed = 0
    assigned = 0
    seed = settings["seed"]

    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        for round_index in range(rounds):
            round_episodes = round_base + (round_index < extra_rounds)
            base, extra = divmod(round_episodes, workers)
            counts = [base + (worker_index < extra) for worker_index in range(workers)]
            future_to_index = {}
            results = [None] * workers
            for worker_index, count in enumerate(counts):
                local = settings.copy()
                local["episodes"] = count
                local["epsilon"] = max(settings["min_epsilon"],
                                       settings["epsilon"] * settings["epsilon_decay"] ** assigned)
                if seed is not None:
                    local["seed"] = (seed + 9973 * (round_index * workers + worker_index + 1)) % 2**32
                future = pool.submit(_worker, clean_map, local, q_table)
                future_to_index[future] = worker_index
                assigned += count
            for future in as_completed(future_to_index):
                worker_index = future_to_index[future]
                results[worker_index] = future.result()
                completed += counts[worker_index]
                if progress:
                    progress(completed)

            q_table = _merge_q_tables(q_table, results, counts)
            for result in results:
                history.extend(result["history"])
                for row in range(clean_map["rows"]):
                    for col in range(clean_map["cols"]):
                        visits[row][col] += result["visits"][row][col]

    add_moving_average(history)
    successes = sum(item["reward"] for item in history)
    agent = QLearningAgent(settings)
    agent.q = {tuple(map(int, key.split(","))): values.copy()
               for key, values in q_table.items()}
    return {"schema_version": 3, "map": clean_map, "settings": settings,
            "visits": visits, "history": history, "successes": successes,
            "total_reward": successes, "frames": replay(clean_map, settings, agent),
            "workers_used": workers, "q_table": q_table}
