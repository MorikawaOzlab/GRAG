"""Small reproducible learning check, separate from the paper's benchmarks.

Run from the project root: python -m scripts.verify_sgcrl --episodes 1000
"""
import argparse
import json
from time import monotonic

import torch

from agents.grag import GRAG
from gridworld.storage import DEFAULT_SETTINGS
from gridworld.training_runtime import TrainingSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    parser.add_argument("--device", default="auto", help="auto (GPU if available), cpu, or cuda:N")
    args = parser.parse_args()
    torch.set_num_threads(1)
    maze = {"name": "SGCRL small validation", "rows": 2, "cols": 3,
            "start": [0, 0], "goal": [1, 2], "obstacles": [[0, 1]]}
    # Deliberately smaller than paper settings for a quick learning check.
    settings = {**DEFAULT_SETTINGS, "max_steps": 16, "crl_hidden_size": 64,
                "crl_repr_dim": 16, "crl_batch_size": 32, "crl_min_replay_size": 64,
                "crl_device": args.device}
    for seed in args.seeds:
        session = TrainingSession(maze, {**settings, "seed": seed}, GRAG)
        started = monotonic()
        session.run_batch(args.episodes)
        frames = session.evaluate()
        print(json.dumps({"seed": seed, "episodes": args.episodes,
                          "device": str(session.agent.device),
                          "steps": session.total_steps, "updates": session.agent._updates,
                          "last_100_success_rate": session.result(compact=True)["recent_success_rate"],
                          "greedy_success": frames[-1]["reached_goal"],
                          "greedy_steps": len(frames) - 1,
                          "seconds": round(monotonic() - started, 2),
                          "metrics": session.agent.metrics}), flush=True)


if __name__ == "__main__":
    main()
