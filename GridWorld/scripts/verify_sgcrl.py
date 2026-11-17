"""Small reproducible learning check, separate from the paper's benchmarks.

Run from the repository root: python GridWorld/scripts/verify_sgcrl.py --episodes 1000
"""
import argparse
import json
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "GridWorld"))

import torch

from agents.grag_impl.grag_sgcrl import GridWorldSGCRL
from gridworld.storage import DEFAULT_SETTINGS
from gridworld.training_runtime import TrainingSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123, 2024, 2025])
    parser.add_argument("--device", default="auto", help="auto (GPU if available), cpu, or cuda:N")
    parser.add_argument("--map", type=Path, help="GridWorld map JSON; default is a small validation maze")
    parser.add_argument("--output", type=Path, help="save settings, per-seed curves and evaluation results as JSON")
    parser.add_argument("--warmup", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--updates-per-step", type=float, default=1.0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    maze = {"name": "SGCRL small validation", "rows": 2, "cols": 3,
            "start": [0, 0], "goal": [1, 2], "obstacles": [[0, 1]]}
    if args.map:
        maze = json.loads(args.map.read_text(encoding="utf-8"))
    settings = {**DEFAULT_SETTINGS, "episodes": args.episodes, "max_steps": args.max_steps,
                "crl_hidden_size": args.hidden_size, "crl_batch_size": args.batch_size,
                "crl_min_replay_size": args.warmup, "crl_updates_per_step": args.updates_per_step,
                "crl_device": args.device}
    results = []
    for seed in args.seeds:
        session = TrainingSession(maze, {**settings, "seed": seed}, GridWorldSGCRL)
        baseline = session.evaluate()
        started = monotonic()
        session.run_batch(args.episodes)
        frames = session.evaluate()
        summary = {"seed": seed, "episodes": args.episodes,
                          "device": str(session.agent.device),
                          "steps": session.total_steps, "updates": session.agent._updates,
                          "last_100_success_rate": session.result(compact=True)["recent_success_rate"],
                          "training_seconds": session.training_seconds,
                          "success_100_milestone": session.success_100_milestone,
                          "greedy_success": frames[-1]["reached_goal"],
                          "initial_greedy_success": baseline[-1]["reached_goal"],
                          "greedy_steps": len(frames) - 1,
                          "seconds": round(monotonic() - started, 2),
                          "metrics": session.agent.metrics}
        print(json.dumps(summary), flush=True)
        results.append({**summary, "history": session.history, "frames": frames})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"map": maze, "settings": settings,
                                          "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
