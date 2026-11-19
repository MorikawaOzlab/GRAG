"""Reproducible GridWorld CPPO learning experiment (not the paper benchmark)."""

import argparse
import json
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "GridWorld"))

import torch

from agents.cppo import CPPOAgent, DEFAULTS
from gridworld.storage import DEFAULT_SETTINGS
from gridworld.training_runtime import TrainingSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    parser.add_argument("--map", type=Path)
    parser.add_argument("--hidden-size", type=int, default=DEFAULTS["hidden_size"])
    parser.add_argument("--depth", type=int, default=DEFAULTS["depth"])
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=DEFAULTS["rollout_steps"])
    parser.add_argument("--entropy-coef", type=float, default=DEFAULTS["entropy_coef"])
    parser.add_argument("--normalize-advantage", action=argparse.BooleanOptionalAction,
                        default=DEFAULTS["normalize_advantage"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    maze = {"name": "CPPO small validation", "rows": 2, "cols": 3,
            "start": [0, 0], "goal": [1, 2], "obstacles": [[0, 1]]}
    if args.map:
        maze = json.loads(args.map.read_text(encoding="utf-8"))
    settings = {**DEFAULT_SETTINGS, "max_steps": args.max_steps, "episodes": args.episodes,
                "cppo_hidden_size": args.hidden_size, "cppo_depth": args.depth,
                "cppo_rollout_steps": args.rollout_steps,
                "cppo_entropy_coef": args.entropy_coef,
                "cppo_normalize_advantage": args.normalize_advantage}
    results = []
    for seed in args.seeds:
        session = TrainingSession(maze, {**settings, "seed": seed}, CPPOAgent)
        before = session.evaluate(force_greedy=True)
        started = monotonic()
        session.run_batch(args.episodes)
        frames = session.evaluate(force_greedy=True)
        summary = {"seed": seed, "episodes": args.episodes, "steps": session.total_steps,
                   "updates": session.agent.updates,
                   "initial_greedy_success": before[-1]["reached_goal"],
                   "greedy_success": frames[-1]["reached_goal"], "greedy_steps": len(frames) - 1,
                   "last_100_success_rate": session.result(compact=True)["recent_success_rate"],
                   "training_seconds": session.training_seconds,
                   "success_100_milestone": session.success_100_milestone,
                   "seconds": round(monotonic() - started, 2), "metrics": session.agent.metrics}
        print(json.dumps(summary), flush=True)
        results.append({**summary, "resolved_settings": session.agent.settings,
                        "history": session.history, "frames": frames})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"map": maze, "settings": settings,
                                          "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
