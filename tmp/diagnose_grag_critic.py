"""Compare GRAG with a shortest-path action ranking in the same PPO update."""
import inspect
import json
import sys
import textwrap
from collections import Counter, deque
from pathlib import Path
from types import MethodType

import torch
from agents.grag_impl.grag import GRAG
from gridworld.world import GridWorld


def run(seed, mode):
    world = GridWorld(json.loads(Path("GridWorld/data/world.json").read_text("utf8")))
    agent = GRAG({"seed": seed})
    agent.initialize_for_gridworld(world.map)
    goal = tuple(world.map["goal"])
    distance = {goal: 0}
    queue = deque([goal])
    while queue:
        p = queue.popleft()
        for a in agent.actions:
            q = world.transition(p, a)[0]
            if q not in distance:
                distance[q] = distance[p] + 1
                queue.append(q)
    cells = sorted(set(distance) - {goal})
    def scores(p):
        return [-distance[world.transition(p, a)[0]] for a in agent.actions]
    if mode == "warmup":
        original_ppo = agent._update_ppo
        def after_first_fit(rows):
            if agent.contrastive_optimizer.state:
                original_ppo(rows)
        agent._update_ppo = after_first_fit
    if mode == "oracle":
        # Diagnostic counterfactual: substitute only the action scores.
        source = textwrap.dedent(inspect.getsource(GRAG._update_ppo))
        line = "q_values = -torch.linalg.vector_norm(state_action_repr - goal_repr, dim=-1)"
        assert source.count(line) == 1
        ns = dict(torch=torch, nn=torch.nn, scores=scores)
        exec(source.replace(line, "q_values = states.new_tensor([scores(tuple(row['observation']['position'])) for row in transitions])"), ns)
        agent._update_ppo = MethodType(ns["_update_ppo"], agent)

    @torch.no_grad()
    def evaluate(positions, target):
        s = torch.stack([agent._prepare_state({**world.map, "position": p, "goal": target}).flatten() for p in positions])
        policy = agent.actor(s).softmax(-1)
        x = torch.cat((s.repeat_interleave(4, 0), torch.eye(4).repeat(len(s), 1)), 1)
        q = -torch.linalg.vector_norm(agent.sa_encoder(x) - agent.goal_encoder(torch.tensor(target).float()), dim=1)
        return policy, q.reshape(-1, 4)

    obs = world.reset()
    length = successes = 0
    for step in range(1, 8193):
        action = agent.choose_action(obs)
        new, done, _ = world.step(action)
        agent.learn(obs, action, float(done), new, done)
        length += 1
        successes += done
        obs = new
        if done or length == 300:
            agent.end_episode()
            obs = world.reset()
            length = 0
        if step in (4096, 8192):
            policy, q = evaluate(cells, goal)
            best = torch.tensor([scores(p) for p in cells])
            def correct(ranking):
                return int((best.gather(1, ranking.argmax(1)[:, None]) == best.max(1, keepdim=True).values).sum())
            target = tuple(world.map["start"])
            indices = [i for i, p in enumerate(cells) if p != target]
            switched, switched_q = evaluate([cells[i] for i in indices], target)
            direction, n = Counter(policy.argmax(1).tolist()).most_common(1)[0]
            print(seed, mode, step, "success", successes, "actor-opt", correct(policy), "learned-Q-opt", correct(q),
                  "mode", agent.actions[direction], n, "goal-switch actor/Q", int((policy[indices].argmax(1) != switched.argmax(1)).sum()),
                  int((q[indices].argmax(1) != switched_q.argmax(1)).sum()), flush=True)

if __name__ == "__main__":
    torch.set_num_threads(1)
    run(int(sys.argv[1]), sys.argv[2])
