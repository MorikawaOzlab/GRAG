import json
from collections import deque
from pathlib import Path
import sys

import numpy as np
import torch
from agents.grag_impl.grag import GRAG
from gridworld.world import GridWorld

torch.set_num_threads(1)
world = GridWorld(json.loads(Path('GridWorld/data/world.json').read_text(encoding='utf8')))
seed = int(sys.argv[1])
agent = GRAG({'seed': seed, 'hindsight_gamma': float(sys.argv[2])})
agent.initialize_for_gridworld(world.map)
goal = tuple(world.map['goal'])
dist = {goal: 0}
queue = deque([goal])
while queue:
    position = queue.popleft()
    for action in agent.actions:
        target = world.transition(position, action)[0]
        if target not in dist:
            dist[target] = dist[position] + 1
            queue.append(target)
cells = sorted(set(dist) - {goal})
indices = {p: i for i, p in enumerate(cells)}
obs = world.reset()
steps = successes = collisions = 0
for t in range(1, 8193):
    action = agent.choose_action(obs)
    nxt, reached, collision = world.step(action)
    agent.learn(obs, action, float(reached), nxt, reached)
    collisions += collision
    successes += reached
    steps += 1
    obs = nxt
    if reached or steps == 300:
        agent.end_episode()
        obs = world.reset()
        steps = 0

states = torch.stack([agent._prepare_state({**world.map, 'position': p}).flatten() for p in cells])
with torch.no_grad():
    probs = agent.actor(states).softmax(-1).numpy()
    q = agent.q_values(states, torch.tensor([goal] * len(cells)).float()).numpy()
transitions = [[world.transition(p, a)[0] for a in agent.actions] for p in cells]
matrix = np.zeros((len(cells), len(cells)))
direct = np.zeros(len(cells))
for i in range(len(cells)):
    for a, target in enumerate(transitions[i]):
        if target == goal:
            direct[i] += probs[i, a]
        else:
            matrix[i, indices[target]] += probs[i, a]
# Discounted probability of hitting the goal, conditional on each first action.
discount = .99
value = np.linalg.solve(np.eye(len(cells)) - discount * matrix, direct)
true_q = np.array([[1.0 if p == goal else discount * value[indices[p]] for p in row] for row in transitions])
best_true = np.isclose(true_q, true_q.max(axis=1, keepdims=True), rtol=1e-6, atol=1e-8)
best_shortest = np.array([[dist[p] for p in row] for row in transitions])
pred = q.argmax(axis=1)
print(json.dumps({'seed': seed, 'gamma': agent.hindsight_gamma, 'successes': successes,
                  'collisions': collisions, 'matches_current_policy_q': int(best_true[np.arange(len(cells)), pred].sum()),
                  'matches_shortest': int((best_shortest[np.arange(len(cells)), pred] == best_shortest.min(axis=1)).sum()),
                  'q_argmax_wall': int(sum(world.transition(p, agent.actions[pred[i]])[2] for i,p in enumerate(cells))),
                  'actor_argmax_wall': int(sum(world.transition(p, agent.actions[probs[i].argmax()])[2] for i,p in enumerate(cells))),
                  'examples': [{'position': p, 'q': np.round(q[indices[p]], 3).tolist(),
                                'true_q': np.round(true_q[indices[p]], 5).tolist(),
                                'next_distance': best_shortest[indices[p]].tolist(),
                                'probs': np.round(probs[indices[p]], 3).tolist()} for p in [(5,0),(5,5),(1,5)]]}))
