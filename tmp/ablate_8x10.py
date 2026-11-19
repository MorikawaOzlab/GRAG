"""Diagnose 8x10 GRAG with temporary, in-process ablations."""
import json
import sys
from collections import Counter, deque
from pathlib import Path

import torch
from agents.grag_impl.grag import GRAG
from gridworld.world import GridWorld

torch.set_num_threads(1)
source = Path('GridWorld/data/training_sessions/b1c1bd0d60c6437b8496dc727af8d7d2.json')
world = GridWorld(json.loads(source.read_text(encoding='utf8'))['map'])
seed, mode = int(sys.argv[1]), sys.argv[2]
agent = GRAG({'seed': seed, 'hindsight_gamma': .99, 'grag_update_interval': 1280})
agent.initialize_for_gridworld(world.map)
goal = tuple(world.map['goal'])
dist = {goal: 0}
todo = deque([goal])
while todo:
    position = todo.popleft()
    for action in agent.actions:
        nxt = world.transition(position, action)[0]
        if nxt not in dist:
            dist[nxt] = dist[position] + 1
            todo.append(nxt)
cells = sorted(set(dist) - {goal})
original_q = agent.q_values
original_ppo = agent._update_ppo
original_hindsight = agent._sample_hindsight_goals
positive_stats = []

def tracked_hindsight(batch):
    goals = original_hindsight(batch)
    coordinates = [tuple(map(int, row)) for row in goals.tolist()]
    counts = Counter(coordinates)
    episode_goal_counts = Counter((row['episode_id'], coordinate) for row, coordinate in zip(batch, coordinates))
    duplicate_negatives = sum(counts[coordinate] - episode_goal_counts[(row['episode_id'], coordinate)]
                              for row, coordinate in zip(batch, coordinates))
    episode_counts = Counter(row['episode_id'] for row in batch)
    all_negatives = sum(len(batch) - episode_counts[row['episode_id']] for row in batch)
    positive_stats.append({'positive_final_goal': counts[goal], 'unique_goals': len(counts),
                           'largest_goal_share': max(counts.values()) / len(batch),
                           'episode_count': len({row['episode_id'] for row in batch}),
                           'identical_negative_fraction': duplicate_negatives / all_negatives if all_negatives else 0})
    return goals

agent._sample_hindsight_goals = tracked_hindsight

if mode == 'freeze_actor':
    agent._update_ppo = lambda rows: None
elif mode == 'oracle_q':
    def oracle_ppo(rows):
        def oracle(states, goals):
            state_grid = states.reshape(-1, 3, world.map['rows'], world.map['cols'])
            position_index = state_grid[:, 2].flatten(1).argmax(1).tolist()
            return states.new_tensor([[-dist[world.transition(divmod(i, world.map['cols']), a)[0]]
                                       for a in agent.actions] for i in position_index])
        agent.q_values = oracle
        try:
            original_ppo(rows)
        finally:
            agent.q_values = original_q
    agent._update_ppo = oracle_ppo

observed = Counter()
successes = collisions = completed = length = 0
obs = world.reset()
for step in range(1, 10241):
    observed[tuple(obs['position'])] += 1
    action = agent.choose_action(obs)
    nxt, reached, collision = world.step(action)
    agent.learn(obs, action, float(reached), nxt, reached)
    obs = nxt
    length += 1
    collisions += collision
    successes += reached
    if reached or length == 600:
        completed += 1
        agent.end_episode()
        obs = world.reset()
        length = 0
    if step in (1280, 5120, 10240):
        probes = [(1, 2), (3, 9), (0, 2)]
        states = torch.stack([agent._prepare_state({**world.map, 'position': p}).flatten() for p in cells])
        with torch.no_grad():
            probs = agent.actor(states).softmax(-1)
            q = original_q(states, torch.tensor([goal] * len(cells)).float())
        best = q.argmax(1).tolist()
        choice = probs.argmax(1).tolist()
        print(json.dumps({'seed': seed, 'mode': mode, 'step': step, 'success': successes,
                          'episodes': completed, 'collisions': collisions,
                          'q_optimal': sum(dist[world.transition(p, agent.actions[a])[0]] == dist[p] - 1 for p,a in zip(cells,best)),
                          'actor_optimal': sum(dist[world.transition(p, agent.actions[a])[0]] == dist[p] - 1 for p,a in zip(cells,choice)),
                          'actor_modes': Counter(agent.actions[a] for a in choice),
                          'unique_visited': len(observed), 'top_visited': observed.most_common(3),
                          'hindsight': positive_stats[-1] if positive_stats else None,
                          'probes': {str(p): {'q': [round(x, 3) for x in q[cells.index(p)].tolist()],
                                              'policy': [round(x, 3) for x in probs[cells.index(p)].tolist()]}
                                     for p in probes}}), flush=True)
