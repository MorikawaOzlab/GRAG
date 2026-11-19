import json
import sys
from collections import Counter, deque
from pathlib import Path

import torch
from agents.grag_impl.grag import GRAG
from gridworld.world import GridWorld

torch.set_num_threads(1)
data = json.loads(Path('GridWorld/data/training_sessions/b1c1bd0d60c6437b8496dc727af8d7d2.json').read_text(encoding='utf8'))
world = GridWorld(data['map'])
agent = GRAG({'seed': int(sys.argv[1]), 'hindsight_gamma': .99, 'grag_update_interval': 1280})
agent.initialize_for_gridworld(world.map)
goal = tuple(world.map['goal'])
distance = {goal: 0}
todo = deque([goal])
while todo:
    p = todo.popleft()
    for action in agent.actions:
        target = world.transition(p, action)[0]
        if target not in distance:
            distance[target] = distance[p] + 1
            todo.append(target)
cells = sorted(set(distance) - {goal})
obs = world.reset()
length = successes = collisions = 0
for step in range(1, 10241):
    action = agent.choose_action(obs)
    new, reached, collision = world.step(action)
    agent.learn(obs, action, float(reached), new, reached)
    length += 1
    successes += reached
    collisions += collision
    obs = new
    if reached or length == 600:
        agent.end_episode()
        obs = world.reset()
        length = 0
    if step in (1280, 5120, 10240):
        states = torch.stack([agent._prepare_state({**world.map, 'position': p}).flatten() for p in cells])
        with torch.no_grad():
            probs = agent.actor(states).softmax(-1)
            q = agent.q_values(states, torch.tensor([goal] * len(cells)).float())
        actors = probs.argmax(1).tolist()
        qs = q.argmax(1).tolist()
        optimal = [[distance[world.transition(p, a)[0]] for a in agent.actions] for p in cells]
        print(json.dumps({'seed': agent.settings['seed'], 'step': step, 'successes': successes,
                          'collisions': collisions, 'actor_mode': Counter(agent.actions[a] for a in actors),
                          'q_correct': sum(optimal[i][a] == min(optimal[i]) for i,a in enumerate(qs)),
                          'actor_correct': sum(optimal[i][a] == min(optimal[i]) for i,a in enumerate(actors)),
                          'walls_actor': sum(world.transition(p, agent.actions[a])[2] for p,a in zip(cells,actors)),
                          'cells': len(cells), 'contrastive_loss': agent.contrastive_loss}), flush=True)
