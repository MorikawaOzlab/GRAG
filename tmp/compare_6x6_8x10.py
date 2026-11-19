import json
import sys
from collections import Counter
from pathlib import Path

import torch
from agents.grag_impl.grag import GRAG
from gridworld.world import GridWorld

torch.set_num_threads(1)
map_id = {'6': 'b2d366d50f4b43028aa63109ae7c4bec',
          '8': 'ff16c6e1b9ab489a9c351c2f293844d3'}[sys.argv[1]]
data = json.loads((Path('GridWorld/data/training_sessions') / (map_id + '.json')).read_text(encoding='utf8'))
world = GridWorld(data['map'])
agent = GRAG({'seed': int(sys.argv[2]), 'grag_update_interval': 1280, 'hindsight_gamma': .99})
agent.initialize_for_gridworld(world.map)
episode = successes = collisions = length = 0
obs = world.reset()
for step in range(1, 20481):
    action = agent.choose_action(obs)
    nxt, reached, collision = world.step(action)
    agent.learn(obs, action, float(reached), nxt, reached)
    length += 1
    collisions += collision
    successes += reached
    obs = nxt
    if reached or length == 70:
        episode += 1
        agent.end_episode()
        obs = world.reset()
        length = 0
    if step in (5120, 10240, 20480):
        states = torch.stack([agent._prepare_state({**world.map, 'position': [r, c]}).flatten()
                              for r in range(world.map['rows']) for c in range(world.map['cols'])
                              if [r,c] not in world.map['obstacles']])
        with torch.no_grad():
            probs = agent.actor(states).softmax(-1)
            best = Counter(agent.actions[a] for a in probs.argmax(-1).tolist())
        print(json.dumps({'size':sys.argv[1], 'seed':sys.argv[2], 'steps':step, 'episodes':episode,
                          'successes':successes,'collisions':collisions,'actor_best':best}),flush=True)
