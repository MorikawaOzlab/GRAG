import json
import sys
from collections import Counter
from pathlib import Path

import torch
from agents.grag_impl.grag import GRAG
from gridworld.world import GridWorld

torch.set_num_threads(1)
data = json.loads(Path('GridWorld/data/training_sessions/b1c1bd0d60c6437b8496dc727af8d7d2.json').read_text(encoding='utf8'))
world = GridWorld(data['map'])
agent = GRAG({'seed': int(sys.argv[1]), 'hindsight_gamma': .99, 'grag_update_interval': 628})
agent.initialize_for_gridworld(world.map)
probe = {**world.map, 'position': [0, 2]}
state = agent._prepare_state(probe).reshape(1, -1)
goal = agent.target_goal_encoder(probe).reshape(1, -1)
original = agent._update_networks
count = successes = collisions = length = 0

def update(transitions):
    global count
    count += 1
    actor_before = [p.detach().clone() for p in agent.actor.parameters()]
    with torch.no_grad():
        before = agent.actor(state).softmax(-1)[0].clone()
    original(transitions)
    with torch.no_grad():
        after = agent.actor(state).softmax(-1)[0]
        q = agent.q_values(state, goal)[0]
        change = sum((a-b).square().sum().item() for a,b in zip(agent.actor.parameters(), actor_before)) ** .5
    if count == 1 or count % 5 == 0:
        positions = Counter(tuple(t['observation']['position']) for t in transitions)
        actions = Counter(agent.actions[t['action']] for t in transitions)
        print(json.dumps({'seed':agent.settings['seed'],'update':count,'step':count*628,
                          'contrastive_updated':agent.contrastive_loss is not None,
                          'actor_change_l2': round(change,7),
                          'policy_before': [round(x,4) for x in before.tolist()],
                          'policy_after': [round(x,4) for x in after.tolist()],
                          'q': [round(x,4) for x in q.tolist()],
                          'batch_actions': actions, 'top_position': positions.most_common(1),
                          'successes': successes, 'collisions': collisions}),flush=True)

agent._update_networks = update
obs = world.reset()
for step in range(1, 20100):
    action = agent.choose_action(obs)
    nxt, reached, collision = world.step(action)
    agent.learn(obs, action, float(reached), nxt, reached)
    obs = nxt
    collisions += collision
    successes += reached
    length += 1
    if reached or length == 300:
        agent.end_episode()
        obs = world.reset()
        length = 0
