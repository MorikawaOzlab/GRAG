import json
from pathlib import Path

import torch

from agents.grag_impl.grag import GRAG
from gridworld.world import GridWorld

torch.set_num_threads(1)
world = GridWorld(json.loads(Path('GridWorld/data/world.json').read_text(encoding='utf8')))
agent = GRAG({'seed': 42})
agent.initialize_for_gridworld(world.map)
obs = world.reset()
episode_steps = successes = updates = 0
previous_loss = None
for step in range(1, 16385):
    action = agent.choose_action(obs)
    next_obs, reached, collision = world.step(action)
    agent.learn(obs, action, float(reached), next_obs, reached)
    if step % 128 == 0 and agent.contrastive_loss is not None:
        updates += 1
    episode_steps += 1
    successes += reached
    obs = next_obs
    if reached or episode_steps == 300:
        agent.end_episode()
        obs = world.reset()
        episode_steps = 0
    if step % 1024 == 0:
        probe = {**world.map, 'position': [5, 5]}
        state = agent._prepare_state(probe).reshape(1, -1)
        goal = agent.target_goal_encoder(probe).reshape(1, -1)
        with torch.no_grad():
            q = agent.q_values(state, goal)[0].tolist()
            probabilities = torch.softmax(agent.actor(state), dim=-1)[0].tolist()
        print(json.dumps({'step': step, 'successes': successes, 'updates': updates,
                          'up_probability': round(probabilities[0], 4),
                          'right_probability': round(probabilities[3], 4),
                          'up_q': round(q[0], 3), 'right_q': round(q[3], 3),
                          'loss': agent.contrastive_loss}), flush=True)
