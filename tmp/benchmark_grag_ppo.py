import json
import time
from pathlib import Path
import torch
from agents.grag import GRAG
from agents.ppo_agent import PPOAgent
from gridworld.world import GridWorld

torch.set_num_threads(1)
map_data = json.loads(Path('GridWorld/data/world.json').read_text(encoding='utf8'))
settings = {'seed': 42, 'max_steps': 300}
for kind in (GRAG, PPOAgent):
    world = GridWorld(map_data)
    agent = kind(settings.copy())
    if kind is GRAG:
        agent.initialize_for_gridworld(world.map)
    timings = {}
    for name in (('_update_contrastive', '_update_ppo') if kind is GRAG else ('_update',)):
        original = getattr(agent, name)
        timings[name] = [0, 0.]
        def wrapped(*args, original=original, name=name, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                timings[name][0] += 1
                timings[name][1] += time.perf_counter() - start
        setattr(agent, name, wrapped)
    observation = world.reset()
    episode_steps = successes = episodes = 0
    start = time.perf_counter()
    for _ in range(4096):
        action = agent.choose_action(observation)
        next_obs, reached, _ = world.step(action)
        agent.learn(observation, action, float(reached), next_obs, reached)
        episode_steps += 1
        observation = next_obs
        if reached or episode_steps == 300:
            agent.end_episode()
            observation = world.reset()
            episode_steps = 0
            successes += reached
            episodes += 1
    print(json.dumps({'agent': kind.__name__, 'seconds': time.perf_counter() - start,
                      'episodes': episodes, 'successes': successes, 'updates': timings}), flush=True)
