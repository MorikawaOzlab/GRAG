"""On-policy PPO rollout collection in spawned CPU processes.

Workers use the same frozen policy. Only the parent updates the shared policy;
independently trained parameter sets are never averaged.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import cloudpickle


def worker_count(requested, episodes):
    available = max(1, (os.cpu_count() or 1) - 2)
    if type(requested) is not int or not 0 <= requested <= (os.cpu_count() or 1):
        raise ValueError('Invalid CPU worker count')
    return min(requested or min(4, available), episodes)


def collect_episode(settings, policy, episode_index):
    import torch
    from .environment import BabyAIEnvironment
    from .ppo_agent import PPOAgent
    torch.set_num_threads(1)
    world = BabyAIEnvironment(env_id=settings.get('env_id', 'BabyAI-GoToRedBall-v0'),
        seed=int(settings.get('seed', 0)), mission=settings.get('mission', ''),
        goal=settings.get('goal'), max_steps=int(settings.get('max_steps', 256)))
    world.episode = episode_index
    try:
        obs = world.reset()
        agent = PPOAgent(obs, (int(settings.get('seed', 0)) + episode_index) % (2**32))
        torch.set_num_threads(1)
        for name, state in cloudpickle.loads(policy).items():
            getattr(agent, name).load_state_dict(state)
        transitions, total, steps = [], 0., 0
        while True:
            action = agent.choose_action(obs)
            next_obs, reward, terminated, truncated, success = world.step(action)
            with torch.no_grad():
                _, value = agent.policy(agent.encode(next_obs))
            transitions.append((*agent.pending, reward, 0. if terminated else value.item(), terminated or truncated))
            total += reward
            steps += 1
            obs = next_obs
            if terminated or truncated:
                break
        return cloudpickle.dumps({'transitions': transitions, 'reward': total,
            'steps': steps, 'success': success, 'world': world, 'observation': obs,
            'action': action})
    finally:
        world.close()


def train_parallel(session, episodes, stop):
    count = worker_count(session.settings.get('cpu_workers', 0), episodes)
    session.execution = {'mode': 'parallel', 'workers': count}
    session.published = session.snapshot()
    # Preserve an unfinished observed episode before starting parallel episodes.
    completed = 0
    if session.steps and not session.ended:
        while not session.ended and not stop.is_set():
            session.step()
        completed += int(session.ended)
    if stop.is_set() or completed >= episodes:
        return
    session.agent.update()
    with ProcessPoolExecutor(max_workers=count, mp_context=get_context('spawn')) as pool:
        while completed < episodes and not stop.is_set():
            batch = min(count, episodes-completed)
            policy = cloudpickle.dumps({name: getattr(session.agent, name).state_dict()
                                       for name in ('net', 'actor', 'critic')})
            first = session.world.episode
            futures = [pool.submit(collect_episode, session.settings, policy, first+i) for i in range(batch)]
            # Finish this on-policy batch before updating. Pause takes effect at
            # the batch boundary; no in-flight samples are discarded.
            results = [cloudpickle.loads(f.result()) for f in futures]
            for result in results:
                session.agent.buffer.extend(result['transitions'])
                session.total_steps += result['steps']
                session.history.append({'episode':len(session.history)+1,
                    'reward':result['reward'], 'steps':result['steps'],
                    'success':bool(result['success']), 'total_steps':session.total_steps})
            session.agent.update()
            last = results[-1]
            session.world.close()
            session.world, session.obs = last['world'], last['observation']
            session.world.episode = first + batch
            session.steps, session.reward = last['steps'], last['reward']
            session.action, session.ended = last['action'], True
            completed += batch
            session.published = session.snapshot()
