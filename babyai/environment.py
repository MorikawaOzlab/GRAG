"""Official BabyAI dynamics and separate user-editable completion rules."""
from .ollama_agent import ACTIONS
from .task import evaluate_goal, evaluate_task, parse_goal, validate_goal


class BabyAIEnvironment:
    def __init__(self, env_id='BabyAI-GoToRedBall-v0', seed=0, mission='', goal=None, max_steps=256):
        try:
            import gymnasium as gym
            import minigrid
        except ImportError as exc:
            raise RuntimeError('pip install -r babyai/requirements.txt を実行してください') from exc
        if not env_id.startswith('BabyAI-') or env_id not in gym.registry:
            raise ValueError('登録済みのBabyAI環境を指定してください')
        self.custom_mission = mission.strip()
        self.goal = validate_goal(goal if goal is not None else parse_goal(mission)) if mission.strip() else None
        self.env = gym.make(env_id, render_mode='rgb_array', max_steps=max_steps)
        self.seed, self.episode, self.max_steps = seed, 0, max_steps
        self.observation, self.memory = None, {}

    def _serialize(self, raw):
        return {'image': raw['image'].tolist(), 'direction': int(raw['direction']), 'carrying': self.state()['carrying'], 'mission': self.custom_mission or str(raw['mission'])}

    def reset(self):
        raw, _ = self.env.reset(seed=self.seed + self.episode)
        self.episode += 1
        self.memory = {}
        self.observation = self._serialize(raw)
        return self.observation

    def state(self):
        env = self.env.unwrapped
        def describe(obj, position=None):
            if obj is None:
                return None
            result = {'type': obj.type, 'color': obj.color}
            if position is not None:
                result['position'] = list(position)
            if obj.type == 'door':
                result.update(is_open=bool(obj.is_open), is_locked=bool(obj.is_locked))
            return result
        objects = [describe(env.grid.get(x,y), (x,y)) for x in range(env.width) for y in range(env.height) if env.grid.get(x,y) is not None]
        return {'objects': objects, 'position': [int(v) for v in env.agent_pos], 'front': [int(v) for v in env.front_pos], 'direction': int(env.agent_dir), 'carrying': describe(env.carrying)}

    def step(self, action):
        if action not in ACTIONS:
            raise ValueError(f'Invalid action: {action}')
        previous = self.observation
        before_position = tuple(self.env.unwrapped.agent_pos)
        raw, reward, terminated, truncated, info = self.env.step(ACTIONS.index(action))
        self.observation = self._serialize(raw)
        moved = tuple(self.env.unwrapped.agent_pos) != before_position
        self.observation['last_transition'] = {
            'action':action, 'moved':moved,
            'blocked_forward':action == 'forward' and not moved,
            'turned':previous['direction'] != self.observation['direction'],
            'inventory_changed':previous.get('carrying') != self.observation.get('carrying'),
        }
        if self.goal is not None:
            success = evaluate_goal(self.goal, self.state(), self.memory)
            terminated = bool(success)
            truncated = self.env.unwrapped.step_count >= self.max_steps and not terminated
            reward = float(success)
        else:
            success = evaluate_task(mission=previous['mission'], previous_observation=previous, action=action, observation=self.observation, reward=reward, terminated=terminated, truncated=truncated, info=info)
        return self.observation, float(reward), bool(terminated), bool(truncated), success

    def close(self):
        self.env.close()
