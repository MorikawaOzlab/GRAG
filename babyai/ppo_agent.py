"""Language-conditioned PPO, with time-limit bootstrapping and clipped loss."""
import torch
from torch import nn
from torch.distributions import Categorical
from .ollama_agent import ACTIONS


class PPOAgent:
    def __init__(self, observation, seed=0):
        torch.manual_seed(seed)
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        self.net = nn.Sequential(nn.Linear(len(self.encode(observation)), 128), nn.Tanh(), nn.Linear(128,128), nn.Tanh())
        self.actor, self.critic = nn.Linear(128,7), nn.Linear(128,1)
        self.optimizer = torch.optim.Adam(list(self.net.parameters())+list(self.actor.parameters())+list(self.critic.parameters()), lr=3e-4)
        self.buffer = []

    def encode(self, obs):
        # UTF-8 bytes preserve language order; all agents receive the same mission.
        raw = obs['mission'].encode('utf-8')
        if len(raw) > 512:
            raise ValueError('命令はUTF-8で512バイト以内にしてください')
        carried = obs.get('carrying') or {}
        carrying = [float(carried.get('type') == v) for v in ('key','ball','box')] + [float(carried.get('color') == v) for v in ('red','green','blue','purple','yellow','grey')]
        image = [v/10 for column in obs['image'] for cell in column for v in cell]
        return torch.tensor(image + carrying + [obs['direction']/3] + [v/255 for v in raw] + [0.]*(512-len(raw)), dtype=torch.float32)

    def policy(self, states):
        hidden = self.net(states)
        return Categorical(logits=self.actor(hidden)), self.critic(hidden).squeeze(-1)

    def choose_action(self, observation):
        with torch.no_grad():
            dist, value = self.policy(self.encode(observation))
            action = dist.sample()
        self.pending = (self.encode(observation), action, dist.log_prob(action), value)
        return ACTIONS[action.item()]

    def learn(self, observation, action, reward, next_observation, terminated, truncated):
        with torch.no_grad():
            _, next_value = self.policy(self.encode(next_observation))
        self.buffer.append((*self.pending, reward, 0. if terminated else next_value.item(), terminated or truncated))
        if len(self.buffer) >= 128:
            self.update()

    def update(self):
        if not self.buffer:
            return
        advantages, running = [], 0.
        for _, _, _, value, reward, next_value, end in reversed(self.buffer):
            running = reward + .99*next_value - value.item() + .99*.95*running*(not end)
            advantages.append(running)
        advantages = torch.tensor(advantages[::-1])
        states = torch.stack([r[0] for r in self.buffer])
        actions = torch.stack([r[1] for r in self.buffer])
        old_logs = torch.stack([r[2] for r in self.buffer])
        returns = advantages + torch.stack([r[3] for r in self.buffer])
        advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
        parameters = list(self.net.parameters())+list(self.actor.parameters())+list(self.critic.parameters())
        for _ in range(4):
            for indices in torch.randperm(len(states)).split(32):
                dist, values = self.policy(states[indices])
                ratio = (dist.log_prob(actions[indices])-old_logs[indices]).exp()
                loss = -torch.minimum(ratio*advantages[indices], ratio.clamp(.8,1.2)*advantages[indices]).mean() + .5*(values-returns[indices]).square().mean() - .01*dist.entropy().mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, .5)
                self.optimizer.step()
        self.buffer.clear()

    def save(self, path):
        torch.save({'net':self.net.state_dict(), 'actor':self.actor.state_dict(), 'critic':self.critic.state_dict(), 'optimizer':self.optimizer.state_dict(), 'buffer':self.buffer, 'rng':torch.get_rng_state()}, path)

    def load(self, path):
        data = torch.load(path, map_location='cpu', weights_only=False)
        for key in ('net','actor','critic','optimizer'):
            getattr(self,key).load_state_dict(data[key])
        self.buffer = data['buffer']
        torch.set_rng_state(data['rng'])
