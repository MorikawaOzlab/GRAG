import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'GridWorld')]
import torch
from gridworld.storage import DEFAULT_SETTINGS
from gridworld.training_runtime import TrainingSession

parser = argparse.ArgumentParser()
parser.add_argument('--variants', nargs='+', default=['baseline', 'raw_advantage', 'more_data', 'raw_more_data'])
parser.add_argument('--episodes', type=int, default=100)
parser.add_argument('--seeds', type=int, nargs='+', default=[42])
args = parser.parse_args()
torch.set_num_threads(1)

def phased_update(self):
    F = torch.nn.functional
    rows, goals, ids = self._hindsight()
    states = torch.tensor([r['state'] for r in rows])
    actions = torch.tensor([r['action'] for r in rows])
    old_logs = torch.tensor([r['log_prob'] for r in rows])
    old_probs = self._distribution(states).probs.detach()
    contrastive_losses = []
    policy_losses = []
    for _ in range(self.config['epochs']):
        for ix in torch.randperm(len(rows), generator=self.generator).split(self.config['batch_size']):
            candidates = ix
            if len(ids[ix].unique()) == 1:
                other = torch.nonzero(ids != ids[ix[0]]).flatten()
                candidates = torch.cat([ix, other[torch.randint(len(other), (1,), generator=self.generator)]])
            phi = self.sa_encoder(torch.cat([states[ix], F.one_hot(actions[ix], 4).float()], -1))
            psi = self.goal_encoder(goals[candidates])
            logits = -torch.linalg.vector_norm(phi[:, None] - psi[None], dim=-1)
            labels = torch.arange(len(ix))
            allowed = ids[ix, None] != ids[candidates][None]
            allowed[labels, labels] = True
            loss = F.cross_entropy(logits.masked_fill(~allowed, -torch.inf), labels)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.sa_encoder.parameters()) + list(self.goal_encoder.parameters()), .5)
            self.optimizer.step()
            contrastive_losses.append(float(loss.detach()))
    advantage = self._advantages(states, actions)
    for _ in range(self.config['epochs']):
        for ix in torch.randperm(len(rows), generator=self.generator).split(self.config['batch_size']):
            dist = self._distribution(states[ix])
            ratio = (dist.log_prob(actions[ix]) - old_logs[ix]).exp()
            adv = advantage[ix]
            if 'raw' not in self.variant and len(ix) > 1:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            loss = -torch.minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv).mean() - self.config['entropy_coef'] * dist.entropy().mean()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), .5)
            self.optimizer.step()
            policy_losses.append(float(loss.detach()))
    self.updates += 1
    with torch.no_grad():
        current = self._distribution(states)
        kl = (old_probs * (old_probs.log() - current.logits)).sum(-1).mean()
    self.metrics = {'updates': self.updates, 'contrastive_loss': sum(contrastive_losses)/len(contrastive_losses),
                    'entropy': float(current.entropy().mean()), 'kl': float(kl),
                    'advantage_std': float(advantage.std(unbiased=False))}
maze = json.loads((ROOT / 'tmp/cppo_wall_map.json').read_text(encoding='utf-8'))
source = (ROOT / 'tmp/cppo_before_investigation.py').read_text(encoding='utf-8')
results = []
for variant in args.variants:
    code = source
    if 'raw' in variant:
        code = code.replace('if len(indices) > 1:', 'if False:')
    if 'scaled_goal' in variant:
        code = code.replace('self.torch.tensor(self.map["goal"], dtype=self.torch.float32)', 'self.torch.tensor(self.map["goal"], dtype=self.torch.float32) / self.torch.tensor([max(1, self.map["rows"] - 1), max(1, self.map["cols"] - 1)]) * 2 - 1')
        code = code.replace('torch.tensor([self.map["goal"]], dtype=torch.float32)', 'torch.tensor([self.map["goal"]], dtype=torch.float32) / torch.tensor([max(1, self.map["rows"] - 1), max(1, self.map["cols"] - 1)]) * 2 - 1')
        code = code.replace('self.goal_encoder(goals[candidates])', 'self.goal_encoder(goals[candidates] / torch.tensor([max(1, self.map["rows"] - 1), max(1, self.map["cols"] - 1)]) * 2 - 1)')
    if 'warmup' in variant:
        code = code.replace('entropy = distribution.entropy().mean()', 'policy = policy * (self.updates >= 5)\n                entropy = distribution.entropy().mean()')
    module = types.ModuleType('cppo_' + variant)
    exec(compile(code, str(ROOT / 'tmp/cppo_before_investigation.py'), 'exec'), module.__dict__)
    if 'phased' in variant:
        module.CPPOAgent._update = phased_update
    module.CPPOAgent.variant = variant
    for seed in args.seeds:
        settings = {**DEFAULT_SETTINGS, 'seed': seed, 'max_steps': 200}
        if 'more_data' in variant:
            settings['cppo_rollout_steps'] = 4096
        if 'paper_scale' in variant:
            settings['cppo_rollout_steps'] = 65536
        if 'entropy' in variant:
            settings['cppo_entropy_coef'] = .1
        if 'strong_entropy' in variant:
            settings['cppo_entropy_coef'] = 1.
        session = TrainingSession(maze, settings, module.CPPOAgent)
        curve = []
        started = time.monotonic()
        for episode in range(1, args.episodes + 1):
            session.run_batch(1)
            if episode % 100 == 0 or episode == args.episodes:
                visited = sum(v > 0 for row in session.visits for v in row)
                obs = {**maze, 'position': [7, 3]}
                probs = session.agent.action_probabilities(obs)
                entropy = -sum(p * math.log(max(p, 1e-30)) for p in probs.values())
                result = {'variant': variant, 'seed': seed, 'episode': episode,
                          'successes': session.successes, 'visited': visited,
                          'wall_policy': probs, 'wall_entropy': entropy,
                          'metrics': session.agent.metrics, 'seconds': round(time.monotonic()-started, 2)}
                curve.append(result)
                print(json.dumps(result), flush=True)
        results.append({'variant': variant, 'seed': seed, 'settings': session.agent.settings,
                        'history': session.history, 'diagnostics': curve})
        path = ROOT / ('output/cppo/diagnosis_' + '_'.join(args.variants) + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'map': maze, 'results': results}, indent=2), encoding='utf-8')
