"""Compare SGCRL updates with legacy sampling/Adam versus replay cache/fused Adam.

This measures update throughput on synthetic features, not time to learn a task.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from agents.grag_impl.grag_sgcrl import SGCRL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.updates, args.trials, args.threads) < 1:
        parser.error('updates, trials and threads must be positive')
    torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(99)
    episodes = [(torch.rand(200, 800, generator=generator),
                 torch.randint(4, (200,), generator=generator),
                 torch.rand(200, 800, generator=generator)) for _ in range(50)]
    agents = {}
    for name, cache, shared in (('legacy_sampling_adam', 0, False),
                                ('cached_fused', 256, False),
                                ('cached_fused_shared', 256, True)):
        agent = SGCRL({'seed': 7, 'crl_device': args.device, 'crl_min_replay_size': 2,
                       'crl_batch_size': args.batch_size, 'crl_replay_cache_mb': cache,
                       'crl_share_actor_features': shared})
        agent.initialize_networks(800, 800, 4, discrete=True)
        agent._episodes, agent._replay_size = deque(episodes), 10000
        if cache == 0:
            for optimizer in (agent.actor_optimizer, agent.encoder_optimizer, agent.alpha_optimizer):
                for group in optimizer.param_groups:
                    group['fused'] = False
        for _ in range(10):
            agent.update()
        agents[name] = agent
    times = {name: [] for name in agents}
    for trial in range(args.trials):
        order = list(agents) if trial % 2 == 0 else list(reversed(agents))
        for name in order:
            agent = agents[name]
            if agent.device.type == 'cuda':
                torch.cuda.synchronize(agent.device)
            started = perf_counter()
            for _ in range(args.updates):
                agent.update()
            if agent.device.type == 'cuda':
                torch.cuda.synchronize(agent.device)
            times[name].append(perf_counter() - started)
    seconds = {name: statistics.median(values) for name, values in times.items()}
    result = {'device': str(agent.device), 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name(agent.device) if agent.device.type == 'cuda' else None,
              'batch_size': args.batch_size, 'state_dim': 800, 'goal_dim': 800,
              'hidden_size': 256, 'repr_dim': 64, 'replay_size': 10000,
              'updates_per_trial': args.updates, 'seconds': times,
              'median_updates_per_second': {name: args.updates / value for name, value in seconds.items()},
              'speedup': seconds['legacy_sampling_adam'] / seconds['cached_fused_shared']}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
