import cProfile
from collections import deque
import io
import json
from pathlib import Path
import pstats
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from agents.grag_impl.grag_sgcrl import SGCRL

torch.set_num_threads(1)
agent = SGCRL({'seed': 7, 'crl_device': 'cuda', 'crl_updates_per_step': 0,
               'crl_min_replay_size': 2})
agent.initialize_networks(800, 800, 4, discrete=True)
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(99)
    agent._episodes = deque((torch.rand(200, 800), torch.randint(4, (200,)), torch.rand(200, 800))
                            for _ in range(50))
agent._replay_size = 10000
for _ in range(5): agent.update()
torch.cuda.synchronize()
start = perf_counter()
for _ in range(100): agent.update()
torch.cuda.synchronize()
elapsed = perf_counter() - start
print(json.dumps({'100_updates_seconds': elapsed, 'updates_per_second': 100 / elapsed}), flush=True)
profiler = cProfile.Profile()
profiler.enable()
for _ in range(20): agent.update()
torch.cuda.synchronize()
profiler.disable()
output = io.StringIO()
pstats.Stats(profiler, stream=output).sort_stats('cumtime').print_stats(18)
print(output.getvalue())
