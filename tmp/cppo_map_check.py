import json
import random
from pathlib import Path
from collections import deque
m = json.loads(Path('tmp/cppo_wall_map.json').read_text())
start, goal = tuple(m['start']), tuple(m['goal'])
walls = set(map(tuple, m['obstacles']))
def move(p, a):
    dr, dc = [(-1,0),(1,0),(0,-1),(0,1)][a]
    n = p[0]+dr, p[1]+dc
    return n if 0 <= n[0] < m['rows'] and 0 <= n[1] < m['cols'] and n not in walls else p
q = deque([(start, 0)])
seen = {start}
shortest = None
while q:
    p, d = q.popleft()
    if p == goal:
        shortest = d
    for a in range(4):
        n = move(p, a)
        if n not in seen:
            seen.add(n)
            q.append((n, d+1))
rng = random.Random(42)
successes = 0
visited = set()
first = None
for ep in range(1, 10001):
    p = start
    for t in range(200):
        p = move(p, rng.randrange(4))
        visited.add(p)
        if p == goal:
            successes += 1
            first = first or ep
            break
print(json.dumps({'shortest': shortest, 'reachable': len(seen), 'random_10000_successes': successes,
                  'first_random_success_episode': first, 'visited': len(visited)}))
