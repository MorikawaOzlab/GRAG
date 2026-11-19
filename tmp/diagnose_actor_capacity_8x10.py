import json
from collections import deque
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from agents.grag_impl.grag import GRAG
from gridworld.world import GridWorld

torch.set_num_threads(1)
data = json.loads(Path('GridWorld/data/training_sessions/b1c1bd0d60c6437b8496dc727af8d7d2.json').read_text(encoding='utf8'))
world = GridWorld(data['map'])
agent = GRAG({'seed': int(sys.argv[1])})
agent.initialize_for_gridworld(world.map)
goal = tuple(world.map['goal'])
distance = {goal: 0}
queue = deque([goal])
while queue:
    position = queue.popleft()
    for action in agent.actions:
        nxt = world.transition(position, action)[0]
        if nxt not in distance:
            distance[nxt] = distance[position] + 1
            queue.append(nxt)
cells = sorted(set(distance) - {goal})
states = torch.stack([agent._prepare_state({**world.map, 'position': p}).flatten() for p in cells])
valid = torch.tensor([[distance[world.transition(p, action)[0]] == distance[p]-1 for action in agent.actions] for p in cells])
target = valid.float() / valid.sum(1, keepdim=True)
optimizer = torch.optim.Adam(agent.actor.parameters(), lr=.0003)
for update in range(1001):
    if update in (0, 50, 200, 1000):
        with torch.no_grad():
            actions = agent.actor(states).argmax(1)
            correct = valid.gather(1, actions[:, None]).sum().item()
        print(json.dumps({'seed':agent.settings['seed'],'updates':update,'correct':correct,'cells':len(cells)}), flush=True)
    if update < 1000:
        logits = agent.actor(states)
        loss = -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
