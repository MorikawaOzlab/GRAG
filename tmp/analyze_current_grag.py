import json
from collections import Counter
from pathlib import Path
import torch
from agents.grag_impl.grag import GRAG
from gridworld.world import GridWorld

torch.set_num_threads(1)
data = json.loads(Path('GridWorld/data/latest_result.json').read_text(encoding='utf-8'))
world = GridWorld(data['map'])
agent = GRAG({'seed': 42})
agent.initialize_for_gridworld(world.map)
cells = [(r, c) for r in range(world.map['rows']) for c in range(world.map['cols'])
         if [r, c] not in world.map['obstacles'] and [r, c] != world.map['goal']]

def stats(step, visits, hits, successes):
    probabilities = torch.tensor([list(agent.action_probabilities({**world.map, 'position': list(p)}).values()) for p in cells])
    actions = probabilities.argmax(dim=1).tolist()
    blocked = sum(world.transition(p, agent.actions[a])[2] for p, a in zip(cells, actions))
    with torch.no_grad():
        states = torch.stack([agent._prepare_state({**world.map, 'position': list(p)}).flatten() for p in cells])
        expanded = states.repeat_interleave(4, 0)
        vectors = agent.sa_encoder(torch.cat((expanded, torch.eye(4).repeat(len(cells), 1)), 1))
        target = agent.goal_encoder(torch.tensor([world.map['goal']], dtype=torch.float32))
        q = -torch.linalg.vector_norm(vectors - target, dim=1).reshape(-1, 4)
        q_actions = q.argmax(1).tolist()
    print(json.dumps({'step': step, 'visited': len(visits), 'collisions': hits, 'successes': successes,
                      'mean_probs': [round(x, 3) for x in probabilities.mean(0).tolist()],
                      'position_std': [round(x, 3) for x in probabilities.std(0).tolist()],
                      'greedy_actions': dict(Counter(agent.actions[a] for a in actions)),
                      'greedy_blocked': blocked, 'cells': len(cells),
                      'q_best_actions': dict(Counter(agent.actions[a] for a in q_actions)),
                      'q_blocked': sum(world.transition(p, agent.actions[a])[2] for p, a in zip(cells, q_actions)),
                      'q_action_range_mean': round((q.max(1).values - q.min(1).values).mean().item(), 4),
                      'contrastive_loss': agent.contrastive_loss, 'policy_loss': agent.policy_loss}), flush=True)

obs = world.reset()
visits = Counter()
hits = successes = steps_in_episode = 0
stats(0, visits, hits, successes)
for t in range(1, 4097):
    visits[tuple(obs['position'])] += 1
    action = agent.choose_action(obs)
    new, reached, collision = world.step(action)
    hits += collision
    successes += reached
    agent.learn(obs, action, float(reached), new, reached)
    obs = new
    steps_in_episode += 1
    if reached or steps_in_episode == 300:
        agent.end_episode()
        obs = world.reset()
        steps_in_episode = 0
    if t in (128, 512, 1024, 2048, 4096):
        stats(t, visits, hits, successes)
