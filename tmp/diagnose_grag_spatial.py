"""Read-only reproduction of GRAG's first contrastive update and Actor capacity."""
import json
import os
import sys
from pathlib import Path
from collections import Counter, deque
from copy import deepcopy

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from agents.grag_impl.grag import GRAG, Transition, state_action_encoder
from agents.grag_impl.components.language import encode_language

torch.set_num_threads(1)
root = Path(__file__).resolve().parents[1]
saved = json.loads((root/'GridWorld/data/latest_result.json').read_text(encoding='utf-8'))
m = saved['map']
agent = GRAG(saved['settings'])
obs = {**m, 'position': m['start']}
agent.initialize_networks(obs)
cells = [(r,c) for r in range(m['rows']) for c in range(m['cols'])
         if [r,c] not in m['obstacles'] and [r,c] != m['goal']]
ids = {p:i for i,p in enumerate(cells)}
goal = tuple(m['goal'])
deltas = [(-1,0),(1,0),(0,-1),(0,1)]
def move(p,a):
    q = (p[0]+deltas[a][0],p[1]+deltas[a][1])
    return q if q in ids or q == goal else p
states = torch.stack([agent.observation_to_tensor({**m,'position':p}) for p in cells])
prompt = agent.generate_goal_prompt(obs)
agent._goal_embedding_cache[prompt] = encode_language(prompt).detach().cpu()
def tr(i,a,success=False,end=False):
    with torch.no_grad(): old = agent.actor(states[i]).log_softmax(-1)[a].item()
    return Transition(states[i],a,old,0.,float(success),0.,success,end,prompt)
with torch.no_grad(): probs = agent.actor(states).softmax(-1).numpy()
rng = np.random.default_rng(42)
positive = []
records = []
episodes = total = 0
while sum(t.reached_goal for t in positive) < 16 and episodes < 10000:
    p = tuple(m['start']); trajectory = []
    for step in range(saved['settings']['max_steps']):
        i = ids[p]; a = int(rng.choice(4,p=probs[i])); q = move(p,a)
        trajectory.append((i,a,q == goal)); total += 1; p=q
        if p == goal: break
    records.extend(trajectory)
    episodes += 1
    if p == goal:
        positive.extend(tr(i,a,s,s) for i,a,s in trajectory)
queries = [tr(i,0) for i in range(len(cells))]
distance={goal:0}; todo=deque([goal])
while todo:
    p=todo.popleft()
    for dr,dc in deltas:
        q=(p[0]+dr,p[1]+dc)
        if q in ids and q not in distance:
            distance[q]=distance[p]+1; todo.append(q)
teacher = torch.tensor([[float(distance.get(move(p,a),999)<distance[p]) for a in range(4)] for p in cells])
print('sample',dict(episodes=episodes,steps=total,positive_transitions=len(positive)),flush=True)
def stats(label):
    with torch.no_grad():
        features = agent._crl_state_features(states)
        inputs = torch.cat((features[:,None,:].expand(-1,4,-1),torch.eye(4)[None].expand(len(cells),-1,-1)),dim=-1)
        scores = (agent.sa_encoder(inputs)*agent.goal_encoder(agent._goal_embedding_cache[prompt])[None,None,:]).sum(-1)
        p = agent.actor(states).softmax(-1)
    out = {'actor_mean':p.mean(0).tolist(), 'actor_position_range':(p.max(0).values-p.min(0).values).tolist(),
           'q_greedy_action_counts':dict(Counter(scores.argmax(-1).tolist())),
           'q_correct_greedy_fraction':teacher.gather(1,scores.argmax(-1)[:,None]).mean().item(),
           'q_blocked_greedy_fraction':sum(move(pos,int(a))==pos for pos,a in zip(cells,scores.argmax(-1)))/len(cells)}
    print(label,json.dumps(out),flush=True)
stats('initial')
for step in range(1,17):
    loss=agent._compute_crl_loss(positive); agent._update_crl_encoders(loss)
    if step in (1,16):
        stats('encoder_steps_'+str(step))
        print('loss',float(loss.detach()),flush=True)
        if step == 1:
            original_actor = deepcopy(agent.actor.state_dict())
            original_rng = agent._generator.get_state()
            with torch.no_grad():
                features = agent._crl_state_features(states)
                goals = agent._goal_embedding_cache[prompt].unsqueeze(0).expand(len(cells), -1)
                q = agent._compute_crl_q(features, goals)
                advantage = q-(torch.from_numpy(probs)*q).sum(-1,keepdim=True)
            data=[Transition(states[i],a,float(np.log(probs[i,a])),0.,0.,0.,success,False,prompt)
                  for i,a,success in records]
            adv=torch.tensor([advantage[i,a].item() for i,a,_ in records])
            agent._update_ppo_actor(data,adv)
            stats('after_one_online_update_actor16')
            agent.actor.load_state_dict(original_actor)
            del agent._grag_actor_optimizer
            agent._generator.set_state(original_rng)


# Independent capacity check: same Actor architecture, exact shortest-path teacher.
alltr = [tr(i,a) for i in range(len(cells)) for a in range(4)]
for _ in range(100):
    with torch.no_grad():
        p=agent.actor(states).softmax(-1)
        adv=(teacher-(p*teacher).sum(-1,keepdim=True)).flatten()
        logs=p.log()
        for i,t in enumerate(alltr): t.old_log_prob=logs[i//4,i%4].item()
    agent._update_ppo_actor(alltr,adv)
with torch.no_grad():
    p=agent.actor(states).softmax(-1)
    accuracy=teacher.gather(1,p.argmax(-1)[:,None]).mean().item()
print('actor_with_oracle_teacher',json.dumps({'correct_greedy_fraction':accuracy,'position_range':(p.max(0).values-p.min(0).values).tolist()}),flush=True)
