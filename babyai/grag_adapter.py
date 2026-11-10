"""Replace/create your BabyAI GRAG in your own module; existing GRAG is untouched.

UI setting grag_entry is module:Class. Constructor receives settings dict.
choose_action(observation)->7-action name; optional learn(obs, action, reward,
next_obs, terminated, truncated), end_episode(), save(path), load(path).
Observation: image[x][y][object,color,state], direction, mission.
"""
import importlib


def create_grag(entry, settings):
    if not entry or ':' not in entry:
        raise ValueError('BabyAI対応GRAGの module:Class を指定してください')
    module, name = entry.split(':',1)
    if module in ('agents.grag', 'agents.grag_impl.grag'):
        raise ValueError('現行GridWorld版GRAGは未対応です。BabyAI対応版のモジュールを指定してください')
    agent = getattr(importlib.import_module(module), name)(settings)
    if not callable(getattr(agent,'choose_action',None)):
        raise ValueError('GRAGにはchoose_actionが必要です')
    return agent
