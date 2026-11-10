"""User-editable task evaluator. Coordinates use [x,y]."""
import re


def parse_goal(mission):
    text = ' '.join(mission.lower().split())
    m = re.fullmatch(r'(go to|pick up|open) (?:the |a )?(?:(red|green|blue|purple|yellow|grey) )?(ball|box|key|door)', text)
    if not m:
        raise ValueError('この命令には判定条件JSONを指定してください。task.pyで言語解釈を拡張できます。')
    verb, color, kind = m.groups()
    obj = {'type': kind}
    if color is not None:
        obj['color'] = color
    return {'type': {'go to': 'goto', 'pick up': 'pickup', 'open': 'open'}[verb], 'object': obj}


def matches(obj, spec):
    return obj is not None and all(obj.get(k) == v for k, v in spec.items())


def validate_goal(goal):
    if not isinstance(goal, dict):
        raise ValueError('判定条件はJSONオブジェクトにしてください')
    kind = goal.get('type')
    if kind in ('and', 'sequence'):
        if not isinstance(goal.get('goals'), list) or not goal['goals']:
            raise ValueError('goalsに1件以上の条件を指定してください')
        for child in goal['goals']:
            validate_goal(child)
    elif kind in ('goto', 'pickup', 'open', 'putnext'):
        for key in (('object', 'target') if kind == 'putnext' else ('object',)):
            if not isinstance(goal.get(key), dict) or not goal[key]:
                raise ValueError(f'{key}に対象物の条件を指定してください')
    else:
        raise ValueError(f'未対応の条件: {kind}')
    return goal


def evaluate_goal(goal, state, memory):
    kind = goal['type']
    if kind == 'and':
        return all([evaluate_goal(g, state, memory.setdefault(str(i), {})) for i, g in enumerate(goal['goals'])])
    if kind == 'sequence':
        index = memory.get('index', 0)
        if index < len(goal['goals']) and evaluate_goal(goal['goals'][index], state, memory.setdefault(str(index), {})):
            index += 1
            memory['index'] = index
        return index == len(goal['goals'])
    if kind == 'pickup':
        return matches(state['carrying'], goal['object'])
    candidates = [o for o in state['objects'] if matches(o, goal['object'])]
    if kind == 'goto':
        return any(o['position'] == state['front'] for o in candidates)
    if kind == 'open':
        return any(o.get('is_open', False) for o in candidates)
    if kind == 'putnext':
        targets = [o for o in state['objects'] if matches(o, goal['target'])]
        return any(abs(a['position'][0]-b['position'][0])+abs(a['position'][1]-b['position'][1]) == 1 for a in candidates for b in targets)
    raise ValueError(f'Unknown goal: {kind}')


def evaluate_task(*, mission, previous_observation, action, observation, reward, terminated, truncated, info):
    return bool(terminated and reward > 0)
