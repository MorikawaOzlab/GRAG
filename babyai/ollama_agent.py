"""Per-step Ollama baseline for BabyAI observations."""
import json
import time
import uuid
import re
from urllib import request

ACTIONS = ("left", "right", "forward", "pickup", "drop", "toggle", "done")
OBJECTS = ('unseen', 'empty', 'wall', 'floor', 'door', 'key', 'ball', 'box', 'goal', 'lava', 'agent')
COLORS = ('red', 'green', 'blue', 'purple', 'yellow', 'grey')
DIRECTIONS = ('east', 'south', 'west', 'north')
FORWARD_VECTORS = ((1,0), (0,1), (-1,0), (0,-1))


def describe_view(observation):
    """Convert x-major MiniGrid data into named rows and egocentric offsets.

    Absolute offsets are displacements from the agent, not global positions.
    Everything here is derived from the partial observation and compass alone.
    """
    image = observation.get('image')
    if not image or not image[0]:
        return None
    width, height = len(image), len(image[0])
    direction = int(observation['direction'])
    fx, fy = FORWARD_VECTORS[direction]
    rx, ry = -fy, fx
    rows, objects = [], []
    for y in range(height):
        row = []
        for x in range(width):
            obj, color, state = image[x][y]
            kind = OBJECTS[obj] if 0 <= obj < len(OBJECTS) else 'unknown'
            if (x,y) == (width//2,height-1):
                row.append('AGENT (carried object is in carrying, not on the floor)')
                continue
            name = COLORS[color] if 0 <= color < len(COLORS) else 'unknown'
            token = kind
            if kind in ('ball','key','box','door'):
                token = name+' '+kind
            if kind == 'door':
                token += ' ('+{0:'open',1:'closed',2:'locked'}.get(state,'unknown')+')'
            row.append(token)
            if kind in ('ball','key','box','door','goal','lava'):
                forward, right = height-1-y, x-width//2
                objects.append({'object':kind, 'color':name, 'view_position':[x,y],
                    'relative_offset':{'forward':forward,'right':right},
                    'absolute_offset':{'east':forward*fx+right*rx, 'south':forward*fy+right*ry},
                    'door_state':{0:'open',1:'closed',2:'locked'}.get(state) if kind=='door' else None})
        rows.append(row)
    return {'orientation':{'forward':DIRECTIONS[direction], 'right':DIRECTIONS[(direction+1)%4],
            'backward':DIRECTIONS[(direction+2)%4], 'left':DIRECTIONS[(direction-1)%4],
            'after_left_turn':DIRECTIONS[(direction-1)%4], 'after_right_turn':DIRECTIONS[(direction+1)%4]},
            'rows_top_to_bottom':rows, 'visible_objects':objects,
            'agent_view_position':[width//2,height-1]}


def describe_front(observation):
    """Decode only the supplied egocentric image, never the full world map."""
    image = observation.get('image')
    if not image or not image[0] or len(image[0]) < 2:
        return None
    x, y = len(image)//2, len(image[0])-2
    obj, color, state = image[x][y]
    kind = OBJECTS[obj] if 0 <= obj < len(OBJECTS) else 'unknown'
    passable = (kind in ('empty', 'floor', 'goal', 'lava') or (kind == 'door' and state == 0))
    return {'view_position':[x,y], 'object':kind,
            'color':COLORS[color] if 0 <= color < len(COLORS) else None,
            'door_state':{0:'open',1:'closed',2:'locked'}.get(state) if kind == 'door' else None,
            'forward_passable':None if kind in ('unseen','unknown') else passable}


def parse_action(content):
    """Read explicit action labels, allowing Markdown without guessing from prose."""
    def unformat(text):
        return text.replace('*', '').replace('`', '').strip().strip('_').strip()

    labeled = []
    for line in content.splitlines():
        match = re.fullmatch(r'action\s*[:：]\s*(.*?)\s*', unformat(line), re.IGNORECASE)
        if match:
            labeled.append(unformat(match.group(1)).lower())
    if labeled:
        if all(action in ACTIONS for action in labeled) and len(set(labeled)) == 1:
            return labeled[0]
    else:
        plain = unformat(content).lower()
        if plain in ACTIONS:
            return plain
    raise ValueError(f"Ollama returned an invalid or ambiguous BabyAI action: {content!r}")


class OllamaAgent:
    def __init__(self, model="gpt-oss:20b", base_url="http://127.0.0.1:11434", timeout=120):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout

    def learn(self, observation, action, reward, next_observation, terminated, truncated):
        # Context memory, not parameter training. Only observed outcomes are used.
        transition = {'action':action, 'result':next_observation.get('last_transition'),
                      'front_after':describe_front(next_observation),
                      'facing_after':DIRECTIONS[int(next_observation['direction'])] if 'direction' in next_observation else None,
                      'reward':reward}
        self.recent_transitions = (getattr(self, 'recent_transitions', []) + [transition])[-8:]

    def end_episode(self):
        self.recent_transitions = []

    def choose_action(self, observation, on_progress=None):
        started = time.monotonic()
        self.trace = {'id':uuid.uuid4().hex, 'status':'requesting', 'thinking':'', 'content':'', 'action':None, 'elapsed':0.}
        def publish():
            self.trace['elapsed'] = round(time.monotonic()-started, 2)
            if on_progress:
                on_progress(dict(self.trace))
        publish()
        prompt = ("You control a BabyAI agent. Your partial view is decoded into named cells in rows_top_to_bottom (rows first, columns left to right). "
                  "The AGENT cell is bottom center. UP IN THIS VIEW IS ALWAYS YOUR CURRENT FORWARD, NOT ALWAYS NORTH. "
                  "Use orientation.forward/right/left to map relative view directions to compass directions. "
                  "relative_offset.forward is the number of cells ahead; relative_offset.right is positive to your right and negative to your left. "
                  "absolute_offset is displacement from you, not a global coordinate: east positive means east, east negative means west; south positive means south, south negative means north. "
                  "For example, when facing west, a cell ahead is west and a cell on your right is north. "
                  "After turning, the egocentric view rotates but the objects in the world have not moved. "
                  "Do not reuse old relative offsets without accounting for turns. Use the current orientation and current view. "
                  "front_cell is exactly one cell straight ahead, not the AGENT cell. "
                  "Walls are impassable: forward into a wall does not move you. Closed or locked doors and objects (ball, box, key) also block forward movement. "
                  "Turn left or right (90 degrees in place) to face another route. A turn does not move you sideways. "
                  "Open a closed door with toggle; a locked door requires carrying the matching-color key. "
                  "A picked-up object is removed from its cell, but pickup requires empty hands. Lava is dangerous. "
                  "If last_transition.blocked_forward is true, the last forward failed. Do not repeat forward without a relevant change in the front cell; turn to find a route or operate the blocking object. "
                  "Use the decoded front_cell and recent_transitions to check progress. Never assume unseen cells are free. Choose exactly one action from "
                  + ", ".join(ACTIONS) + ". "
                  + "In your final response, write a brief Japanese explanation, then on the last line "
                  + "write ACTION: followed by exactly one of the action names above. "
                  + "Do not translate the action identifier. Do not format the ACTION line with Markdown.\n"
                  + json.dumps({'observation':{k:v for k,v in observation.items() if k != 'image'},
                                'view':describe_view(observation), 'front_cell':describe_front(observation),
                                'recent_transitions':getattr(self, 'recent_transitions', [])}, ensure_ascii=False))
        body = json.dumps({"model": self.model, "stream": True,
                           "messages": [{"role":"system", "content":"Write your final explanation in Japanese. Keep action identifiers and ACTION: in English."},
                                        {"role": "user", "content": prompt}]}).encode("utf-8")
        req = request.Request(self.base_url + "/api/chat", data=body,
                              headers={"Content-Type": "application/json"})
        try:
            done = False
            with request.urlopen(req, timeout=self.timeout) as response:
                for line in response:
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get('error'):
                        raise ValueError(chunk['error'])
                    message = chunk.get('message', {})
                    for key in ('thinking', 'content'):
                        self.trace[key] += message.get(key, '')
                    self.trace['status'] = 'generating' if self.trace['content'] else 'thinking'
                    publish()
                    if chunk.get('done'):
                        done = True
                        break
            if not done:
                raise ValueError('Ollama stream ended before completion')
            action = parse_action(self.trace['content'])
            self.trace.update(status='generated', action=action)
            publish()
        except Exception as exc:
            self.trace.update(status='error', error=str(exc))
            publish()
            raise
        return action
