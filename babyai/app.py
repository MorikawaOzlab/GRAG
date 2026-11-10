"""Local BabyAI training application: python -m babyai.app."""
import json
import cloudpickle as pickle
import threading
import uuid
import os
from bisect import bisect_left
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs
from .environment import BabyAIEnvironment
from .ollama_agent import OllamaAgent

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
LOCK = threading.RLock()
STOP = threading.Event()
SESSION = None
WORKER = None
ERROR = ''
OBSERVATION_DELAY = .16


class Session:
    def __init__(self, settings):
        self.settings = settings
        self.id = uuid.uuid4().hex
        self.world = BabyAIEnvironment(env_id=settings.get('env_id','BabyAI-GoToRedBall-v0'), seed=int(settings.get('seed',0)), mission=settings.get('mission',''), goal=settings.get('goal'), max_steps=int(settings.get('max_steps',256)))
        self.obs = self.world.reset()
        mode = settings.get('agent','ppo')
        if mode == 'ppo':
            from .ppo_agent import PPOAgent
            self.agent = PPOAgent(self.obs, int(settings.get('seed',0)))
        elif mode == 'ollama':
            self.agent = OllamaAgent(model=settings.get('model','gpt-oss:20b'), base_url=settings.get('ollama_url','http://127.0.0.1:11434'))
        elif mode == 'grag':
            from .grag_adapter import create_grag
            self.agent = create_grag(settings.get('grag_entry',''), settings)
        else:
            raise ValueError('Unknown agent')
        self.history, self.steps, self.reward, self.total_steps = [], 0, 0., 0
        self.action, self.ended = None, False
        self.published = self.snapshot()

    def step(self):
        if self.ended:
            self.obs = self.world.reset()
            self.steps, self.reward, self.ended = 0, 0., False
        previous = self.obs
        if isinstance(self.agent, OllamaAgent):
            def publish_trace(trace):
                metadata = {'episode':len(self.history)+1, 'step':self.steps+1,
                            'total_steps':self.total_steps+1}
                trace.update(metadata)
                self.agent.trace.update(metadata)
                self.published = {**self.published, 'llm':trace}
            try:
                action = self.agent.choose_action(previous, on_progress=publish_trace)
            except Exception:
                self.agent.trace.update(episode=len(self.history)+1, step=self.steps+1,
                                        total_steps=self.total_steps+1)
                self.llm_history = getattr(self, 'llm_history', []) + [dict(self.agent.trace)]
                self.published = self.snapshot()
                raise
        else:
            action = self.agent.choose_action(previous)
        obs, reward, terminated, truncated, success = self.world.step(action)
        if callable(getattr(self.agent,'learn',None)):
            self.agent.learn(previous, action, reward, obs, terminated, truncated)
        self.obs, self.action = obs, action
        self.steps += 1
        self.total_steps += 1
        self.reward += reward
        if isinstance(self.agent, OllamaAgent):
            self.agent.trace.update(status='executed', reward=reward, step=self.steps,
                                    episode=len(self.history)+1, total_steps=self.total_steps,
                                    result=obs.get('last_transition'))
            self.llm_history = getattr(self, 'llm_history', []) + [dict(self.agent.trace)]
        self.ended = terminated or truncated
        if self.ended:
            if callable(getattr(self.agent,'end_episode',None)):
                self.agent.end_episode()
            self.history.append({'episode':len(self.history)+1,'reward':self.reward,'steps':self.steps,'success':bool(success),'total_steps':self.total_steps})

        self.published = self.snapshot()

    def normalize_llm_trace(self, trace):
        """Read older cumulative-step checkpoints without changing their files."""
        if not trace or 'episode' in trace or 'step' not in trace:
            return trace
        total = trace.get('total_steps', trace['step'])
        ends = [row['total_steps'] for row in self.history]
        index = bisect_left(ends, total)
        previous = ends[index-1] if index else 0
        return {**trace, 'episode':index+1, 'step':total-previous, 'total_steps':total}

    def snapshot(self):
        return {'id':self.id,'settings':self.settings,'observation':self.obs,'world':self.world.state(), 'action':self.action,'steps':self.steps,'total_steps':self.total_steps,'history':list(self.history[-2000:]), 'episodes':len(self.history),'ended':self.ended,
                'execution':getattr(self, 'execution', {'mode':'serial','workers':1}),
                'llm':self.normalize_llm_trace(getattr(self.agent, 'trace', None)),
                'llm_history':[self.normalize_llm_trace(t) for t in getattr(self, 'llm_history', [])[-20:]],
                'llm_log_count':len(getattr(self, 'llm_history', []))}


def save_session():
    if SESSION is None:
        return
    DATA.mkdir(exist_ok=True)
    path = DATA / (SESSION.id+'.pickle')
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(pickle.dumps(SESSION))
    temporary.replace(path)


def run(episodes, observe):
    global ERROR
    try:
        if SESSION.settings.get('agent', 'ppo') == 'ppo':
            import torch
            torch.set_num_threads(max(1, (os.cpu_count() or 1)-2))
        if not observe and SESSION.settings.get('agent', 'ppo') == 'ppo':
            from .parallel_training import worker_count, train_parallel
            if worker_count(SESSION.settings.get('cpu_workers', 0), episodes) > 1:
                train_parallel(SESSION, episodes, STOP)
                with LOCK:
                    save_session()
                return
        SESSION.execution = {'mode':'serial', 'workers':1}
        target = len(SESSION.history)+episodes
        while not STOP.is_set() and len(SESSION.history)<target:
            SESSION.step()
            if observe:
                STOP.wait(OBSERVATION_DELAY)
        with LOCK:
            save_session()
    except Exception as exc:
        ERROR = str(exc)


class Handler(BaseHTTPRequestHandler):
    def reply(self, value, code=200):
        body = json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlsplit(self.path).path == '/api/llm-logs':
            try:
                after = int(parse_qs(urlsplit(self.path).query).get('after', ['0'])[0])
                if after < 0:
                    raise ValueError('Invalid log offset')
                history = getattr(SESSION, 'llm_history', []) if SESSION else []
                normalize = getattr(SESSION, 'normalize_llm_trace', lambda trace:trace)
                self.reply({'id':SESSION.id if SESSION else None,
                            'logs':[normalize(trace) for trace in history[after:after+100]]})
            except ValueError as exc:
                self.reply({'error':str(exc)}, 400)
            return
        if self.path == '/':
            body = (ROOT/'index.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ('/static/training.css', '/static/training.js', '/static/log_follow.js'):
            name = self.path.rsplit('/', 1)[1]
            body = (ROOT / name).read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', ('text/css' if name.endswith('.css') else 'text/javascript') + '; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == '/api/state':
            with LOCK:
                self.reply({'session':SESSION.published if SESSION else None,'running':bool(WORKER and WORKER.is_alive()),'error':ERROR})
        elif self.path == '/api/saves':
            self.reply({'saves':[p.stem for p in DATA.glob('*.pickle')]})
        elif self.path == '/api/cpu':
            self.reply({'logical_cpus':os.cpu_count() or 1, 'default_workers':min(4,max(1,(os.cpu_count() or 1)-2))})
        else:
            self.reply({'error':'Not found'},404)

    def do_POST(self):
        global SESSION, WORKER, ERROR, OBSERVATION_DELAY
        try:
            size = int(self.headers.get('Content-Length','0'))
            if not 0 < size <= 100000:
                raise ValueError('Invalid request size')
            body = json.loads(self.rfile.read(size))
            if self.path == '/api/speed' or (self.path == '/api/run' and body.get('observe', False)):
                speed_ms = body.get('speed_ms', 160)
                if type(speed_ms) is not int or speed_ms not in (30, 80, 160, 320):
                    raise ValueError('speed_ms must be 30, 80, 160 or 320')
                OBSERVATION_DELAY = speed_ms / 1000
                if self.path == '/api/speed':
                    self.reply({'ok': True, 'speed_ms': speed_ms})
                    return
            if self.path == '/api/pause':
                STOP.set()
                self.reply({'ok':True})
                return
            if WORKER and WORKER.is_alive():
                raise ValueError('一時停止後に操作してください')
            with LOCK:
                if self.path == '/api/new':
                    from .parallel_training import worker_count
                    worker_count(body.get('cpu_workers', 0), 1)
                    steps = int(body.get('max_steps',256))
                    if not 1 <= steps <= 100000:
                        raise ValueError('最大ステップは1〜100000です')
                    candidate = Session(body)
                    if SESSION:
                        SESSION.world.close()
                    SESSION = candidate
                    save_session()
                elif self.path == '/api/load':
                    sid = body['id']
                    if len(sid)!=32 or any(c not in '0123456789abcdef' for c in sid):
                        raise ValueError('Invalid session id')
                    # Only checkpoints produced locally by this application.
                    candidate = pickle.loads((DATA/(sid+'.pickle')).read_bytes())
                    if SESSION:
                        SESSION.world.close()
                    SESSION = candidate
                elif self.path == '/api/save':
                    save_session()
                elif self.path in ('/api/run','/api/step'):
                    if SESSION is None:
                        raise ValueError('新しい実験を作成してください')
                    episodes = int(body.get('episodes',1))
                    if not 1 <= episodes <= 1000000:
                        raise ValueError('試行数は1〜1000000です')
                    STOP.clear()
                    ERROR = ''
                    if self.path == '/api/step':
                        def one():
                            global ERROR
                            try:
                                SESSION.step()
                                with LOCK:
                                    save_session()
                            except Exception as exc:
                                ERROR = str(exc)
                        WORKER = threading.Thread(target=one, daemon=True)
                    else:
                        WORKER = threading.Thread(target=run,args=(episodes,bool(body.get('observe',False))),daemon=True)
                    WORKER.start()
                else:
                    raise ValueError('Unknown API')
            self.reply({'ok':True})
        except Exception as exc:
            self.reply({'error':str(exc)},400)


if __name__ == '__main__':
    print('BabyAI: http://127.0.0.1:5010')
    ThreadingHTTPServer(('127.0.0.1',5010),Handler).serve_forever()
