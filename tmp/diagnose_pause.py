"""Reproduce pause waiting stages without touching live jobs or saved experiments."""
import json
from pathlib import Path
import sys
from threading import Event, Thread
from time import monotonic, sleep
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'GridWorld')]
from gridworld import training_app as app
from gridworld.training_runtime import TrainingSession
from gridworld.storage import DEFAULT_SETTINGS

update_entered, update_release = Event(), Event()
save_entered, save_release = Event(), Event()


class BlockingAgent:
    def __init__(self, settings):
        pass

    def choose_action(self, observation, greedy=False):
        return 'up'

    def learn(self, *args):
        pass

    def end_episode(self):
        update_entered.set()
        if not update_release.wait(5):
            raise TimeoutError('diagnostic update gate')

    def export(self):
        return {}

    def checkpoint(self):
        save_entered.set()
        if not save_release.wait(5):
            raise TimeoutError('diagnostic save gate')
        return {}


maze = {'name': 'pause diagnostic', 'rows': 2, 'cols': 2,
        'start': [0, 0], 'goal': [1, 1], 'obstacles': []}
settings = {**DEFAULT_SETTINGS, 'max_steps': 1}
session = TrainingSession(maze, settings, BlockingAgent)
sid = '0' * 32
job = {'session_id': sid, 'mode': 'diagnostic', 'status': 'running',
       'execution': 'fast', 'phase': 'test', 'episodes_before': 0,
       'total': 100, '_interactive': True, '_control': 'fast', '_target': 100,
       '_after_fast': 'paused', '_wake': Event(), 'step_delay_ms': 20}
observations = []
with patch.dict(app.LEARNING_AGENTS, {'diagnostic': BlockingAgent}), \
        patch.object(app, 'learning_sessions', {sid: {'session': session}}), \
        patch.object(app, 'current_job', job), patch.object(app, 'write_json'):
    worker = Thread(target=app.run_job, args=(job, maze, settings, 'diagnostic'), daemon=True)
    worker.start()
    try:
        assert update_entered.wait(3)
        start = monotonic()
        with app.job_lock:
            job['_control'] = 'paused'
            job['_wake'].set()
        observations.append({'stage': 'pause_request_accepted_while_updating',
                             'status': job['status'], 'save_started': save_entered.is_set(),
                             'request_seconds': monotonic() - start})
        update_release.set()
        assert save_entered.wait(3)
        observations.append({'stage': 'checkpoint_in_progress', 'status': job['status'],
                             'control': job['_control']})
        save_release.set()
        deadline = monotonic() + 3
        while job['status'] != 'paused' and monotonic() < deadline:
            sleep(0.005)
        assert job['status'] == 'paused', job
        observations.append({'stage': 'checkpoint_finished', 'status': job['status']})
    finally:
        update_release.set()
        save_release.set()
        with app.job_lock:
            job['_control'] = 'close'
            job['_wake'].set()
        worker.join(3)
    assert not worker.is_alive()
print(json.dumps(observations, indent=2))
