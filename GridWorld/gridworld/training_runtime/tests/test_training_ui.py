"""Prove the UI remains responsive while the worker cannot serve HTTP requests."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
from threading import Thread, Lock
from types import SimpleNamespace
from unittest.mock import patch
from time import monotonic, sleep
import unittest
from urllib.request import Request, urlopen

from gridworld.storage import DEFAULT_SETTINGS
from gridworld.training_ui import TrainingBridge, make_handler


def busy_backend(connection, options):
    state = {"job_id": "busy-job", "session_id": "a" * 32, "status": "running",
             "execution": "fast", "mode": "test", "phase": "updating"}

    class BusyHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value):
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except ConnectionError:
                pass

        def do_GET(self):
            self.reply(state)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/api/test/busy':
                # Hold this process's GIL longer than the latency bound in the UI test.
                previous = sys.getswitchinterval()
                sys.setswitchinterval(5)
                try:
                    options['entered'].set()
                    end = monotonic() + 2
                    while monotonic() < end:
                        pass
                finally:
                    sys.setswitchinterval(previous)
                options['finished'].set()
            elif body.get('action') == 'pause':
                state['status'] = 'paused'
            self.reply(state)

    with ThreadingHTTPServer(('127.0.0.1', 0), BusyHandler) as server:
        connection.send({'port': server.server_port})
        connection.close()
        server.serve_forever()


class TrainingUITests(unittest.TestCase):
    def test_gateway_clock_advances_between_worker_snapshots_and_stops_when_idle(self):
        bridge = object.__new__(TrainingBridge)
        bridge.lock = Lock()
        bridge.cache = {'status': 'running', 'training_clock': {'seconds': 5., 'running': True}}
        bridge.updated = 10.
        bridge.process = SimpleNamespace(pid=123)
        bridge.pending_pause = bridge.control_error = None
        with patch('gridworld.training_ui.monotonic', return_value=17.):
            self.assertEqual(bridge.status()['training_clock']['seconds'], 12.)
            self.assertEqual(bridge.cache['training_clock']['seconds'], 5.)
            bridge.cache['training_clock']['running'] = False
            self.assertEqual(bridge.status()['training_clock']['seconds'], 5.)

    def start(self, bridge):
        self.bridge = bridge
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(bridge))
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'
        self.addCleanup(self.cleanup_server)
        self.assertTrue(bridge.ready.wait(20))

    def cleanup_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(3)
        self.bridge.close()

    def get(self, path):
        with urlopen(self.base + path, timeout=5) as response:
            return response.read()

    def post(self, path, body):
        request = Request(self.base + path, json.dumps(body).encode(),
                          {'Content-Type': 'application/json'})
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    def wait_status(self, predicate):
        end = monotonic() + 10
        while monotonic() < end:
            state = json.loads(self.get('/api/train/status'))
            if predicate(state):
                return state
            self.assertNotEqual(state['status'], 'error', state)
            sleep(.02)
        self.fail('Worker did not reach the requested state')

    def test_ui_and_pause_ack_remain_fast_while_worker_holds_gil(self):
        context = multiprocessing.get_context('spawn')
        entered, finished = context.Event(), context.Event()
        self.start(TrainingBridge(target=busy_backend, options={'entered': entered, 'finished': finished}))
        self.wait_status(lambda state: state['status'] == 'running')
        busy = Thread(target=lambda: self.post('/api/test/busy', {}), daemon=True)
        busy.start()
        self.assertTrue(entered.wait(5))
        latencies = []
        for path in ('/', '/static/training.js', '/api/train/status'):
            before = monotonic()
            self.get(path)
            latencies.append(monotonic() - before)
        before = monotonic()
        acknowledgement = self.post('/api/train/control', {'session_id': 'a' * 32, 'action': 'pause'})
        latencies.append(monotonic() - before)
        self.assertTrue(acknowledgement['control_pending'])
        self.assertFalse(finished.is_set(), 'The UI must respond before the worker finishes')
        self.assertLess(max(latencies), .75, latencies)
        self.assertNotEqual(acknowledgement['worker_pid'], os.getpid())
        self.wait_status(lambda state: state['status'] == 'paused')
        busy.join(5)
        self.assertFalse(busy.is_alive())

    def test_real_worker_trains_pauses_and_saves(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        folder = directory.name
        root = Path(folder)
        (root / 'maps').mkdir()
        maze = {'name': 'gateway test', 'rows': 2, 'cols': 3, 'start': [0, 0],
                'goal': [1, 2], 'obstacles': [[0, 1]]}
        key = hashlib.sha256(maze['name'].encode()).hexdigest()
        (root / 'maps' / (key + '.json')).write_text(json.dumps(maze), encoding='utf-8')
        (root / 'training_settings.json').write_text(json.dumps(DEFAULT_SETTINGS), encoding='utf-8')
        self.start(TrainingBridge(options={'data_dir': folder}))
        job = self.post('/api/train', {'map_name': maze['name'], 'execution': 'fast',
                        'episodes': 100000, 'training_settings': {'max_steps': 16}})
        self.wait_status(lambda state: state.get('preview', {}).get('total_steps', 0) > 0)
        result = self.post('/api/train/control', {'session_id': job['session_id'], 'action': 'pause'})
        self.assertTrue(result['control_pending'])
        paused = self.wait_status(lambda state: state['status'] == 'paused')
        self.assertLess(paused['preview']['completed_episodes'], 100000)
        self.assertTrue((root / 'training_sessions' / (job['session_id'] + '.json')).exists())

    def test_real_cppos_binary_checkpoint_through_gateway(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / 'maps').mkdir()
        maze = {'name': 'cppo gateway', 'rows': 2, 'cols': 3, 'start': [0, 0],
                'goal': [1, 2], 'obstacles': [[0, 1]]}
        key = hashlib.sha256(maze['name'].encode()).hexdigest()
        (root / 'maps' / (key + '.json')).write_text(json.dumps(maze), encoding='utf-8')
        (root / 'training_settings.json').write_text(json.dumps(DEFAULT_SETTINGS), encoding='utf-8')
        self.start(TrainingBridge(options={'data_dir': str(root)}))
        job = self.post('/api/train', {'map_name': maze['name'], 'mode': 'cppo', 'execution': 'fast',
                        'episodes': 4, 'training_settings': {'max_steps': 8, 'cppo_hidden_size': 16,
                        'cppo_depth': 2, 'cppo_repr_dim': 8, 'cppo_batch_size': 8, 'cppo_rollout_steps': 12}})
        self.wait_status(lambda state: state['status'] == 'paused')
        saved = json.loads((root / 'training_sessions' / (job['session_id'] + '.json')).read_text())
        self.assertEqual(saved['agent_checkpoint']['model']['schema_version'], 3)
        self.assertGreater(saved['agent_checkpoint']['model']['updates'], 0)
        self.assertIsNone(saved['q_table'])
        latest = json.loads((root / 'latest_result.json').read_text())
        self.assertIsNone(latest['agent_checkpoint'])


if __name__ == '__main__':
    unittest.main()
