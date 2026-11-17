"""Lightweight UI/API gateway; training and checkpoint work live in a child process."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from http.server import ThreadingHTTPServer
import json
import multiprocessing
import uuid
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from gridworld.web_common import Handler


def _backend_main(connection, options):
    """Spawn entry point: import PyTorch and construct learners only in this process."""
    try:
        from gridworld import storage, training_app
        if options.get("data_dir"):
            root = Path(options["data_dir"])
            storage.DATA, storage.MAPS = root, root / "maps"
            storage.SETTINGS = root / "settings.json"
            storage.TRAINING_SETTINGS = root / "training_settings.json"
            storage.WORLD = training_app.WORLD = root / "world.json"
            storage.RESULT = training_app.RESULT = root / "latest_result.json"

        class BackendHandler(training_app.TrainingHandler):
            def log_message(self, *args):
                pass

            def handle(self):
                try:
                    super().handle()
                except ConnectionError:
                    # A progress poll can expire while a long update is running.
                    pass

        with ThreadingHTTPServer(("127.0.0.1", 0), BackendHandler) as server:
            connection.send({"port": server.server_port})
            connection.close()
            server.serve_forever()
    except Exception as error:
        try:
            connection.send({"error": str(error)})
        except (OSError, EOFError):
            pass
        raise


class TrainingBridge:
    """Serve cached progress immediately even if a learner holds the worker's GIL."""

    def __init__(self, *, options=None, target=_backend_main):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe(duplex=False)
        self.process = context.Process(target=target, args=(child, options or {}), daemon=True)
        self.lock = Lock()
        self.ready, self.closed = Event(), Event()
        self.port = None
        self.cache = {"status": "initializing", "phase": "学習プロセスを準備中"}
        self.updated = monotonic()
        self.pending_pause = None
        self.control_error = None
        self.controls = ThreadPoolExecutor(max_workers=1, thread_name_prefix="training-control")
        self.process.start()
        child.close()
        self.monitor = Thread(target=self._monitor, name="training-progress", daemon=True)
        self.monitor.start()

    def status(self):
        with self.lock:
            state = deepcopy(self.cache)
            state["snapshot_age_seconds"] = max(0, monotonic() - self.updated)
            clock = state.get("training_clock")
            if clock and clock.get("running"):
                clock["seconds"] += state["snapshot_age_seconds"]
            state["worker_pid"] = self.process.pid
            if self.pending_pause is not None:
                state.update(control_pending=True, phase="停止要求を受け付けました。更新の完了を待っています")
            if self.control_error:
                state["control_error"] = self.control_error
            return state

    def request(self, path, body=None, *, timeout=120):
        if not self.ready.wait(20) or self.port is None:
            return 503, json.dumps({"error": "学習プロセスの起動を待っています。少し待って再操作してください。"}).encode()
        request = Request(f"http://127.0.0.1:{self.port}{path}", data=body,
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    def pause(self, body):
        request = json.loads(body)
        with self.lock:
            state = self.cache
            if request.get("session_id") != state.get("session_id") or not state.get("session_id"):
                raise ValueError("停止する学習セッションが画面と一致しません。")
            if state.get("status") not in ("running", "paused"):
                raise ValueError("停止する学習がありません。")
            if self.pending_pause is None:
                self.pending_pause = state["session_id"]
                self.control_error = None
                self.controls.submit(self._send_pause, body)
        return self.status()

    def _send_pause(self, body):
        try:
            code, response = self.request("/api/train/control", body)
            if code >= 400:
                raise ValueError(json.loads(response).get("error", "停止要求に失敗しました。"))
        except Exception as error:
            with self.lock:
                self.pending_pause = None
                self.control_error = str(error)

    def accept_job(self, state):
        if not isinstance(state, dict) or "job_id" not in state:
            return
        with self.lock:
            self.cache, self.updated = state, monotonic()

    def _monitor(self):
        try:
            while not self.closed.is_set() and self.port is None:
                if self.connection.poll(0.1):
                    info = self.connection.recv()
                    if "error" in info:
                        raise RuntimeError(info["error"])
                    self.port = info["port"]
                    self.ready.set()
                elif not self.process.is_alive():
                    raise RuntimeError("学習プロセスが起動できませんでした。")
            while not self.closed.is_set():
                if not self.process.is_alive():
                    raise RuntimeError("学習プロセスが終了しました。保存済みの学習は再起動後に復元できます。")
                try:
                    code, body = self.request("/api/train/status", timeout=1)
                    if code == 200:
                        state = json.loads(body)
                        with self.lock:
                            self.cache, self.updated = state, monotonic()
                            if (self.pending_pause is not None and
                                    (state.get("status") in ("paused", "complete", "error")
                                     or state.get("session_id") != self.pending_pause)):
                                self.pending_pause = None
                except (OSError, URLError, TimeoutError):
                    # Heavy work may delay the worker. Never block the UI on it.
                    pass
                self.closed.wait(0.15)
        except (OSError, EOFError, RuntimeError) as error:
            if not self.closed.is_set():
                with self.lock:
                    self.cache = {"status": "error", "error": str(error)}
                    self.pending_pause = None
                self.ready.set()

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        self.ready.set()
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(5)
        self.connection.close()
        self.monitor.join(2)
        self.controls.shutdown(wait=True, cancel_futures=True)


def make_handler(bridge):
    class TrainingUIHandler(Handler):
        page = "training.html"

        def handle(self):
            try:
                super().handle()
            except ConnectionError:
                pass

        def log_message(self, *args):
            pass

        def relay(self, path, body=None):
            try:
                code, payload = bridge.request(path, body)
            except (OSError, URLError, TimeoutError):
                return self.json({"error": "学習プロセスが処理中です。進捗画面は引き続き操作できます。"}, 503)
            if body is not None and code < 400:
                bridge.accept_job(json.loads(payload))
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def get_api(self, path):
            if path == "/api/runtime":
                return self.json({"runtime_id": self.server.runtime_id})
            if path in ("/static/training.js", "/static/training.css"):
                name = path.rsplit("/", 1)[-1]
                kind = "application/javascript" if name.endswith(".js") else "text/css"
                return self.file(Path(__file__).parent / "training_runtime" / name, kind + "; charset=utf-8")
            if path == "/api/train/status":
                return self.json(bridge.status())
            return self.relay(path)

        def post_api(self, path, body):
            request = self.parse_json(body)
            if path == "/api/restart":
                self.server.restart_requested = True
                self.json({"restarting": True})
                Thread(target=self.server.shutdown, daemon=True).start()
                return
            if path == "/api/shutdown":
                self.json({"stopping": True})
                Thread(target=self.server.shutdown, daemon=True).start()
                return
            if path == "/api/train/control" and request.get("action") == "pause":
                return self.json(bridge.pause(body))
            return self.relay(path, body)

    return TrainingUIHandler


def main():
    while True:
        bridge = TrainingBridge()
        restart = False
        try:
            with ThreadingHTTPServer(("127.0.0.1", 5003), make_handler(bridge)) as server:
                server.runtime_id = uuid.uuid4().hex
                server.restart_requested = False
                print("http://127.0.0.1:5003 (UI and training use separate processes)", flush=True)
                server.serve_forever()
                restart = server.restart_requested
        except KeyboardInterrupt:
            pass
        finally:
            bridge.close()
        if not restart:
            break


if __name__ == "__main__":
    main()
