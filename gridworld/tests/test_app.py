"""Focused integration checks for the file APIs and training invariants."""

import json
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from gridworld import storage
from gridworld import parallel_training
from gridworld import settings_app
from gridworld import training_app
from gridworld.map_app import MapHandler
from gridworld.settings_app import SettingsHandler
from gridworld.training_app import TrainingHandler
from gridworld.training_app import display_result
from gridworld.training import train
from gridworld.training import add_moving_average
from gridworld.parallel_training import train_parallel
from agents.q_learning import QLearningAgent
from gridworld.world import GridWorld


MAP = {"name": "検証", "rows": 2, "cols": 3, "start": [0, 0],
       "goal": [1, 2], "obstacles": [[0, 1]]}


class PlannedAgent:
    """Deterministic route: one collision, then three actual entries."""

    def __init__(self, settings):
        self.actions = iter(("right", "down", "right", "right"))

    def choose_action(self, observation, greedy=False):
        return next(self.actions, "up")

    def learn(self, observation, action, reward, next_observation, reached_goal):
        pass

    def end_episode(self):
        pass

    def export(self):
        return {}


class ReferenceQLearningAgent(QLearningAgent):
    """Uses the generic environment path for exact-result comparison."""


class AppTests(unittest.TestCase):
    def test_hundred_episode_moving_average(self):
        history = [{"reward": 1 if episode in (1, 101) else 0} for episode in range(1, 102)]
        add_moving_average(history)
        self.assertEqual(history[0]["moving_average_reward"], 1)
        self.assertAlmostEqual(history[99]["moving_average_reward"], 0.01)
        self.assertAlmostEqual(history[100]["moving_average_reward"], 0.01)

    def test_parallel_training_counts_all_episodes(self):
        settings = {**storage.DEFAULT_SETTINGS, "episodes": 20, "max_steps": 15}
        with patch.object(parallel_training.os, "cpu_count", return_value=4):
            result = train_parallel(MAP, settings)
        self.assertEqual(result["workers_used"], 2)
        self.assertEqual(len(result["history"]), 20)
        self.assertEqual(result["total_reward"], result["successes"])
        self.assertEqual(result["history"][-1]["cumulative_reward"], result["total_reward"])

    def test_worker_count_reserves_two_logical_cpus(self):
        with patch.object(parallel_training.os, "cpu_count", return_value=20):
            self.assertEqual(parallel_training.worker_count(1000), 18)
            self.assertEqual(parallel_training.worker_count(5), 5)
        with patch.object(parallel_training.os, "cpu_count", return_value=2):
            self.assertEqual(parallel_training.worker_count(1000), 1)

    def test_display_result_is_compact(self):
        history = [{"episode": i, "cumulative_reward": i // 10} for i in range(1, 1001)]
        shown = display_result({"history": history, "q_table": {"0,0": [1, 2, 3, 4]}})
        self.assertLessEqual(len(shown["history"]), 701)
        self.assertEqual(shown["history"][-1], history[-1])
        self.assertIsNone(shown["q_table"])

    def test_fast_path_preserves_q_learning_results(self):
        settings = {**storage.DEFAULT_SETTINGS, "episodes": 50, "max_steps": 25}
        fast = train(MAP, settings)
        reference = train(MAP, settings, ReferenceQLearningAgent)
        self.assertEqual(fast, reference)

    def test_world_and_visit_count(self):
        world = GridWorld(MAP)
        observation, done, collision = world.step("right")
        self.assertEqual(observation["position"], [0, 0])
        self.assertFalse(done)
        self.assertTrue(collision)
        settings = {**storage.DEFAULT_SETTINGS, "episodes": 1, "max_steps": 4}
        result = train(MAP, settings, PlannedAgent)
        self.assertEqual(result["visits"], [[0, 0, 0], [1, 1, 1]])
        self.assertEqual(result["successes"], 1)
        self.assertEqual(result["total_reward"], 1)
        self.assertEqual(result["history"][0]["reward"], 1)
        self.assertEqual(result["history"][0]["cumulative_reward"], 1)
        self.assertEqual(result["frames"][0]["position"], [0, 0])

    def test_unreachable_map_is_allowed(self):
        blocked = {"name": "到達不能", "rows": 2, "cols": 2, "start": [0, 0],
                   "goal": [1, 1], "obstacles": [[0, 1], [1, 0]]}
        settings = {**storage.DEFAULT_SETTINGS, "episodes": 3, "max_steps": 5}
        result = train(blocked, settings)
        self.assertEqual(result["successes"], 0)
        self.assertEqual(result["total_reward"], 0)
        self.assertTrue(all(item["reward"] == 0 and item["cumulative_reward"] == 0
                            for item in result["history"]))
        self.assertEqual(result["visits"], [[0, 0], [0, 0]])
        self.assertEqual(len(result["frames"]), 6)

    def test_http_file_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with patch.object(storage, "MAPS", base / "maps"), \
                 patch.object(storage, "SETTINGS", base / "settings.json"), \
                 patch.object(training_app, "WORLD", base / "world.json"), \
                 patch.object(training_app, "RESULT", base / "result.json"), \
                 patch.object(parallel_training.os, "cpu_count", return_value=3):
                servers = [ThreadingHTTPServer(("127.0.0.1", 0), handler)
                           for handler in (MapHandler, SettingsHandler, TrainingHandler)]
                threads = [threading.Thread(target=server.serve_forever, daemon=True)
                           for server in servers]
                for thread in threads:
                    thread.start()
                try:
                    urls = [f"http://127.0.0.1:{server.server_port}" for server in servers]
                    self.assertIn("マップ作成", urlopen(urls[0]).read().decode("utf-8"))
                    settings_page = urlopen(urls[1]).read().decode("utf-8")
                    training_page = urlopen(urls[2]).read().decode("utf-8")
                    self.assertIn("環境設定", settings_page)
                    self.assertNotIn('id="reward_goal"', settings_page)
                    self.assertIn("AI学習", training_page)
                    self.assertIn('id="learning-curve"', training_page)
                    def post(url, data):
                        request = Request(url, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                                          {"Content-Type": "application/json"}, method="POST")
                        return json.load(urlopen(request))
                    post(urls[0] + "/api/maps", MAP)
                    self.assertEqual(json.load(urlopen(urls[2] + "/api/maps"))[0]["name"], "検証")
                    loaded = json.load(urlopen(urls[0] + "/api/maps/%E6%A4%9C%E8%A8%BC"))
                    self.assertEqual(loaded, MAP)
                    self.assertEqual(json.load(urlopen(urls[0] + "/api/maps/%E6%A4%9C%E8%A8%BC/download")), MAP)
                    uploaded = {**MAP, "name": "アップロード"}
                    payload = json.dumps(uploaded, ensure_ascii=False).encode("utf-8")
                    body = (b"--test-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"map.json\"\r\n"
                            b"Content-Type: application/json\r\n\r\n" + payload + b"\r\n--test-boundary--\r\n")
                    upload_request = Request(urls[0] + "/api/maps/upload", body,
                                             {"Content-Type": "multipart/form-data; boundary=test-boundary"}, method="POST")
                    self.assertEqual(json.load(urlopen(upload_request)), uploaded)
                    settings = {**storage.DEFAULT_SETTINGS, "episodes": 5, "max_steps": 15,
                                "reward_step": -1, "reward_collision": -3,
                                "reward_goal": 100, "workers": 99}
                    environment = json.load(urlopen(urls[1] + "/api/settings"))
                    self.assertNotIn("episodes", environment)
                    self.assertNotIn("max_steps", environment)
                    with self.assertRaises(HTTPError) as error:
                        post(urls[1] + "/api/settings", settings)
                    self.assertEqual(error.exception.code, 400)
                    job = post(urls[2] + "/api/train", {"map_name": "検証", "episodes": 5,
                               "training_settings": {"max_steps": 15}})
                    self.assertEqual(job["status"], "running")
                    deadline = time.monotonic() + 5
                    while True:
                        status = json.load(urlopen(urls[2] + "/api/train/status"))
                        if status["status"] == "complete":
                            break
                        self.assertNotEqual(status["status"], "error")
                        self.assertLess(time.monotonic(), deadline)
                        time.sleep(0.01)
                    self.assertEqual(status["completed"], 5)
                    result = json.load(urlopen(urls[2] + "/api/result"))
                    self.assertEqual(len(result["history"]), 5)
                    self.assertIsNone(result["q_table"])
                    self.assertIsInstance(storage.read_json(base / "result.json")["q_table"], dict)
                    self.assertEqual(result["schema_version"], 3)
                    self.assertEqual(result["workers_used"], 1)
                    self.assertEqual(result["total_reward"], result["successes"])
                    self.assertEqual(result["history"][-1]["cumulative_reward"], result["total_reward"])
                    self.assertTrue(all(item["reward"] in (0, 1) for item in result["history"]))
                    self.assertEqual(storage.read_json(base / "world.json"), MAP)
                    self.assertEqual(json.load(urlopen(urls[2] + "/api/result"))["map"], MAP)
                finally:
                    for server in servers:
                        server.shutdown()
                        server.server_close()
                    for thread in threads:
                        thread.join()


if __name__ == "__main__":
    unittest.main()
