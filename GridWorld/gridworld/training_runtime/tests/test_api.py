from contextlib import ExitStack
from http.server import ThreadingHTTPServer
import json
import importlib.util
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from gridworld import storage
from gridworld import training_app
from gridworld.training_runtime import TrainingSession
from .test_session import CountingAgent, MAP, SETTINGS


class QuietHandler(training_app.TrainingHandler):
    def log_message(self, *args):
        pass


class SessionApiTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.base_path = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for owner, key, value in (
                (storage, "MAPS", self.base_path / "maps"),
                (training_app, "WORLD", self.base_path / "world.json"),
                (training_app, "RESULT", self.base_path / "latest_result.json"),
                (training_app, "current_job", None),
                (training_app, "learning_sessions", {})):
            self.stack.enter_context(patch.object(owner, key, value))
        self.stack.enter_context(patch.object(training_app, "load_settings", return_value=SETTINGS.copy()))
        storage.save_map(MAP)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.stack.close()

    def get(self, path):
        return json.load(urlopen(self.base + path, timeout=5))

    def post(self, path, value):
        request = Request(self.base + path, json.dumps(value).encode("utf-8"),
                          {"Content-Type": "application/json"}, method="POST")
        return json.load(urlopen(request, timeout=5))

    def wait(self, job):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = self.get("/api/train/status")
            if status["status"] == "complete":
                self.assertEqual(status["job_id"], job["job_id"])
                return self.get("/api/result")
            self.assertNotEqual(status["status"], "error", status.get("error"))
            time.sleep(.01)
        self.fail("Training did not complete")

    def test_grag_success_threshold_is_kept_from_training_settings(self):
        result = self.wait(self.post("/api/train", {
            "map_name": MAP["name"], "mode": "grag", "episodes": 1,
            "training_settings": {"max_steps": 1, "grag_positive_update_count": 8},
        }))
        self.assertEqual(result["settings"]["grag_positive_update_count"], 8)
        self.assertEqual(result["successes"], 0)

    def test_grag_hindsight_gamma_reaches_agent_and_is_persisted(self):
        result = self.wait(self.post("/api/train", {
            "map_name": MAP["name"], "mode": "grag", "episodes": 1,
            "training_settings": {"max_steps": 1, "hindsight_gamma": 0.9,
                                  "grag_update_interval": 256, "contrastive_learning_rate": 0.0007},
        }))
        session = training_app.learning_sessions[result["session_id"]]["session"]
        self.assertEqual(session.agent.hindsight_gamma, 0.9)
        self.assertEqual(session.agent.UPDATE_INTERVAL, 256)
        self.assertEqual(session.agent.contrastive_optimizer.param_groups[0]["lr"], 0.0007)
        self.assertEqual(result["settings"]["hindsight_gamma"], 0.9)
        with patch.object(storage, "SETTINGS", self.base_path / "settings.json"):
            self.assertEqual(storage.load_settings()["hindsight_gamma"], 0.9)
            self.assertEqual(storage.load_settings()["grag_update_interval"], 256)
            self.assertEqual(storage.load_settings()["contrastive_learning_rate"], 0.0007)

    def test_grag_rejects_invalid_hindsight_gamma(self):
        with self.assertRaises(HTTPError) as error:
            self.post("/api/train", {"map_name": MAP["name"], "mode": "grag", "episodes": 1,
                                     "training_settings": {"max_steps": 1, "hindsight_gamma": 1}})
        self.assertEqual(error.exception.code, 400)

    def test_grag_rejects_invalid_update_interval(self):
        with self.assertRaises(HTTPError) as error:
            self.post("/api/train", {"map_name": MAP["name"], "mode": "grag", "episodes": 1,
                                     "training_settings": {"max_steps": 1, "grag_update_interval": 0}})
        self.assertEqual(error.exception.code, 400)

    def test_start_continue_evaluate_and_restart_keep_cumulative_learning(self):
        first = self.wait(self.post("/api/train", {"map_name": MAP["name"], "episodes": 100}))
        self.assertEqual(first["settings"]["episodes"], 100)
        self.assertIsNone(first["q_table"])
        self.assertIsNone(first["agent_checkpoint"])
        sid = first["session_id"]
        second = self.wait(self.post("/api/train", {"map_name": MAP["name"], "episodes": 25,
                                                    "session_id": sid}))
        self.assertEqual(second["session_id"], sid)
        self.assertEqual(len(second["history"]), 125)
        self.assertEqual(second["history"][:100], first["history"])
        before = storage.read_json(training_app.saved_session_path(sid))
        evaluated = self.wait(self.post("/api/train/evaluate", {"map_name": MAP["name"],
                            "episodes": 0, "session_id": sid}))
        after = storage.read_json(training_app.saved_session_path(sid))
        self.assertEqual(evaluated["settings"]["episodes"], 125)
        self.assertEqual(after["history"], before["history"])
        self.assertEqual(after["agent_checkpoint"], before["agent_checkpoint"])
        training_app.learning_sessions.clear()
        final = self.wait(self.post("/api/train", {"map_name": MAP["name"], "episodes": 5,
                                                   "session_id": sid}))
        self.assertEqual(final["settings"]["episodes"], 130)
        continuous = TrainingSession(MAP, SETTINGS)
        continuous.run_batch(130)
        saved = storage.read_json(training_app.saved_session_path(sid))
        self.assertEqual(saved["history"], continuous.history)
        self.assertEqual(saved["q_table"], continuous.agent.export())

    def test_arbitrary_registered_agent_uses_the_common_api(self):
        with patch.dict(training_app.LEARNING_AGENTS, {"counting": CountingAgent}):
            self.assertIn("counting", [item["key"] for item in self.get("/api/agents")])
            first = self.wait(self.post("/api/train", {"map_name": MAP["name"], "mode": "counting", "episodes": 3}))
            final = self.wait(self.post("/api/train", {"map_name": MAP["name"], "mode": "counting",
                             "episodes": 2, "session_id": first["session_id"]}))
            self.assertEqual(final["settings"]["episodes"], 5)
            agent = training_app.learning_sessions[first["session_id"]]["session"].agent
            self.assertEqual(agent.episodes, 5)
            self.assertEqual(agent.updates, 15)

    def test_benchmark_starts_fresh_and_stops_at_first_100_successes(self):
        with patch.dict(training_app.LEARNING_AGENTS, {"benchmark_test": CountingAgent}):
            old = self.wait(self.post("/api/train", {"map_name": MAP["name"],
                            "mode": "benchmark_test", "episodes": 3}))
            final = self.wait(self.post("/api/train", {"map_name": MAP["name"],
                              "mode": "benchmark_test", "execution": "benchmark", "episodes": 120}))
            self.assertNotEqual(old["session_id"], final["session_id"])
            self.assertEqual(final["completed_episodes"], 100)
            self.assertEqual(final["benchmark"]["status"], "reached")
            self.assertEqual(final["benchmark"]["start_episode"], 0)
            self.assertEqual(final["success_100_milestone"]["episode"], 100)
            self.assertEqual(final["training_seconds"], final["success_100_milestone"]["seconds"])
            self.assertEqual(self.get("/api/train/status")["benchmark_outcome"], "reached")
            with self.assertRaises(HTTPError) as error:
                self.post("/api/train", {"map_name": MAP["name"], "mode": "benchmark_test",
                          "execution": "benchmark", "episodes": 120, "session_id": old["session_id"]})
            self.assertEqual(error.exception.code, 400)

    def test_benchmark_reports_limit_without_claiming_100_percent(self):
        class FailingAgent(CountingAgent):
            def choose_action(self, observation, greedy=False):
                return "up"

        with patch.dict(training_app.LEARNING_AGENTS, {"benchmark_test": FailingAgent}):
            final = self.wait(self.post("/api/train", {"map_name": MAP["name"],
                              "mode": "benchmark_test", "execution": "benchmark", "max_hours": 0.000000001}))
            self.assertGreaterEqual(final["training_seconds"], 0.000000001 * 3600)
            self.assertEqual(final["benchmark"]["status"], "limit")
            self.assertEqual(final["benchmark"]["max_hours"], 0.000000001)
            self.assertIsNone(final["success_100_milestone"])

    def test_benchmark_can_be_stopped_and_cannot_change_training_mode(self):
        entered, release = threading.Event(), threading.Event()

        class BlockingAgent(CountingAgent):
            def choose_action(self, observation, greedy=False):
                if not greedy:
                    entered.set()
                    release.wait(timeout=5)
                return super().choose_action(observation, greedy)

        with patch.dict(training_app.LEARNING_AGENTS, {"benchmark_test": BlockingAgent}):
            job = self.post("/api/train", {"map_name": MAP["name"], "mode": "benchmark_test",
                            "execution": "benchmark", "episodes": 120})
            try:
                self.assertTrue(entered.wait(timeout=5))
                with self.assertRaises(HTTPError) as error:
                    self.post("/api/train/control", {"session_id": job["session_id"],
                              "action": "skip", "episodes": 5})
                self.assertEqual(error.exception.code, 400)
                self.post("/api/train/control", {"session_id": job["session_id"], "action": "pause"})
            finally:
                release.set()
            final = self.wait(job)
            self.assertEqual(final["benchmark"]["status"], "cancelled")
            self.assertIsNone(final["success_100_milestone"])

    def test_sgcrl_train_evaluate_and_restart(self):
        import torch
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            settings = {**SETTINGS, "crl_hidden_size": 16, "crl_batch_size": 8,
                        "crl_min_replay_size": 2}
            with patch.object(training_app, "load_settings", return_value=settings), \
                    patch.object(training_app, "configure_ppo_cpu_threads", return_value=1):
                first = self.wait(self.post("/api/train", {"map_name": MAP["name"],
                                       "mode": "sgcrl", "episodes": 4}))
                sid = first["session_id"]
                training_app.learning_sessions.clear()
                final = self.wait(self.post("/api/train", {"map_name": MAP["name"],
                                 "mode": "sgcrl", "episodes": 2, "session_id": sid}))
                self.assertEqual(final["completed_episodes"], 6)
                agent = training_app.learning_sessions[sid]["session"].agent
                self.assertGreater(agent._updates, 0)
                self.assertEqual(final["mode"], "sgcrl")
                self.assertGreater(len(final["frames"]), 1)
        finally:
            torch.set_num_threads(threads)

    def test_cppo_train_policy_evaluate_and_restart(self):
        import torch
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            settings = {**SETTINGS, "cppo_hidden_size": 16, "cppo_depth": 2,
                        "cppo_repr_dim": 8, "cppo_batch_size": 8, "cppo_rollout_steps": 8}
            with patch.object(training_app, "load_settings", return_value=settings), \
                    patch.object(training_app, "configure_ppo_cpu_threads", return_value=1):
                self.assertIn("cppo", [agent["key"] for agent in self.get("/api/agents")])
                first = self.wait(self.post("/api/train", {"map_name": MAP["name"],
                                       "mode": "cppo", "episodes": 4}))
                sid = first["session_id"]
                policy = self.post("/api/train/policy", {"session_id": sid, "mode": "cppo",
                                   "map_name": MAP["name"], "position": MAP["start"]})
                self.assertAlmostEqual(sum(policy["probabilities"].values()), 1, places=6)
                before = storage.read_json(training_app.saved_session_path(sid))["agent_checkpoint"]
                self.wait(self.post("/api/train/evaluate", {"map_name": MAP["name"],
                          "mode": "cppo", "session_id": sid}))
                self.assertEqual(before, storage.read_json(training_app.saved_session_path(sid))["agent_checkpoint"])
                training_app.learning_sessions.clear()
                final = self.wait(self.post("/api/train", {"map_name": MAP["name"], "mode": "cppo",
                                 "episodes": 2, "session_id": sid}))
                self.assertEqual(final["completed_episodes"], 6)
                self.assertEqual(final["mode"], "cppo")
                self.assertGreater(training_app.learning_sessions[sid]["session"].agent.updates, 0)
        finally:
            torch.set_num_threads(threads)

    def test_mismatched_map_agent_and_invalid_requests_are_rejected(self):
        first = self.wait(self.post("/api/train", {"map_name": MAP["name"], "episodes": 2}))
        sid = first["session_id"]
        other_map = {**MAP, "name": "別マップ", "goal": [1, 1]}
        storage.save_map(other_map)
        with patch.dict(training_app.LEARNING_AGENTS, {"counting": CountingAgent}):
            bad = [{"map_name": other_map["name"], "session_id": sid},
                   {"map_name": MAP["name"], "mode": "counting", "session_id": sid},
                   {"map_name": MAP["name"], "mode": []},
                   {"map_name": MAP["name"], "session_id": "../other"}]
            bad.extend({"map_name": MAP["name"], "episodes": count} for count in (0, -1, True, 1000001))
            for value in bad:
                with self.subTest(value=value):
                    with self.assertRaises(HTTPError) as error:
                        self.post("/api/train", value)
                    self.assertEqual(error.exception.code, 400)
        self.assertEqual(self.get("/api/result")["settings"]["episodes"], 2)

    def test_live_curve_is_available_and_concurrent_jobs_are_rejected(self):
        paused, release = threading.Event(), threading.Event()

        class BlockingAgent(CountingAgent):
            def choose_action(self, observation, greedy=False):
                if self.episodes == 1 and not greedy:
                    paused.set()
                    release.wait(timeout=5)
                return super().choose_action(observation, greedy)

        with patch.dict(training_app.LEARNING_AGENTS, {"blocking": BlockingAgent}):
            job = self.post("/api/train", {"map_name": MAP["name"], "mode": "blocking", "episodes": 2})
            try:
                self.assertTrue(paused.wait(timeout=5))
                status = self.get("/api/train/status")
                self.assertEqual(status["preview"]["history"][-1]["episode"], 1)
                self.assertEqual(status["preview"]["settings"]["episodes"], 1)
                self.assertIsNone(status["preview"]["q_table"])
                with self.assertRaises(HTTPError) as error:
                    self.post("/api/train", {"map_name": MAP["name"], "episodes": 1})
                self.assertEqual(error.exception.code, 409)
            finally:
                release.set()
            self.assertEqual(self.wait(job)["settings"]["episodes"], 2)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is required")
    def test_grag_can_train_continue_and_restore_through_the_same_api(self):
        import torch
        from gridworld.training_runtime.tests.test_ppo_session import SETTINGS as ppo_settings
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with patch.object(training_app, "load_settings", return_value=ppo_settings.copy()):
                first = self.wait(self.post("/api/train", {"map_name": MAP["name"],
                                       "mode": "grag", "episodes": 2}))
                sid = first["session_id"]
                self.assertEqual(first["settings"]["episodes"], 2)
                training_app.learning_sessions.clear()
                final = self.wait(self.post("/api/train", {"map_name": MAP["name"],
                                 "mode": "grag", "episodes": 2, "session_id": sid}))
                self.assertEqual(final["settings"]["episodes"], 4)
                saved = storage.read_json(training_app.saved_session_path(sid))
                self.assertIn("_actor_optimizer", saved["agent_checkpoint"]["optimizers"])
                self.assertIn("_encoder_optimizer", saved["agent_checkpoint"]["optimizers"])
                self.assertIsNotNone(saved["agent_checkpoint"]["generator"])
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
