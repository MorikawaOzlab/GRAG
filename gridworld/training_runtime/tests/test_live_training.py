"""ライブ学習・高速スキップ・途中停止と、メモリ上限の契約を確認する。"""

from copy import deepcopy
import json
import importlib.util
import os
import threading
import time
import unittest
from unittest.mock import patch

from gridworld import storage
from gridworld import training_app
from gridworld.training_runtime import TrainingSession
from . import test_api
from .test_session import CountingAgent, MAP, SETTINGS


TRAP = {"name": "移動できないマップ", "rows": 2, "cols": 2,
        "start": [0, 0], "goal": [1, 1], "obstacles": [[0, 1], [1, 0]]}


class StepSessionTests(unittest.TestCase):
    def test_mid_trial_checkpoint_resumes_the_same_step_and_random_state(self):
        continuous = TrainingSession(MAP, {**SETTINGS, "max_steps": 20})
        split = TrainingSession(MAP, {**SETTINGS, "max_steps": 20})
        for _ in range(2):
            continuous.step()
            split.step()
        restored = TrainingSession.restore(json.loads(json.dumps(split.result(checkpoint=True))))
        for _ in range(30):
            continuous.step()
            restored.step()
        self.assertEqual(restored.history, continuous.history)
        self.assertEqual(restored.live, continuous.live)
        self.assertEqual(restored.visits, continuous.visits)
        self.assertEqual(restored.agent.checkpoint(), continuous.agent.checkpoint())

    def test_limit_change_applies_to_next_trial(self):
        session = TrainingSession(TRAP, {**SETTINGS, "max_steps": 5})
        session.step()
        session.step()
        session.set_max_steps(2)
        for _ in range(3):
            session.step()
        self.assertEqual(session.history[-1]["steps"], 5)
        session.step()
        state = session.step()
        self.assertTrue(state["finished"])
        self.assertEqual(session.history[-1]["steps"], 2)
        self.assertEqual(session.completed_episodes, 2)

    def test_curve_storage_is_bounded_and_total_survives_restore(self):
        session = TrainingSession(TRAP, {**SETTINGS, "max_steps": 1})
        session.run_batch(5000)
        self.assertLessEqual(len(session.history), session.MAX_CURVE_POINTS)
        self.assertEqual(session.completed_episodes, 5000)
        self.assertEqual(session.history[-1]["episode"], 5000)
        self.assertEqual(session.frames, [])
        self.assertFalse(hasattr(session, "trajectories"))
        restored = TrainingSession.restore(json.loads(json.dumps(session.result(checkpoint=True))))
        restored.run_batch(25)
        self.assertEqual(restored.completed_episodes, 5025)
        self.assertEqual(restored.total_steps, 5025)
        self.assertLessEqual(len(restored.history), restored.MAX_CURVE_POINTS)
        self.assertEqual(restored.result()["settings"]["episodes"], 5025)


class LiveApiTests(unittest.TestCase):
    setUp = test_api.SessionApiTests.setUp
    get = test_api.SessionApiTests.get
    post = test_api.SessionApiTests.post

    def tearDown(self):
        job = training_app.current_job
        if job and job.get("_interactive") and job["status"] in ("running", "paused"):
            with training_app.job_lock:
                job["_control"] = "close"
                job["_wake"].set()
            deadline = time.monotonic() + 10
            while job["status"] not in ("complete", "error") and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertIn(job["status"], ("complete", "error"))
        test_api.SessionApiTests.tearDown(self)

    def state_when(self, predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = self.get("/api/train/status")
            self.assertNotEqual(state["status"], "error", state.get("error"))
            if predicate(state):
                return state
            time.sleep(.01)
        self.fail("指定した学習状態に到達しませんでした。")

    def change(self, action, **fields):
        return self.post("/api/train/control", {
            "action": action, "session_id": training_app.current_job["session_id"], **fields})

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is required")
    def test_ppo_cpu_threads_leave_two_logical_cpus_free(self):
        import torch
        from gridworld.training_runtime.ppo_adapter import GRAGSession, PPOAgentSession
        for mode, agent_type in (("grag", GRAGSession), ("ppo", PPOAgentSession)):
            with self.subTest(mode=mode):
                measured = []

                class ThreadCheckedPPO(agent_type):
                    def choose_action(self, observation, greedy=False):
                        measured.append(torch.get_num_threads())
                        return super().choose_action(observation, greedy)

                    def learn(self, *args):
                        measured.append(torch.get_num_threads())
                        return super().learn(*args)

                with patch.dict(training_app.LEARNING_AGENTS, {mode: ThreadCheckedPPO}):
                    self.post("/api/train", {"map_name": MAP["name"], "mode": mode,
                              "execution": "fast", "episodes": 2,
                              "training_settings": {"max_steps": 3, "ppo_hidden_size": 8}})
                    state = self.state_when(lambda item: item["status"] == "paused")
                    expected = max(1, (os.cpu_count() or 1) - 2)
                    self.assertEqual(state["cpu_threads"], expected)
                    self.assertTrue(measured)
                    self.assertEqual(set(measured), {expected})

    def test_standalone_ppo_and_grag_are_both_selectable(self):
        from urllib.request import urlopen
        agents = {item["key"]: item["name"] for item in self.get("/api/agents")}
        self.assertEqual(agents["ppo"], "PPOエージェント")
        self.assertIn("grag", agents)
        with urlopen(self.base + "/", timeout=5) as response:
            page = response.read().decode("utf-8")
        self.assertIn('<option value="ppo">PPOエージェント</option>', page)
        self.assertIn('<option value="grag">GRAG</option>', page)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is required")
    def test_standalone_ppo_observes_pauses_restores_and_skips(self):
        from urllib.error import HTTPError
        from agents.ppo_agent import PPOAgent
        job = self.post("/api/train", {"map_name": MAP["name"], "mode": "ppo",
                        "execution": "paused", "step_delay_ms": 2000,
                        "training_settings": {"max_steps": 6, "ppo_hidden_size": 8,
                                              "ppo_epochs": 1, "ppo_rollout_steps": 4}})
        sid = job["session_id"]
        self.state_when(lambda item: item["status"] == "paused")
        self.change("observe")
        self.state_when(lambda item: item.get("preview", {}).get("total_steps", 0) >= 1)
        self.change("pause")
        paused = self.state_when(lambda item: item["status"] == "paused")
        saved = storage.read_json(training_app.saved_session_path(sid))
        self.assertTrue(saved["episode_state"]["active"])
        self.assertGreater(len(saved["agent_checkpoint"]["rollout"]), 0)
        self.assertEqual(saved["frames"], [])
        self.assertIsInstance(training_app.learning_sessions[sid]["session"].agent, PPOAgent)
        with self.assertRaises(HTTPError) as error:
            self.post("/api/train", {"map_name": MAP["name"], "mode": "grag",
                                     "session_id": sid, "execution": "paused"})
        self.assertEqual(error.exception.code, 400)
        training_app.learning_sessions.clear()
        resumed = self.post("/api/train", {"map_name": MAP["name"], "mode": "ppo",
                                           "session_id": sid, "execution": "paused"})
        restored = self.state_when(lambda item: item["job_id"] == resumed["job_id"]
                                              and item["status"] == "paused")
        self.assertEqual(restored["preview"]["live"], paused["preview"]["live"])
        agent = training_app.learning_sessions[sid]["session"].agent
        self.assertIsInstance(agent, PPOAgent)
        self.assertEqual(json.loads(json.dumps(agent.checkpoint())), saved["agent_checkpoint"])
        self.change("skip", episodes=2)
        finished = self.state_when(lambda item: item["status"] == "paused"
                                              and item["preview"]["completed_episodes"] == 2)
        self.assertEqual(finished["preview"]["frames"], [])
        self.assertIs(training_app.learning_sessions[sid]["session"].agent, agent)

    def test_start_at_zero_and_keep_only_current_position(self):
        job = self.post("/api/train", {"map_name": MAP["name"], "execution": "paused",
                                      "training_settings": {"max_steps": 17}})
        state = self.state_when(lambda item: item["status"] == "paused")
        self.assertEqual(state["preview"]["completed_episodes"], 0)
        self.assertEqual(state["preview"]["live"]["position"], MAP["start"])
        self.assertEqual(state["preview"]["settings"]["max_steps"], 17)
        saved = storage.read_json(training_app.saved_session_path(job["session_id"]))
        self.assertEqual(saved["frames"], [])
        self.assertEqual(saved["history"], [])
        self.assertNotIn("trajectories", saved)
        self.assertNotIn("_wake", state)

    def test_observation_pause_restores_mid_trial_without_resetting(self):
        storage.save_map(TRAP)
        job = self.post("/api/train", {"map_name": TRAP["name"], "execution": "observe",
                      "step_delay_ms": 20, "training_settings": {"max_steps": 100}})
        self.state_when(lambda item: item.get("preview", {}).get("live", {}).get("step", 0) >= 3)
        self.change("pause")
        paused = self.state_when(lambda item: item["status"] == "paused")
        before = deepcopy(paused["preview"])
        time.sleep(.08)
        self.assertEqual(self.get("/api/train/status")["preview"]["live"], before["live"])
        self.assertEqual(before["completed_episodes"], 0)
        training_app.learning_sessions.clear()
        resumed = self.post("/api/train", {"map_name": TRAP["name"], "execution": "paused",
                                          "session_id": job["session_id"]})
        self.state_when(lambda item: item["job_id"] == resumed["job_id"] and item["status"] == "paused")
        self.change("observe")
        after = self.state_when(lambda item: item.get("preview", {}).get("live", {}).get("step", 0) > before["live"]["step"])
        self.assertEqual(after["preview"]["live"]["episode"], 1)

    def test_fast_learning_publishes_curve_heatmap_and_can_stop_early(self):
        blocked, release = threading.Event(), threading.Event()

        class SlowAgent(CountingAgent):
            def choose_action(self, observation, greedy=False):
                if not greedy:
                    time.sleep(.002)
                    if self.episodes == 30:
                        blocked.set()
                        release.wait(timeout=5)
                return super().choose_action(observation, greedy)

        with patch.dict(training_app.LEARNING_AGENTS, {"slow": SlowAgent}):
            self.post("/api/train", {"map_name": MAP["name"], "mode": "slow",
                                    "execution": "fast", "episodes": 1000})
            try:
                self.assertTrue(blocked.wait(timeout=5))
                live = self.get("/api/train/status")
                self.assertEqual(live["execution"], "fast")
                self.assertGreater(live["preview"]["history"][-1]["episode"], 0)
                self.assertGreater(sum(map(sum, live["preview"]["visits"])), 0)
                self.change("pause")
            finally:
                release.set()
            paused = self.state_when(lambda item: item["status"] == "paused")
            self.assertLess(paused["preview"]["completed_episodes"], 1000)
            self.assertGreater(paused["preview"]["completed_episodes"], 0)
            self.assertEqual(paused["preview"]["frames"], [])
            before = paused["preview"]["total_steps"]
            time.sleep(.05)
            self.assertEqual(self.get("/api/train/status")["preview"]["total_steps"], before)

    def test_skip_exact_count_then_continue_the_same_agent(self):
        with patch.dict(training_app.LEARNING_AGENTS, {"counting": CountingAgent}):
            job = self.post("/api/train", {"map_name": MAP["name"], "mode": "counting",
                                          "execution": "paused"})
            self.state_when(lambda item: item["status"] == "paused")
            agent = training_app.learning_sessions[job["session_id"]]["session"].agent
            self.change("skip", episodes=7)
            completed = self.state_when(lambda item: item["status"] == "paused"
                                        and item.get("preview", {}).get("completed_episodes") == 7)
            self.assertEqual(completed["completed"], 7)
            self.assertIs(training_app.learning_sessions[job["session_id"]]["session"].agent, agent)
            self.assertEqual(agent.episodes, 7)
            self.change("observe")
            self.state_when(lambda item: item["execution"] == "observe" and item["status"] == "running")
            self.change("pause")
            self.state_when(lambda item: item["status"] == "paused")
            self.assertIs(training_app.learning_sessions[job["session_id"]]["session"].agent, agent)

    def test_skip_from_observation_returns_to_observation(self):
        with patch.dict(training_app.LEARNING_AGENTS, {"counting": CountingAgent}):
            self.post("/api/train", {"map_name": MAP["name"], "mode": "counting",
                                    "execution": "observe", "step_delay_ms": 320})
            self.state_when(lambda item: item.get("preview", {}).get("live", {}).get("step", 0) >= 1)
            self.change("skip", episodes=5)
            state = self.state_when(lambda item: item.get("last_skip") == 5 and item["execution"] == "observe")
            self.assertEqual(state["preview"]["completed_episodes"], 5)

    def test_limit_control_and_new_experiment_preserve_paused_model(self):
        first = self.post("/api/train", {"map_name": MAP["name"], "execution": "paused"})
        self.state_when(lambda item: item["status"] == "paused")
        self.change("limit", max_steps=3)
        self.state_when(lambda item: item.get("preview", {}).get("settings", {}).get("max_steps") == 3)
        self.change("skip", episodes=2)
        state = self.state_when(lambda item: item["status"] == "paused"
                                and item.get("preview", {}).get("completed_episodes") == 2)
        self.assertEqual(state["preview"]["settings"]["max_steps"], 3)
        second = self.post("/api/train", {"map_name": MAP["name"], "execution": "paused"})
        state = self.state_when(lambda item: item["job_id"] == second["job_id"] and item["status"] == "paused")
        self.assertEqual(state["preview"]["completed_episodes"], 0)
        self.assertNotEqual(first["session_id"], second["session_id"])
        self.assertTrue(training_app.saved_session_path(first["session_id"]).is_file())
        self.assertNotIn(first["session_id"], training_app.learning_sessions)


if __name__ == "__main__":
    unittest.main()
