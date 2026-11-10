"""行動確率の正しさ、読み取り時の副作用、AIと環境の独立性を検証する。"""

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from agents.grag import GRAG
from agents.ppo_agent import PPOAgent
from agents.q_learning import ACTIONS, QLearningAgent
from gridworld import storage
from gridworld import training_app
from gridworld.training_runtime import TrainingSession
from gridworld.training_runtime.ppo_adapter import GRAGSession, PPOAgentSession
from . import test_api, test_live_training
from .test_session import MAP, SETTINGS, CountingAgent


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
PPO_SETTINGS = {**SETTINGS, "seed": 7, "max_steps": 6, "ppo_hidden_size": 8,
                "ppo_epochs": 1, "ppo_rollout_steps": 4, "ppo_minibatch_size": 4}


class QPolicyTests(unittest.TestCase):
    def test_ai_configuration_is_separate_and_limit_uses_public_method(self):
        session = TrainingSession(MAP, SETTINGS)
        session.agent.settings["discount_factor"] = 0.01
        self.assertEqual(session.settings["discount_factor"], SETTINGS["discount_factor"])
        self.assertEqual(SETTINGS["discount_factor"], 0.95)
        session.set_max_steps(7)
        self.assertEqual(session.agent.settings["max_steps"], SETTINGS["max_steps"])
        session.step()
        self.assertEqual(session.agent.settings["max_steps"], 7)

    def test_probabilities_include_exploration_and_agree_with_actual_sampling(self):
        agent = QLearningAgent({**SETTINGS, "epsilon": 0.2})
        agent.q[(1, 0)] = [3.0, 3.0, 0.0, 0.0]
        observation = {**MAP, "position": [1, 0]}
        before = deepcopy(agent.checkpoint())
        probabilities = agent.action_probabilities(observation)
        self.assertEqual(agent.checkpoint(), before)
        self.assertAlmostEqual(probabilities["up"], 0.85)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        sampled = QLearningAgent(SETTINGS.copy())
        sampled.load_checkpoint(before)
        counts = dict.fromkeys(ACTIONS, 0)
        for _ in range(10000):
            counts[sampled.choose_action(observation)] += 1
        for action in ACTIONS:
            self.assertAlmostEqual(counts[action] / 10000, probabilities[action], delta=0.015)

    def test_unvisited_state_inspection_does_not_create_q_values_or_consume_rng(self):
        agent = QLearningAgent({**SETTINGS, "epsilon": 1.0})
        before = deepcopy(agent.checkpoint())
        self.assertEqual(agent.action_probabilities({**MAP, "position": [1, 1]}),
                         dict.fromkeys(ACTIONS, 0.25))
        self.assertEqual(agent.checkpoint(), before)

    def test_cell_validation_wall_goal_and_unsupported_ai(self):
        session = TrainingSession(MAP, SETTINGS)
        before = deepcopy(session.result(checkpoint=True))
        wall = session.inspect_policy([0, 1])
        self.assertTrue(wall["wall"])
        self.assertIsNone(wall["probabilities"])
        goal = session.inspect_policy(MAP["goal"])
        self.assertTrue(goal["goal"])
        self.assertTrue(goal["supported"])
        self.assertIn("right", goal["blocked_actions"])
        self.assertEqual(session.result(checkpoint=True), before)
        for bad in (None, [], [0], [True, 0], [0.0, 0], [-1, 0], [0, 3], "0,0"):
            with self.subTest(position=bad):
                with self.assertRaises(ValueError):
                    session.inspect_policy(bad)
        unsupported = TrainingSession(MAP, SETTINGS, CountingAgent).inspect_policy([1, 0])
        self.assertFalse(unsupported["supported"])

    def test_invalid_ai_probabilities_are_rejected(self):
        class InvalidAgent(CountingAgent):
            def action_probabilities(self, observation):
                return dict.fromkeys(ACTIONS, float("nan"))
        with self.assertRaises(ValueError):
            TrainingSession(MAP, SETTINGS, InvalidAgent).inspect_policy([1, 0])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class NeuralPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.threads)

    def test_zero_training_inspection_preserves_model_and_future_random_actions(self):
        for agent_type in (GRAGSession, PPOAgentSession):
            with self.subTest(agent=agent_type.__name__):
                session = TrainingSession(MAP, PPO_SETTINGS, agent_type)
                control = TrainingSession(MAP, PPO_SETTINGS, agent_type)
                before = deepcopy(session.result(checkpoint=True))
                rng = self.torch.random.get_rng_state().clone()
                probabilities = session.inspect_policy([1, 0])["probabilities"]
                self.assertEqual(session.result(checkpoint=True), before)
                self.assertTrue(self.torch.equal(self.torch.random.get_rng_state(), rng))
                self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=6)
                self.assertTrue(all(0 < value < 1 for value in probabilities.values()))
                self.assertEqual(session.inspect_policy([1, 0])["probabilities"], probabilities)
                session.run_batch(3)
                control.run_batch(3)
                self.assertEqual(session.agent.checkpoint(), control.agent.checkpoint())
                self.assertEqual(session.history, control.history)

    def test_initialized_policy_inspection_preserves_pending_action_and_rollout(self):
        for agent_type in (GRAGSession, PPOAgentSession):
            with self.subTest(agent=agent_type.__name__):
                session = TrainingSession(MAP, PPO_SETTINGS, agent_type)
                session.step()
                before = deepcopy(session.result(checkpoint=True))
                probabilities = session.inspect_policy([1, 1])["probabilities"]
                self.assertEqual(session.result(checkpoint=True), before)
                observation = {**MAP, "position": [1, 1]}
                action = session.agent.choose_action(observation)
                uses_pending = hasattr(session.agent, "_pending")
                pending = deepcopy(session.agent._pending if uses_pending else session.agent.checkpoint())
                generator = session.agent._generator.get_state().clone()
                again = session.agent.action_probabilities(observation)
                self.assertGreater(again[action], 0)
                if uses_pending:
                    self.assertAlmostEqual(again[action], math.exp(pending[2]), places=6)
                self.assertEqual(again, probabilities)
                self.assertEqual(session.agent._pending if uses_pending else session.agent.checkpoint(), pending)
                self.assertTrue(self.torch.equal(generator, session.agent._generator.get_state()))
                session.agent.learn(observation, action, 0, observation, False)

    def test_agents_can_import_and_run_without_environment_modules(self):
        source = """
import importlib.abc, sys
class RejectEnvironment(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"gridworld", "world", "storage", "training", "training_app", "training_runtime", "web_common"}:
            raise AssertionError("AI imported environment: " + fullname)
sys.meta_path.insert(0, RejectEnvironment())
from agents.grag import GRAG
from agents.ppo_agent import PPOAgent
from agents.q_learning import QLearningAgent
import torch
torch.set_num_threads(1)
settings = {"seed": 7, "epsilon": 0.2, "learning_rate": 0.1, "discount_factor": 0.95,
            "epsilon_decay": 0.9, "min_epsilon": 0.05, "ppo_hidden_size": 8, "ppo_epochs": 1}
observation = {"name": "AI standalone", "rows": 2, "cols": 3, "start": [0, 0],
               "goal": [1, 2], "obstacles": [[0, 1]], "position": [0, 0]}
for kind in (GRAG, PPOAgent, QLearningAgent):
    agent = kind(settings.copy())
    probabilities = agent.action_probabilities(observation)
    assert abs(sum(probabilities.values()) - 1) < 1e-5
    action = agent.choose_action(observation)
    agent.learn(observation, action, 0, observation, False)
    agent.end_episode()
    saved = agent.checkpoint()
    restored = kind(settings.copy())
    restored.load_checkpoint(saved)
    assert restored.action_probabilities(observation) == agent.action_probabilities(observation)
print("AI_ISOLATED")
"""
        result = subprocess.run([sys.executable, "-X", "utf8", "-c", source],
                                cwd=Path(__file__).resolve().parents[3],
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("AI_ISOLATED", result.stdout)


class PolicyApiTests(unittest.TestCase):
    setUp = test_api.SessionApiTests.setUp
    tearDown = test_api.SessionApiTests.tearDown
    get = test_api.SessionApiTests.get
    post = test_api.SessionApiTests.post
    wait = test_api.SessionApiTests.wait

    def test_saved_policy_inspection_does_not_start_jobs_write_files_or_cache_models(self):
        result = self.wait(self.post("/api/train", {"map_name": MAP["name"], "episodes": 2}))
        sid = result["session_id"]
        saved_path = training_app.saved_session_path(sid)
        before_file = saved_path.read_bytes()
        before_status = deepcopy(self.get("/api/train/status"))
        training_app.learning_sessions.clear()
        with patch.object(training_app.executor, "submit") as submit, \
             patch.object(training_app, "write_json") as write:
            policy = self.post("/api/train/policy", {
                "session_id": sid, "mode": "q_learning", "map_name": MAP["name"], "position": [1, 0]})
        self.assertEqual(policy["completed_episodes"], 2)
        self.assertEqual(set(policy["probabilities"]), set(ACTIONS))
        submit.assert_not_called()
        write.assert_not_called()
        self.assertEqual(training_app.learning_sessions, {})
        self.assertEqual(saved_path.read_bytes(), before_file)
        self.assertEqual(self.get("/api/train/status"), before_status)
        bad = [
            {"position": [False, 0]}, {"position": [4, 0]}, {"position": None},
            {"session_id": "../bad"}, {"mode": "grag"},
            {"mode": []}, {"map_name": "different"},
        ]
        for fields in bad:
            with self.subTest(fields=fields):
                with self.assertRaises(HTTPError) as error:
                    self.post("/api/train/policy", {"session_id": sid, "mode": "q_learning",
                              "map_name": MAP["name"], "position": [1, 0], **fields})
                self.assertEqual(error.exception.code, 400)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class LivePolicyApiTests(unittest.TestCase):
    setUp = test_api.SessionApiTests.setUp
    tearDown = test_live_training.LiveApiTests.tearDown
    get = test_api.SessionApiTests.get
    post = test_api.SessionApiTests.post
    state_when = test_live_training.LiveApiTests.state_when
    change = test_live_training.LiveApiTests.change

    def test_live_agents_publish_cell_probabilities_without_changing_learning(self):
        for mode in ("q_learning", "ppo", "grag"):
            with self.subTest(mode=mode):
                job = self.post("/api/train", {"map_name": MAP["name"], "mode": mode,
                                "execution": "observe", "step_delay_ms": 2000,
                                "training_settings": {key: value for key, value in PPO_SETTINGS.items() if key != "episodes"}})
                self.state_when(lambda item: item.get("preview", {}).get("total_steps", 0) >= 1)
                self.change("pause")
                state = self.state_when(lambda item: item["status"] == "paused")
                session = training_app.learning_sessions[job["session_id"]]["session"]
                before = deepcopy(session.result(checkpoint=True))
                with patch.object(session.agent, "choose_action", side_effect=AssertionError("Inspection chose an action")), \
                     patch.object(training_app, "write_json") as write:
                    result = self.post("/api/train/policy", {"session_id": job["session_id"],
                                       "mode": mode, "map_name": MAP["name"], "position": [1, 0]})
                self.assertEqual(result["total_steps"], state["preview"]["total_steps"])
                self.assertEqual(result["position"], [1, 0])
                self.assertEqual(result["blocked_actions"], ["down", "left"])
                self.assertEqual(session.result(checkpoint=True), before)
                write.assert_not_called()
