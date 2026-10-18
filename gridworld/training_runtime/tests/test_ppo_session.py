import importlib.util
import json
import unittest
from unittest.mock import patch

from agents.grag import GRAG
from gridworld.storage import DEFAULT_SETTINGS
from gridworld.training_runtime import TrainingSession
from gridworld.training_runtime.ppo_adapter import GRAGSession, PPOAgentSession
from .test_session import MAP


SETTINGS = {**DEFAULT_SETTINGS, "episodes": 6, "max_steps": 6, "seed": 7,
            "ppo_hidden_size": 8, "ppo_epochs": 1, "ppo_minibatch_size": 4,
            "ppo_rollout_steps": 4, "ppo_target_kl": None}


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is required")
class PPOSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.threads)

    def test_copied_ppo_trains_reproducibly_without_using_grag(self):
        baseline = TrainingSession(MAP, SETTINGS, PPOAgentSession)
        baseline.run_batch(6)
        with patch.object(GRAG, "choose_action", side_effect=AssertionError("GRAG was used")):
            copied = TrainingSession(MAP, SETTINGS, PPOAgentSession)
            copied.run_batch(6)
        self.assertEqual(copied.history, baseline.history)
        self.assertEqual(copied.visits, baseline.visits)
        self.assertEqual(copied.agent.checkpoint(), baseline.agent.checkpoint())
        self.assertNotIsInstance(copied.agent, GRAG)

    def test_ppo_checkpoint_restores_optimizer_rng_and_matches_uninterrupted_learning(self):
        for agent_type in (GRAGSession, PPOAgentSession):
            with self.subTest(agent=agent_type.__name__):
                continuous = TrainingSession(MAP, SETTINGS, agent_type)
                continuous.run_batch(6)
                split = TrainingSession(MAP, SETTINGS, agent_type)
                split.run_batch(3)
                split.evaluate()
                saved = json.loads(json.dumps(split.result(checkpoint=True)))
                restored = TrainingSession.restore(saved, agent_type)
                restored.run_batch(3)
                self.assertEqual(restored.history, continuous.history)
                self.assertEqual(restored.visits, continuous.visits)
                self.assertEqual(restored.agent.checkpoint(), continuous.agent.checkpoint())

    def test_checkpoint_in_middle_of_trial_restores_ppo_rollout_and_optimizer(self):
        for agent_type in (GRAGSession, PPOAgentSession):
            with self.subTest(agent=agent_type.__name__):
                continuous = TrainingSession(MAP, SETTINGS, agent_type)
                continuous.run_batch(6)
                split = TrainingSession(MAP, SETTINGS, agent_type)
                split.step()
                split.step()
                saved = json.loads(json.dumps(split.result(checkpoint=True)))
                self.assertTrue(saved["episode_state"]["active"])
                self.assertGreater(len(saved["agent_checkpoint"]["rollout"]), 0)
                restored = TrainingSession.restore(saved, agent_type)
                restored.run_batch(6)
                self.assertEqual(restored.history, continuous.history)
                self.assertEqual(restored.visits, continuous.visits)
                self.assertEqual(restored.agent.checkpoint(), continuous.agent.checkpoint())

    def test_current_ppo_evaluation_leaves_training_state_unchanged(self):
        for agent_type in (GRAGSession, PPOAgentSession):
            with self.subTest(agent=agent_type.__name__):
                session = TrainingSession(MAP, SETTINGS, agent_type)
                session.run_batch(2)
                before = session.agent.checkpoint()
                frames = session.evaluate()
                self.assertGreater(len(frames), 1)
                self.assertEqual(session.agent.checkpoint(), before)

    def test_original_grag_works_with_common_evaluation_without_module_deepcopy(self):
        session = TrainingSession(MAP, SETTINGS, GRAG)
        session.run_batch(1)
        before = session.agent.export()
        rng = session.agent._generator.get_state().tolist()
        session.evaluate()
        self.assertEqual(session.agent.export(), before)
        self.assertEqual(session.agent._generator.get_state().tolist(), rng)


if __name__ == "__main__":
    unittest.main()
