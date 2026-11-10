"""Configuration and reproducibility for the SGCRL grid adaptation."""
import unittest

import torch

from agents.grag import GRAG
from gridworld.training_runtime import TrainingSession
from .test_grag_update import MAP, SETTINGS


class TrainingSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_paper_defaults_and_ppo_settings_do_not_change_algorithm(self):
        agent = GRAG({"ppo_learning_rate": 1., "discount_factor": 0.5})
        self.assertEqual(agent.gamma, 0.99)
        self.assertEqual(agent.learning_rate, 3e-4)
        self.assertEqual(agent.hidden_size, 256)
        self.assertEqual(agent.repr_dim, 64)
        self.assertEqual(agent.min_replay, 10000)
        self.assertEqual(agent.capacity, 1000000)
        self.assertEqual(agent.batch_size, 256)

    def test_invalid_settings_fail_at_construction(self):
        for settings in ({"crl_gamma": 1}, {"crl_batch_size": 1},
                         {"crl_hidden_size": True}, {"crl_learning_rate": float("nan")},
                         {"crl_min_replay_size": 10, "crl_replay_capacity": 2},
                         {"crl_target_entropy": 2}, {"crl_updates_per_step": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                GRAG(settings)

    def test_initial_random_data_and_network_initialization_preserve_global_rng(self):
        rng = torch.random.get_rng_state().clone()
        sessions = [TrainingSession(MAP, SETTINGS, GRAG) for _ in range(2)]
        for _ in range(12):
            self.assertEqual(sessions[0].step(), sessions[1].step())
        self.assertEqual(sessions[0].agent.checkpoint(), sessions[1].agent.checkpoint())
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_initial_data_threshold_prevents_updates(self):
        session = TrainingSession(MAP, {**SETTINGS, "crl_min_replay_size": 100}, GRAG)
        session.run_batch(2)
        self.assertEqual(session.agent._updates, 0)
        self.assertGreater(session.agent._replay_size, 0)

    def test_different_maze_is_rejected(self):
        agent = GRAG(SETTINGS)
        obs = {**MAP, "position": [0, 0]}
        agent.choose_action(obs)
        with self.assertRaises(ValueError):
            agent.choose_action({**obs, "obstacles": [[0, 1]]})


if __name__ == "__main__":
    unittest.main()
