"""GRAG episode boundaries in the training runtime."""

import unittest
from unittest.mock import patch

import torch

from agents.grag_impl.grag import GRAG
from agents.grag_impl.components.losses import compute_crl_loss
from agents.grag_impl.grag import state_action_encoder
from gridworld.storage import DEFAULT_SETTINGS
from gridworld.training_runtime import TrainingSession


MAP = {"name": "boundary", "rows": 2, "cols": 2, "start": [0, 0],
       "goal": [1, 1], "obstacles": []}


class EpisodeBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_timeout_is_recorded_by_runtime(self):
        session = TrainingSession(MAP, {**DEFAULT_SETTINGS, "seed": 7,
                                         "max_steps": 1}, GRAG)
        session.run_batch(2)
        self.assertEqual(session.completed_episodes, 2)
        self.assertEqual([t.episode_end for t in session.agent.memory], [True, True])

    def test_timeout_is_not_included_in_following_success(self):
        agent = GRAG({"seed": 7})
        observation = {**MAP, "position": [0, 0]}
        for success in (False, False, True):
            action = agent.choose_action(observation)
            agent.learn(observation, action, float(success), observation, success)
            if success or len(agent.memory) == 1:
                agent.end_episode()
        agent._goal_embedding_cache[agent.memory[0].goal_prompt] = torch.zeros(384)
        with patch.object(agent, "_encode_crl_features", wraps=agent._encode_crl_features) as encode:
            loss = compute_crl_loss(agent, agent.memory, state_action_encoder, None)
        self.assertIsNotNone(loss)
        self.assertEqual(encode.call_args_list[0].args[1].shape[0], 2)
        self.assertTrue(agent.memory[-1].episode_end)

    def test_end_after_rollout_was_consumed(self):
        agent = GRAG({"seed": 7, "grag_positive_update_count": 1})
        observation = {**MAP, "position": [0, 0]}
        action = agent.choose_action(observation)
        with patch.object(agent, "_update_grag", return_value=True), patch.object(
                agent, "_update_ppo", side_effect=agent.memory.clear):
            agent.learn(observation, action, 1., observation, True)
        agent.end_episode()
        self.assertEqual(agent.memory, [])


if __name__ == "__main__":
    unittest.main()
