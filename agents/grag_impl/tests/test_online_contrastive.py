import unittest
from unittest.mock import patch

import torch

from agents.grag_impl.grag import GRAG


class OnlineContrastiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_each_step_uses_only_current_trajectory_action_outcomes(self):
        agent = GRAG({"seed": 7, "actions": (0, 1), "grag_update_interval": 100,
                      "grag_contrastive_epochs": 2})
        agent.initialize_networks(1, 1)
        with patch.object(agent.goal_encoder, "forward", wraps=agent.goal_encoder.forward) as encode, \
             patch.object(agent.contrastive_optimizer, "step", wraps=agent.contrastive_optimizer.step) as step:
            for state, next_state in ((1., 2.), (2., 3.), (3., 3.)):
                action = agent.choose_action([state])
                agent.learn([state], action, 0, [next_state], False)
            self.assertEqual(step.call_count, 4)
            self.assertEqual([call.args[0][:, 0].tolist() for call in encode.call_args_list],
                             [[2., 3.], [2., 3.], [2., 3., 3.], [2., 3., 3.]])
            agent.end_episode()
            action = agent.choose_action([10.])
            agent.learn([10.], action, 0, [11.], False)
            self.assertEqual(step.call_count, 4)
            action = agent.choose_action([11.])
            agent.learn([11.], action, 0, [12.], False)
            self.assertEqual(encode.call_args.args[0][:, 0].tolist(), [11., 12.])
        self.assertEqual(len(agent._update_batch), 5)

    def test_first_collision_has_finite_loss_and_updates_each_epoch(self):
        agent = GRAG({"seed": 7, "actions": (0, 1)})
        agent.initialize_networks(1, 1)
        action = agent.choose_action([1.])
        with patch.object(agent.contrastive_optimizer, "step", wraps=agent.contrastive_optimizer.step) as step:
            agent.learn([1.], action, 0, [1.], False)
        self.assertEqual(step.call_count, 0)
        self.assertTrue(torch.isfinite(torch.tensor(agent.contrastive_loss)))

    def test_each_step_and_scheduled_updates_both_run(self):
        agent = GRAG({"seed": 7, "actions": (0, 1), "grag_update_interval": 2,
                      "grag_contrastive_epochs": 2, "target_goal_encoder": lambda obs: [0.]})
        agent.initialize_networks(1, 1)
        with patch.object(agent, "_update_contrastive", wraps=agent._update_contrastive) as online, \
             patch.object(agent, "_update_batched_contrastive", wraps=agent._update_batched_contrastive) as batch, \
             patch.object(agent.contrastive_optimizer, "step", wraps=agent.contrastive_optimizer.step) as step:
            action = agent.choose_action([0.])
            agent.learn([0.], action, 0, [1.], False)
            agent.end_episode()
            action = agent.choose_action([1.])
            agent.learn([1.], action, 0, [2.], False)
        self.assertEqual(online.call_count, 2)
        self.assertEqual(batch.call_count, 1)
        self.assertEqual(step.call_count, 2)
        self.assertEqual(len(batch.call_args.args[0]), 2)
        self.assertEqual(agent._contrastive_pending, [])

    def test_scheduled_update_waits_for_distinct_episode_negatives(self):
        agent = GRAG({"seed": 7, "actions": (0, 1), "grag_contrastive_epochs": 2})
        agent.initialize_networks(1, 1)
        first = {"state": torch.tensor([0.]), "next_state": torch.tensor([1.]),
                 "action": 0, "episode_id": 0}
        second = {"state": torch.tensor([1.]), "next_state": torch.tensor([2.]),
                  "action": 1, "episode_id": 1}
        with patch.object(agent.contrastive_optimizer, "step", wraps=agent.contrastive_optimizer.step) as step:
            self.assertFalse(agent._update_batched_contrastive([first]))
            self.assertEqual(step.call_count, 0)
            self.assertTrue(agent._update_batched_contrastive([second]))
            self.assertEqual(step.call_count, 2)
        self.assertEqual(agent._contrastive_pending, [])


if __name__ == "__main__":
    unittest.main()
