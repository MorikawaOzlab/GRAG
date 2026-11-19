import unittest
from unittest.mock import patch

import torch

from agents.grag_impl.grag import GRAG


class MemoryTests(unittest.TestCase):
    def test_configured_update_interval_controls_timing(self):
        agent = GRAG({"seed": 4, "actions": (0, 1), "grag_update_interval": 3})
        agent.initialize_networks(2, 1)
        with patch.object(agent, "_update_networks") as update:
            for step in range(3):
                action = agent.choose_action([1, 2])
                agent.learn([1, 2], action, 0, [2, 3], False)
                if step == 0:
                    agent.end_episode()
                self.assertEqual(update.call_count, 0)
            agent.end_episode()
            self.assertEqual(update.call_count, 1)
            self.assertEqual(len(update.call_args.args[0]), 3)

    def test_updates_every_128_steps_across_episodes(self):
        agent = GRAG({"seed": 4, "actions": (0, 1), "grag_update_interval": 128})
        agent.initialize_networks(2, 1)
        with patch.object(agent, "_update_networks") as update:
            for step in range(256):
                action = agent.choose_action([1, 2])
                agent.learn([1, 2], action, 0, [2, 3], False)
                if step == 49:
                    agent.end_episode()
                if step == 127:
                    self.assertEqual(update.call_count, 0)
                    agent.end_episode()
                    batch = update.call_args.args[0]
                    self.assertEqual([row["episode_id"] for row in batch], [0] * 50 + [1] * 78)
            self.assertEqual(update.call_count, 1)
            self.assertEqual(len(agent._update_batch), 128)
            self.assertEqual(len(agent.trajectory), 128)

    def test_transitions_and_episode_boundaries(self):
        agent = GRAG({"seed": 3, "actions": ("a", "b")})
        agent.initialize_networks(2, 1)
        observation, next_observation = [1., 2.], [3., 4.]
        probabilities = agent.action_probabilities(observation)
        action = agent.choose_action(observation)
        agent.learn(observation, action, 1, next_observation, True)
        observation[0] = 99
        next_observation[0] = 99
        record = agent.trajectory[0]
        self.assertEqual(record["observation"], [1., 2.])
        self.assertEqual(record["next_observation"], [3., 4.])
        self.assertTrue(torch.equal(record["state"], torch.tensor([1., 2.])))
        self.assertTrue(torch.equal(record["next_state"], torch.tensor([3., 4.])))
        self.assertAlmostEqual(record["log_prob"], torch.tensor(probabilities[action]).log().item(), places=6)
        self.assertTrue(record["reached_goal"])
        agent.end_episode()
        self.assertEqual(len(agent.episodes), 1)
        self.assertEqual(agent.trajectory, [])
        action = agent.choose_action([0., 0.])
        agent.learn([0., 0.], action, 0, [0., 1.], False)
        agent.end_episode()  # time limit, not goal
        self.assertEqual([len(episode) for episode in agent.episodes], [1, 1])
        self.assertFalse(agent.episodes[1][0]["reached_goal"])

    def test_pending_and_evaluation(self):
        agent = GRAG({"seed": 4, "actions": (0, 1)})
        agent.initialize_networks(2, 1)
        agent.choose_action([1, 2], greedy=True)
        self.assertIsNone(agent.pending)
        action = agent.choose_action([1, 2])
        with self.assertRaises(RuntimeError):
            agent.choose_action([1, 2])
        with self.assertRaises(RuntimeError):
            agent.end_episode()
        with self.assertRaises(RuntimeError):
            agent.learn([1, 2], 1 - action, 0, [2, 3], False)
        agent.choose_action([2, 3], greedy=True)
        agent.learn([1, 2], action, 0, [2, 3], False)
        self.assertIsNone(agent.pending)


if __name__ == "__main__":
    unittest.main()
