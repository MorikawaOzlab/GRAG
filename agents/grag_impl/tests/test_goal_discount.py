import unittest
from unittest.mock import patch

import torch

from agents.grag import GRAG


class GoalDiscountTests(unittest.TestCase):
    def setUp(self):
        self.agent = GRAG({"seed": 1, "actions": (0, 1), "hindsight_gamma": 0.5})
        self.agent.initialize_networks(1, 1)

    def test_hindsight_offsets_match_sampled_future_transitions(self):
        batch = [
            {"episode_id": episode, "next_state": torch.tensor([goal])}
            for episode, goal in [(0, 1.), (0, 2.), (0, 3.), (1, 4.)]
        ]

        with patch("torch.multinomial", side_effect=lambda weights, **_: torch.tensor([len(weights) - 1])):
            goals, offsets = self.agent._sample_hindsight_goals(batch)

        self.assertEqual(goals.flatten().tolist(), [3., 3., 3., 4.])
        self.assertEqual(offsets.tolist(), [2, 1, 0, 0])

    def test_bidirectional_loss_discounts_positives_and_uses_shortest_duplicate_offset(self):
        batch = [
            {"state": torch.tensor([state]), "action": action, "episode_id": 0}
            for state, action in [(0., 0), (0., 0), (1., 1), (0., 0), (2., 0)]
        ]
        goals = torch.tensor([[1.], [1.], [1.], [2.], [3.]])
        offsets = torch.tensor([3, 1, 2, 0, 2])
        reference = next(self.agent.sa_encoder.parameters())
        unique_goals = torch.tensor([[1.], [2.], [3.]]).to(reference)
        unique_sa = torch.tensor([[0., 1., 0.], [1., 0., 1.], [2., 1., 0.]]).to(reference)
        with torch.no_grad():
            scores = -torch.cdist(self.agent.goal_encoder(unique_goals), self.agent.sa_encoder(unique_sa))
            goal_to_sa = scores.log_softmax(dim=1)
            sa_to_goal = scores.T.log_softmax(dim=1)
            expected_goal_to_sa = -((0.5 * goal_to_sa[0, 0] + 0.25 * goal_to_sa[0, 1]) / 2
                                    + goal_to_sa[1, 0] + 0.25 * goal_to_sa[2, 2]) / 3
            expected_sa_to_goal = -((0.5 * sa_to_goal[0, 0] + sa_to_goal[0, 1]) / 2
                                    + 0.25 * sa_to_goal[1, 0] + 0.25 * sa_to_goal[2, 2]) / 3

        with patch.object(self.agent, "_sample_hindsight_goals", return_value=(goals, offsets)):
            self.assertTrue(self.agent._update_contrastive(batch))

        expected = expected_goal_to_sa + expected_sa_to_goal
        self.assertAlmostEqual(self.agent.contrastive_loss, expected.item(), places=6)
