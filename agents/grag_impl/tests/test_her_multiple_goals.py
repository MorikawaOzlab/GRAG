import unittest
from unittest.mock import patch

import torch

from agents.grag_impl.grag import GRAG


class MultipleHindsightGoalsTests(unittest.TestCase):
    def test_samples_up_to_four_future_transitions_in_same_episode(self):
        agent = GRAG({"seed": 1, "hindsight_gamma": 0.5})
        agent.initialize_networks(1, 1)
        batch = [{"episode_id": 0, "next_state": torch.tensor([float(i)])}
                 for i in range(6)]
        batch += [{"episode_id": 1, "next_state": torch.tensor([10.])}]
        goals, owners = agent._sample_hindsight_goals(batch)
        for index in range(len(batch)):
            selected = goals[owners == index, 0].tolist()
            self.assertEqual(len(selected), min(4, 6 - index) if index < 6 else 1)
            self.assertEqual(len(selected), len(set(selected)))
            self.assertTrue(all(index <= goal < 6 for goal in selected) if index < 6
                            else selected == [10.])

    def test_grouped_goals_use_all_positive_state_actions(self):
        agent = GRAG({"seed": 1, "actions": (0, 1)})
        agent.initialize_networks(1, 1)
        batch = [{"episode_id": episode, "state": torch.tensor([float(i)]),
                  "action": i % 2, "next_state": torch.tensor([float(i + 1)])}
                 for i, episode in enumerate((0, 0, 1))]
        batch.append({**batch[0], "episode_id": 2})  # Same (s, a) in another transition.
        goals = torch.tensor([[1.], [2.], [2.], [3.], [1.]])
        owners = torch.tensor([0, 0, 1, 2, 3])
        reference = next(agent.sa_encoder.parameters())
        inputs = torch.tensor([[0., 1., 0.], [1., 0., 1.], [2., 1., 0.]]).to(reference)
        with torch.no_grad():
            logits = agent.goal_encoder(torch.tensor([[1.], [2.], [3.]]).to(reference)) @ agent.sa_encoder(inputs).T
            expected = []
            for row, positive in enumerate(((0,), (0, 1), (2,))):
                log_prob = torch.log_softmax(logits[row], dim=0)
                expected.append(-log_prob[list(positive)].mean())
            expected_loss = torch.stack(expected).mean().item()
        with patch.object(agent, "_sample_hindsight_goals", return_value=(goals, owners)):
            self.assertTrue(agent._update_contrastive(batch))
        self.assertAlmostEqual(agent.contrastive_loss, expected_loss, places=5)

    def test_skips_goals_without_negatives_and_updates_other_goals(self):
        agent = GRAG({"seed": 1, "actions": (0, 1)})
        agent.initialize_networks(1, 1)
        batch = [{"episode_id": 0, "state": torch.tensor([0.]), "action": 0,
                  "next_state": torch.tensor([1.])},
                 {"episode_id": 0, "state": torch.tensor([1.]), "action": 1,
                  "next_state": torch.tensor([2.])}]
        # Both transitions reach goal 2, so only goal 1 can be contrasted.
        with patch.object(agent.goal_encoder, "forward", wraps=agent.goal_encoder.forward) as encode:
            self.assertTrue(agent._update_contrastive(batch))
        self.assertEqual(encode.call_args.args[0].flatten().tolist(), [1.])

    def test_one_goal_waits_for_a_negative_group(self):
        agent = GRAG({"seed": 1, "actions": (0, 1)})
        agent.initialize_networks(1, 1)
        batch = [{"episode_id": 0, "state": torch.tensor([0.]), "action": 0,
                  "next_state": torch.tensor([1.])}]
        self.assertFalse(agent._update_contrastive(batch))
        self.assertEqual(len(agent._contrastive_pending), 1)
        second = [{"episode_id": 0, "state": torch.tensor([1.]), "action": 1,
                   "next_state": torch.tensor([2.])}]
        self.assertTrue(agent._update_contrastive(second))
        self.assertEqual(agent._contrastive_pending, [])


if __name__ == "__main__":
    unittest.main()
