import unittest
from unittest.mock import patch

import torch

from agents.grag_impl.grag import GRAG
from agents.grag_impl.gridworld_adapter import GridWorldAdapter


class ContrastiveUpdateTests(unittest.TestCase):
    def test_hindsight_samples_future_within_episode(self):
        agent = GRAG({"seed": 1, "hindsight_gamma": 0.5})
        agent.initialize_networks(1, 1)
        transitions = [
            {"episode_id": 0, "next_state": torch.tensor([1.])},
            {"episode_id": 0, "next_state": torch.tensor([2.])},
            {"episode_id": 1, "next_state": torch.tensor([10.])},
            {"episode_id": 1, "next_state": torch.tensor([11.])},
        ]
        weights_seen = []

        def choose_last(weights, num_samples, generator):
            weights_seen.append(weights.clone())
            return torch.tensor([len(weights) - 1])

        with patch("torch.multinomial", side_effect=choose_last):
            goals = agent._sample_hindsight_goals(transitions)
        self.assertEqual(goals.flatten().tolist(), [2., 2., 11., 11.])
        self.assertTrue(torch.equal(weights_seen[0], torch.tensor([1., 0.5])))
        self.assertTrue(torch.equal(weights_seen[1], torch.tensor([1.])))

    def test_updates_encoders_when_another_episode_supplies_negatives(self):
        agent = GRAG({"seed": 42})
        agent.UPDATE_INTERVAL = 2
        agent.initialize_networks(problem=GridWorldAdapter({"rows": 2, "cols": 2}))

        def observation(position):
            return {"rows": 2, "cols": 2, "obstacles": [],
                    "goal": [1, 1], "position": position}

        def step(start, end):
            obs = observation(start)
            action = agent.choose_action(obs)
            agent.learn(obs, action, 0, observation(end), False)

        before_sa = [p.detach().clone() for p in agent.sa_encoder.parameters()]
        before_goal = [p.detach().clone() for p in agent.goal_encoder.parameters()]
        before_actor = [p.detach().clone() for p in agent.actor.parameters()]
        with patch.object(agent, "_update_ppo"):
            step([0, 0], [0, 1])
            step([0, 1], [1, 1])
            self.assertIsNone(agent.contrastive_loss)
            self.assertEqual(len(agent._contrastive_pending), 2)
            agent._update_ppo.assert_not_called()
            agent.end_episode()
            step([1, 0], [0, 0])
            step([0, 0], [1, 0])
            agent._update_ppo.assert_called_once()

        self.assertTrue(torch.isfinite(torch.tensor(agent.contrastive_loss)))
        self.assertEqual(agent._contrastive_pending, [])
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_sa, agent.sa_encoder.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_goal, agent.goal_encoder.parameters())))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before_actor, agent.actor.parameters())))


if __name__ == "__main__":
    unittest.main()
