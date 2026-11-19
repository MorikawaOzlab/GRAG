import unittest
from unittest.mock import patch

import torch
from torch import nn

from agents.grag_impl.grag import GRAG
from agents.grag_impl.gridworld_adapter import GridWorldAdapter


class StateActionEncoder(nn.Module):
    def forward(self, state_action):
        # The second action contributes one to the representation.
        return state_action[:, -1:]


class NegativeGoalEncoder(nn.Module):
    def forward(self, goal):
        return -torch.ones_like(goal)


class ContrastivePPOTests(unittest.TestCase):
    def test_advantages_are_computed_before_any_minibatch_update(self):
        agent = GRAG({"seed": 7, "grag_update_interval": 4, "cppo_batch_size": 2,
                      "target_goal_encoder": lambda obs: [0.]})
        agent.initialize_networks(1, 1)
        for i in range(4):
            action = agent.choose_action([float(i)])
            agent.learn([float(i)], action, 0, [float(i + 1)], False)
            if i == 1:
                agent.end_episode()
        real_q = agent.q_values
        real_step = agent.optimizer.step
        stepped = []
        def q_before_update(*args):
            self.assertEqual(stepped, [])
            return real_q(*args)
        def step(*args, **kwargs):
            stepped.append(True)
            return real_step(*args, **kwargs)
        with patch.object(agent, "q_values", side_effect=q_before_update), \
             patch.object(agent.optimizer, "step", side_effect=step):
            agent.end_episode()
        self.assertEqual(len(stepped), 2)
        self.assertEqual(agent.updates, 1)
    def test_q_values_are_negative_euclidean_distances(self):
        agent = GRAG({"seed": 2, "actions": (0, 1)})
        agent.initialize_networks(1, 1)
        agent.sa_encoder = StateActionEncoder()
        agent.goal_encoder = NegativeGoalEncoder()
        values = agent.q_values(torch.tensor([[1.], [2.]]), torch.tensor([[0.], [3.]]))
        torch.testing.assert_close(values, torch.tensor([[-1., -2.], [-1., -2.]]))

    def test_q_distance_uses_representation_magnitude_and_zero_vectors_are_finite(self):
        class StateAction(nn.Module):
            def forward(self, x):
                return torch.stack((x[:, 0] * x[:, 1], x[:, 0] * x[:, 2]), dim=-1)

        class Goal(nn.Module):
            def forward(self, x):
                return torch.cat((x, x), dim=-1)

        agent = GRAG({"seed": 2, "actions": (0, 1)})
        agent.initialize_networks(1, 1)
        agent.sa_encoder, agent.goal_encoder = StateAction(), Goal()
        q = agent.q_values(torch.tensor([[2.], [8.], [0.]]), torch.tensor([[3.], [9.], [0.]]))
        expected = torch.tensor([[-10 ** 0.5, -10 ** 0.5], [-82 ** 0.5, -82 ** 0.5], [0., 0.]])
        torch.testing.assert_close(q, expected)
        self.assertTrue(torch.isfinite(q).all())

    def test_q_advantage_updates_actor_without_updating_encoders(self):
        agent = GRAG({"seed": 2, "actions": (0, 1),
                      "target_goal_encoder": lambda obs: [0.]})
        agent.initialize_networks(1, 1)
        agent.sa_encoder = StateActionEncoder()
        agent.goal_encoder = NegativeGoalEncoder()
        observation = [1.]
        probability_before = agent.action_probabilities(observation)[0]
        transitions = []
        for action in (0, 1) * 8:
            transitions.append({"state": torch.tensor(observation), "observation": observation,
                                "action": action,
                                "log_prob": torch.tensor(agent.action_probabilities(observation)[action]).log().item()})
        agent._update_ppo(transitions)
        self.assertGreater(agent.action_probabilities(observation)[0], probability_before)
        self.assertTrue(torch.isfinite(torch.tensor(agent.policy_loss)))

    def test_goal_dimension_is_checked(self):
        agent = GRAG({"seed": 1, "target_goal_encoder": lambda obs: [1, 2]})
        agent.initialize_networks(1, 1)
        row = {"state": torch.tensor([1.]), "observation": [1.], "action": 0, "log_prob": -1.}
        with self.assertRaises(ValueError):
            agent._update_ppo([row])

    def test_gridworld_updates_at_128_steps(self):
        agent = GRAG({"seed": 7, "grag_update_interval": 128})
        agent.initialize_networks(problem=GridWorldAdapter({"rows": 2, "cols": 2}))
        actor_before = [p.detach().clone() for p in agent.actor.parameters()]
        for step in range(128):
            obs = {"rows": 2, "cols": 2, "obstacles": [], "goal": [1, 1],
                   "position": [step % 2, (step // 2) % 2]}
            next_obs = {**obs, "position": [(step + 1) % 2, (step // 2) % 2]}
            action = agent.choose_action(obs)
            agent.learn(obs, action, 0, next_obs, False)
            if step == 63:
                agent.end_episode()
        agent.end_episode()
        self.assertIsNotNone(agent.contrastive_loss)
        self.assertIsNotNone(agent.policy_loss)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters())))
        self.assertEqual(agent._update_batch, [])


if __name__ == "__main__":
    unittest.main()
