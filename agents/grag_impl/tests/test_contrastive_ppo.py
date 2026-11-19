import unittest

import torch
from torch import nn

from agents.grag_impl.grag import GRAG
from agents.grag_impl.gridworld_adapter import GridWorldAdapter


class StateActionEncoder(nn.Module):
    def forward(self, state_action):
        # The first action is closer to the zero goal.
        return state_action[:, -1:]


class ZeroGoalEncoder(nn.Module):
    def forward(self, goal):
        return torch.zeros_like(goal)


class ContrastivePPOTests(unittest.TestCase):
    def test_q_advantage_updates_actor_without_updating_encoders(self):
        agent = GRAG({"seed": 2, "actions": (0, 1),
                      "target_goal_encoder": lambda obs: [0.]})
        agent.initialize_networks(1, 1)
        agent.sa_encoder = StateActionEncoder()
        agent.goal_encoder = ZeroGoalEncoder()
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
        self.assertIsNotNone(agent.contrastive_loss)
        self.assertIsNotNone(agent.policy_loss)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters())))
        self.assertEqual(agent._update_batch, [])


if __name__ == "__main__":
    unittest.main()
