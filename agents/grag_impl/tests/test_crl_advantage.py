"""CRL inference reuse must preserve values, gradients and parameter updates."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from agents.grag_impl.grag import ACTIONS, GRAG, state_action_encoder


class CRLAdvantageTests(unittest.TestCase):
    def setUp(self):
        self.agent = GRAG({"seed": 7})
        with torch.random.fork_rng():
            torch.manual_seed(7)
            self.agent.actor = nn.Sequential(nn.Flatten(start_dim=-3), nn.Linear(12, 4))
            self.agent.sa_encoder = nn.Sequential(nn.Linear(12, 8), nn.SiLU(), nn.Linear(8, 3))
            self.agent.goal_encoder = nn.Sequential(
                nn.Linear(384, 8), nn.LayerNorm(8), nn.SiLU(), nn.Linear(8, 3))
            self.goal = torch.randn(384)
        self.observation = dict(rows=2, cols=2, obstacles=[], goal=[1, 1], position=[0, 0])

    def reference_score(self, features, goal):
        sa = self.agent._encode_crl_features(self.agent.sa_encoder, features)
        g = self.agent._encode_crl_features(self.agent.goal_encoder, goal)
        return -(sa - g).norm(dim=-1)

    def reference_advantage(self, observation):
        state = self.agent.observation_to_tensor(observation)
        states = state.unsqueeze(0).expand(4, -1, -1, -1)
        features = state_action_encoder(states, torch.arange(4))
        scores = self.reference_score(features, self.goal)
        probs = self.agent.actor(state).softmax(-1)
        return scores - (scores * probs).sum()

    def test_advantage_matches_reference_and_reuses_goal_across_positions(self):
        goal_forwards = []
        handle = self.agent.goal_encoder.register_forward_hook(
            lambda *args: goal_forwards.append(True))
        try:
            with patch("agents.grag_impl.grag.encode_language", return_value=self.goal) as encode:
                results = [self.agent._compute_crl_advantage(self.observation)]
                for index, action in enumerate(ACTIONS):
                    torch.testing.assert_close(
                        self.agent._compute_crl_advantage(self.observation, action), results[0][index])
                moved = {**self.observation, "position": [0, 1]}
                results.append(self.agent._compute_crl_advantage(moved))
                self.assertEqual(encode.call_count, 1)
                self.assertEqual(len(goal_forwards), 1)
        finally:
            handle.remove()
        with torch.no_grad():
            for result, observation in zip(results, (self.observation, moved)):
                torch.testing.assert_close(result, self.reference_advantage(observation))
                self.assertFalse(result.requires_grad)

    def test_cache_invalidates_on_input_weight_load_and_dtype_changes(self):
        features = torch.zeros(4, 12)
        with torch.no_grad():
            self.agent._compute_crl_score(features, self.goal)
            self.goal.add_(0.5)
            torch.testing.assert_close(
                self.agent._compute_crl_score(features, self.goal),
                self.reference_score(features, self.goal))
            self.agent.goal_encoder[0].weight.mul_(1.2)
            torch.testing.assert_close(
                self.agent._compute_crl_score(features, self.goal),
                self.reference_score(features, self.goal))
            state = {name: value + 0.1 for name, value in self.agent.goal_encoder.state_dict().items()}
            self.agent.goal_encoder.load_state_dict(state)
            torch.testing.assert_close(
                self.agent._compute_crl_score(features, self.goal),
                self.reference_score(features, self.goal))
            self.agent.goal_encoder.double()
            torch.testing.assert_close(
                self.agent._compute_crl_score(features, self.goal),
                self.reference_score(features, self.goal))

    def test_cached_inference_does_not_interrupt_training_gradients(self):
        features = torch.zeros(4, 12, requires_grad=True)
        self.goal.requires_grad_(True)
        with torch.no_grad():
            self.agent._compute_crl_score(features, self.goal)
        inputs = (features, self.goal, *self.agent.sa_encoder.parameters(),
                  *self.agent.goal_encoder.parameters())
        actual = torch.autograd.grad(self.agent._compute_crl_score(features, self.goal).sum(), inputs)
        expected = torch.autograd.grad(self.reference_score(features, self.goal).sum(), inputs)
        for gradient, reference in zip(actual, expected):
            torch.testing.assert_close(gradient, reference)

    def test_stochastic_goal_encoder_is_not_cached(self):
        self.agent.goal_encoder.insert(1, nn.Dropout())
        with patch.object(self.agent.goal_encoder, "forward", wraps=self.agent.goal_encoder.forward) as forward:
            with torch.no_grad():
                for _ in range(2):
                    self.agent._compute_crl_score(torch.zeros(4, 12), self.goal)
            self.assertEqual(forward.call_count, 2)


if __name__ == "__main__":
    unittest.main()
