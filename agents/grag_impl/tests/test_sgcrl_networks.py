"""SGCRL initialization, reproducibility and contrastive gradient checks."""

import unittest

import torch

from agents.sgcrl import SGCRL


class SGCRLNetworkTests(unittest.TestCase):
    def test_forward_and_contrastive_gradients(self):
        agent = SGCRL({"seed": 7, "crl_device": "cpu"})
        agent.initialize_networks(state_dim=11, goal_dim=3, action_dim=2)
        self.assertEqual(agent.actor(torch.zeros(14)).sample().shape, (2,))
        policy = agent.actor(torch.zeros(5, 14))
        actions = policy.rsample()
        self.assertEqual(actions.shape, (5, 2))
        self.assertTrue((actions.abs() < 1).all())
        log_probs = policy.log_prob(actions)
        self.assertEqual(log_probs.shape, (5,))
        self.assertTrue(torch.isfinite(log_probs).all())
        (-log_probs.mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in agent.actor.parameters()))
        agent.actor.zero_grad(set_to_none=True)
        phi = agent.sa_encoder(torch.ones(5, 13))
        psi = agent.goal_encoder(torch.ones(5, 3))
        self.assertEqual(phi.shape, (5, 64))
        self.assertEqual(psi.shape, (5, 64))
        (phi @ psi.T).sum().backward()
        for encoder in (agent.sa_encoder, agent.goal_encoder):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in agent.actor.parameters()))

    def test_seed_preserves_global_rng_and_reinitialization_preserves_weights(self):
        before = torch.random.get_rng_state().clone()
        first = SGCRL({"seed": 7, "crl_device": "cpu"})
        second = SGCRL({"seed": 7, "crl_device": "cpu"})
        for agent in (first, second):
            agent.initialize_networks(11, 3, 2)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for name in ("actor", "sa_encoder", "goal_encoder"):
            for a, b in zip(getattr(first, name).parameters(), getattr(second, name).parameters()):
                self.assertTrue(torch.equal(a, b))
        actor = first.actor
        with torch.no_grad():
            next(actor.parameters()).fill_(0.123)
        first.initialize_networks(11, 3, 2)
        self.assertIs(first.actor, actor)
        self.assertTrue(torch.all(next(actor.parameters()) == 0.123))
        with self.assertRaises(ValueError):
            first.initialize_networks(12, 3, 2)
        with self.assertRaises(ValueError):
            first.initialize_networks(11, 3, 2, discrete=True)

    def test_invalid_dimensions_can_be_corrected_before_initialization(self):
        agent = SGCRL({"seed": 7, "crl_device": "cpu"})
        for size in (0, -1, True, 2.5):
            for dims in ((size, 3, 2), (11, size, 2), (11, 3, size)):
                with self.assertRaises(ValueError):
                    agent.initialize_networks(*dims)
        self.assertIsNone(agent.actor)
        agent.initialize_networks(11, 3, 2)
        self.assertIsNotNone(agent.actor)

    def test_discrete_policy_supports_arbitrary_action_count_and_batch_dimensions(self):
        agent = SGCRL({"seed": 7, "crl_device": "cpu"})
        agent.initialize_networks(8, 5, 7, discrete=True)
        policy = agent.actor(torch.zeros(2, 3, 13))
        self.assertEqual(policy.probs.shape, (2, 3, 7))
        self.assertTrue(torch.allclose(policy.probs.sum(-1), torch.ones(2, 3)))
        actions = policy.sample()
        self.assertEqual(actions.shape, (2, 3))
        one_hot = torch.nn.functional.one_hot(actions, 7).float()
        phi = agent.sa_encoder(torch.cat((torch.zeros(2, 3, 8), one_hot), -1))
        self.assertEqual(phi.shape, (2, 3, 64))
        self.assertTrue(torch.isfinite(policy.log_prob(actions)).all())

    def test_greedy_discrete_actions_are_deterministic_without_side_effects(self):
        agent = SGCRL({"seed": 7, "crl_device": "cpu"})
        agent.initialize_networks(8, 5, 7, discrete=True)
        with torch.no_grad():
            agent.actor.network[-1].weight.zero_()
            agent.actor.network[-1].bias.copy_(torch.tensor([0., 1., -2., 5., 3., 2., 4.]))
        state = torch.zeros(2, 3, 8, requires_grad=True)
        goal = torch.zeros(2, 3, 5, requires_grad=True)
        before = torch.random.get_rng_state().clone()
        actions = agent.choose_action(state, goal, greedy=True)
        self.assertTrue(torch.equal(actions, torch.full((2, 3), 3)))
        self.assertEqual(actions.dtype, torch.long)
        self.assertFalse(actions.requires_grad)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in agent.actor.parameters()))
        self.assertEqual(agent.choose_action(state[0, 0], goal[0, 0], greedy=True).shape, ())
        sampled = agent.choose_action(state, goal)
        self.assertEqual(sampled.shape, (2, 3))
        self.assertTrue(((0 <= sampled) & (sampled < 7)).all())

    def test_greedy_continuous_actions_use_squashed_mean(self):
        agent = SGCRL({"seed": 7, "crl_device": "cpu"})
        agent.initialize_networks(8, 5, 2)
        agent.actor.double()
        with torch.no_grad():
            agent.actor.network[-1].weight.zero_()
            agent.actor.network[-1].bias.copy_(torch.tensor([0.5, -1., 2., 3.]))
        state = torch.zeros(3, 8, requires_grad=True)
        goal = torch.zeros(3, 5, requires_grad=True)
        before = torch.random.get_rng_state().clone()
        actions = agent.choose_action(state, goal, greedy=True)
        expected = torch.tensor([0.5, -1.], dtype=torch.float64).tanh().expand(3, 2)
        self.assertTrue(torch.allclose(actions, expected))
        self.assertEqual(actions.dtype, torch.float64)
        self.assertFalse(actions.requires_grad)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(agent.choose_action(state[0], goal[0], greedy=True).shape, (2,))
        sampled = agent.choose_action(state, goal)
        self.assertEqual(sampled.shape, (3, 2))
        self.assertTrue((sampled.abs() <= 1).all())

    def test_choose_action_requires_initialization(self):
        with self.assertRaisesRegex(RuntimeError, "initialize_networks"):
            SGCRL({"crl_device": "cpu"}).choose_action(torch.zeros(8), torch.zeros(5), greedy=True)


if __name__ == "__main__":
    unittest.main()
