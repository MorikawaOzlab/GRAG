"""Paper objective, episode replay, gradients and reproducible resume."""

import base64
import io
import math
import unittest
from unittest.mock import patch

import torch

from agents.grag_impl.grag_sgcrl import SGCRL, GridWorldSGCRL


class SGCRLLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def agent(self, *, discrete=True, **settings):
        agent = SGCRL({"seed": 7, "crl_hidden_size": 16, "crl_batch_size": 8,
                       "crl_min_replay_size": 2, "crl_updates_per_step": 0,
                       "crl_device": "cpu", **settings})
        agent.initialize_networks(2, 2, 3 if discrete else 1, discrete=discrete)
        return agent

    def episode(self, agent, offset=0, length=4):
        for t in range(length):
            agent.learn(torch.tensor([offset, t]),
                        torch.tensor(t % 3) if agent.actor.discrete else torch.tensor([0.2]),
                        torch.tensor([offset, t + 1]), torch.tensor([offset, 99]),
                        truncated=t == length - 1)

    def test_critic_matches_explicit_infonce_and_lse_formula(self):
        agent = self.agent()
        states = torch.tensor([[0., 0.], [0., 1.], [1., 0.]])
        actions = torch.tensor([0, 1, 2])
        goals = torch.tensor([[0., 1.], [1., 0.], [1., 1.]])
        logits = agent.state_goal_scores(states, actions, goals)
        expected = sum(-logits[i, i] + torch.log(torch.exp(logits[i]).sum())
                       + 0.01 * torch.log(torch.exp(logits[i]).sum()) ** 2
                       for i in range(3)) / 3
        self.assertTrue(torch.allclose(agent.critic_loss(states, actions, goals), expected))

    def test_grid_policy_cache_updates_and_checkpoint_preserves_action_sequence(self):
        agent = GridWorldSGCRL({'seed': 7, 'crl_device': 'cpu', 'crl_hidden_size': 16,
                                'crl_min_replay_size': 2, 'crl_updates_per_step': 0})
        obs = {'rows': 2, 'cols': 2, 'obstacles': [], 'goal': [1, 1], 'position': [0, 0]}
        agent._features(obs)
        agent.core._steps = 2
        with patch.object(agent.core, 'action_probabilities', wraps=agent.core.action_probabilities) as forward:
            first = agent.action_probabilities(obs)
            agent.choose_action(obs)
            self.assertEqual(forward.call_count, 1)
            agent.core._updates += 1
            self.assertEqual(agent.action_probabilities(obs), first)
            self.assertEqual(forward.call_count, 2)
        restored = GridWorldSGCRL({'crl_device': 'cpu'})
        restored.load_checkpoint(agent.checkpoint())
        self.assertEqual([agent.choose_action(obs) for _ in range(30)],
                         [restored.choose_action(obs) for _ in range(30)])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_graph_matches_eager_updates_and_checkpoint(self):
        from collections import deque
        episodes = deque([(torch.randn(40, 12), torch.randint(4, (40,)), torch.randn(40, 12))])
        def build(graph):
            agent = SGCRL({'seed': 7, 'crl_device': 'cuda', 'crl_hidden_size': 16,
                           'crl_repr_dim': 8, 'crl_batch_size': 8, 'crl_cuda_graph': graph})
            agent.initialize_networks(12, 12, 4, discrete=True)
            agent._episodes, agent._replay_size = episodes, 40
            return agent
        eager, graph = build(False), build(True)
        for _ in range(20):
            eager.update()
            graph.update()
        self.assertIsNotNone(graph._cuda_graph, graph._cuda_graph_error)
        for left, right in zip(
                list(eager.actor.parameters()) + list(eager.sa_encoder.parameters()) +
                list(eager.goal_encoder.parameters()) + [eager.log_alpha],
                list(graph.actor.parameters()) + list(graph.sa_encoder.parameters()) +
                list(graph.goal_encoder.parameters()) + [graph.log_alpha]):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
        restored = SGCRL({'crl_device': 'cuda'})
        restored.load_checkpoint(graph.checkpoint())
        self.assertIsNone(restored._cuda_graph)
        graph.update()
        restored.update()
        for left, right in zip(graph.actor.parameters(), restored.actor.parameters()):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_graph_rebuilds_after_replay_changes(self):
        from collections import deque
        episode = (torch.randn(40, 12), torch.randint(4, (40,)), torch.randn(40, 12))
        agents = []
        for enabled in (False, True):
            agent = SGCRL({'seed': 13, 'crl_device': 'cuda', 'crl_hidden_size': 16,
                           'crl_repr_dim': 8, 'crl_batch_size': 8,
                           'crl_updates_per_step': 0, 'crl_min_replay_size': 2,
                           'crl_cuda_graph': enabled})
            agent.initialize_networks(12, 12, 4, discrete=True)
            agent._episodes, agent._replay_size = deque([episode]), 40
            agents.append(agent)
        eager, graph = agents
        for _ in range(8):
            eager.update()
            graph.update()
        self.assertTrue(graph._graph_inline_sampling)
        for agent in agents:
            agent._episode = [(torch.ones(12), torch.tensor(1), torch.ones(12))]
            agent.end_episode()
        self.assertIsNone(graph._cuda_graph)
        for _ in range(8):
            eager.update()
            graph.update()
        self.assertTrue(graph._graph_inline_sampling)
        for left, right in zip(eager.actor.parameters(), graph.actor.parameters()):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)

    def test_future_sampling_stays_in_episode_and_has_discounted_distribution(self):
        agent = self.agent(crl_gamma=0.5, crl_batch_size=8000)
        self.episode(agent, offset=10)
        self.episode(agent, offset=20)
        states, _, goals = agent.sample_batch()
        self.assertTrue(torch.equal(states[:, 0], goals[:, 0]))
        self.assertTrue((goals[:, 1] > states[:, 1]).all())
        self.assertTrue((goals[:, 1] <= 4).all())
        selected = goals[states[:, 1] == 0, 1]
        for future, probability in enumerate((8/15, 4/15, 2/15, 1/15), 1):
            self.assertAlmostEqual((selected == future).float().mean().item(), probability, delta=0.04)
        immediate = self.agent(crl_gamma=0, crl_batch_size=100)
        self.episode(immediate)
        states, _, goals = immediate.sample_batch()
        self.assertTrue(torch.equal(goals[:, 1], states[:, 1] + 1))

    def test_updates_train_all_networks_for_discrete_and_continuous_actions(self):
        for discrete in (True, False):
            with self.subTest(discrete=discrete):
                agent = self.agent(discrete=discrete)
                self.episode(agent)
                before = [[p.detach().clone() for p in net.parameters()]
                          for net in (agent.actor, agent.sa_encoder, agent.goal_encoder)]
                metrics = agent.update()
                self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
                for net, previous in zip((agent.actor, agent.sa_encoder, agent.goal_encoder), before):
                    self.assertTrue(any(not torch.equal(p, old) for p, old in zip(net.parameters(), previous)))
                self.assertTrue(all(p.grad is None for p in agent._encoder_parameters()))
                self.assertLess(agent.log_alpha.item() * metrics["entropy"], 0)

    def test_discrete_actor_matches_exact_expectation_and_only_trains_actor(self):
        agent = self.agent(crl_random_goals=0)
        states, goals = torch.tensor([[0., 1.], [1., 0.]]), torch.tensor([[1., 1.], [2., 1.]])
        distribution = agent.actor(torch.cat((states, goals), -1))
        with torch.no_grad():
            q = torch.stack([(agent.encode_state_actions(states, torch.full((2,), action))
                              * agent.encode_goals(goals)).sum(-1) for action in range(3)], -1)
        expected = -(distribution.probs * q).sum(-1).mean() - distribution.entropy().mean()
        loss, _ = agent.actor_loss(states, goals)
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in agent._encoder_parameters()))

    def test_replay_capacity_and_idempotent_end_episode(self):
        agent = self.agent(crl_max_replay_size=5)
        self.episode(agent, offset=10)
        self.episode(agent, offset=20)
        self.assertEqual(agent._replay_size, 5)
        self.assertEqual([len(ep[0]) for ep in agent._episodes], [1, 4])
        agent.end_episode()
        self.assertEqual(agent._replay_size, 5)

    def test_replay_cache_limit_and_invalidation(self):
        cached = self.agent()
        self.episode(cached, offset=10)
        cached.sample_batch()
        self.assertIsNotNone(cached._replay_cache)
        self.episode(cached, offset=20)
        self.assertIsNone(cached._replay_cache)
        cached.sample_batch()
        self.assertEqual(len(cached._replay_cache[3]), 2)
        uncached = self.agent(crl_replay_cache_mb=0)
        self.episode(uncached)
        uncached.sample_batch()
        self.assertIsNone(uncached._replay_cache)

    def test_vectorized_sampling_is_uniform_over_episodes_not_transitions(self):
        agent = self.agent(crl_batch_size=8000)
        self.episode(agent, offset=10, length=1)
        self.episode(agent, offset=20, length=9)
        states, _, _ = agent.sample_batch()
        fraction = (states[:, 0] == 10).float().mean().item()
        self.assertAlmostEqual(fraction, 0.5, delta=0.03)

    def test_random_warmup_skips_unused_actor_forward(self):
        agent = self.agent()
        with patch.object(agent.actor, 'forward', side_effect=AssertionError('unused inference')):
            action = agent.choose_action(torch.zeros(2), torch.ones(2))
            self.assertTrue(0 <= action.item() < 3)
            with self.assertRaises(AssertionError):
                agent.choose_action(torch.zeros(2), torch.ones(2), greedy=True)

    def test_shared_actor_features_match_full_action_computation_and_gradients(self):
        agent = self.agent(crl_hidden_size=32)
        states = torch.tensor([[0., 1.], [1., 0.], [1., 1.]])
        goals = torch.tensor([[1., 1.], [2., 1.], [0., 0.]])
        for random_goals in (0., 0.5, 1.):
            with self.subTest(random_goals=random_goals):
                agent.random_goals = random_goals
                agent.share_actor_features = False
                reference, _ = agent.actor_loss(states, goals)
                reference_gradients = torch.autograd.grad(reference, tuple(agent.actor.parameters()))
                agent.share_actor_features = True
                optimized, _ = agent.actor_loss(states, goals)
                optimized_gradients = torch.autograd.grad(optimized, tuple(agent.actor.parameters()))
                self.assertTrue(torch.allclose(reference, optimized, atol=1e-6))
                for expected, actual in zip(reference_gradients, optimized_gradients):
                    self.assertTrue(torch.allclose(expected, actual, atol=1e-6))

    def test_custom_encoders_use_generic_action_computation(self):
        agent = self.agent()
        agent.sa_encoder = torch.nn.Linear(5, 64)
        agent.goal_encoder = torch.nn.Linear(2, 64)
        self.assertFalse(agent._can_share_actor_features())
        states = torch.ones(3, 2)
        phi = agent._all_action_representations(states)
        self.assertEqual(phi.shape, (3, 3, 64))

    def test_checkpoint_restores_replay_pending_episode_optimizers_and_rng(self):
        agent = self.agent()
        self.episode(agent)
        agent.update()
        agent.learn(torch.zeros(2), torch.tensor(1), torch.ones(2), torch.ones(2))
        restored = SGCRL.from_export(agent.checkpoint())
        self.assertEqual(len(restored._episode), 1)
        self.assertEqual(restored._replay_size, agent._replay_size)
        for a, b in zip(agent.sample_batch(), restored.sample_batch()):
            self.assertTrue(torch.equal(a, b))
        for current in (agent, restored):
            current.end_episode()
            current.update()
        self.assertEqual(agent.metrics, restored.metrics)
        for a, b in zip(agent.actor.parameters(), restored.actor.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_adapter_ignores_rewards_and_enforces_single_collection_goal(self):
        observation = {"rows": 2, "cols": 3, "goal": [1, 2], "position": [0, 0], "obstacles": []}
        agents = [GridWorldSGCRL({"seed": 7, "crl_min_replay_size": 2,
                                 "crl_batch_size": 4, "crl_hidden_size": 16,
                                 "crl_device": "cpu"}) for _ in range(2)]
        for agent, reward in zip(agents, (-999, 999)):
            agent.choose_action(observation)
            agent.learn(observation, "right", reward,
                        {**observation, "position": [0, 1]}, False)
            agent.end_episode()
            agent.learn(observation, "down", reward,
                        {**observation, "position": [1, 0]}, False)
            agent.end_episode()
        self.assertEqual(agents[0].metrics, agents[1].metrics)
        for a, b in zip(agents[0].core.actor.parameters(), agents[1].core.actor.parameters()):
            self.assertTrue(torch.equal(a, b))
        with self.assertRaises(ValueError):
            agents[0].choose_action({**observation, "goal": [1, 1]})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_checkpoint_migrates_cpu_cuda_rng_for_new_and_legacy_formats(self):
        for source_device, target_device in (("cpu", "cuda"), ("cuda", "cpu")):
            for legacy in (False, True):
                with self.subTest(source=source_device, target=target_device, legacy=legacy):
                    source = self.agent(crl_device=source_device)
                    self.episode(source)
                    source.update()
                    saved = source.checkpoint()
                    payload = torch.load(io.BytesIO(base64.b64decode(saved["data"])),
                                         map_location="cpu", weights_only=True)
                    if legacy:
                        payload.pop("action_random_device")
                    payload["settings"]["crl_device"] = target_device
                    stream = io.BytesIO()
                    torch.save(payload, stream)
                    saved["data"] = base64.b64encode(stream.getvalue()).decode("ascii")
                    restored = SGCRL.from_export(saved)
                    self.assertEqual(restored.device.type, target_device)
                    self.assertEqual(restored._replay_size, source._replay_size)
                    self.assertEqual(restored._updates, source._updates)
                    for a, b in zip(source.actor.parameters(), restored.actor.parameters()):
                        self.assertTrue(torch.equal(a.detach().cpu(), b.detach().cpu()))
                    for a, b in zip(source.actor_optimizer.state.values(),
                                    restored.actor_optimizer.state.values()):
                        self.assertTrue(torch.equal(a["exp_avg"].cpu(), b["exp_avg"].cpu()))
                    metrics = restored.update()
                    self.assertTrue(all(math.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
