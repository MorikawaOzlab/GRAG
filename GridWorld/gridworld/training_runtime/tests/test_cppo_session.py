"""CPPO algorithm and GridWorld persistence/inspection integration."""

import json
import unittest
from unittest.mock import patch

import torch

from agents.cppo import CPPOAgent
from gridworld.storage import DEFAULT_SETTINGS
from gridworld.training_runtime import TrainingSession


MAP = {"name": "CPPO integration", "rows": 2, "cols": 3, "start": [0, 0],
       "goal": [1, 2], "obstacles": [[0, 1]]}
SETTINGS = {**DEFAULT_SETTINGS, "max_steps": 8, "cppo_hidden_size": 16,
            "cppo_depth": 2, "cppo_repr_dim": 8, "cppo_rollout_steps": 12,
            "cppo_batch_size": 8}


class CPPOSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def session(self):
        return TrainingSession(MAP, SETTINGS, CPPOAgent)

    def test_json_resume_mid_episode_is_identical_across_updates(self):
        original = self.session()
        original.run_batch(4)
        original.step()
        saved = json.loads(json.dumps(original.result(checkpoint=True)))
        restored = TrainingSession.restore(saved, CPPOAgent)
        for _ in range(40):
            self.assertEqual(original.step(), restored.step())
        self.assertGreater(original.agent.updates, saved["agent_checkpoint"]["model"]["updates"])
        self.assertEqual(original.agent.checkpoint(), restored.agent.checkpoint())

    def test_old_mlp_checkpoint_is_rejected(self):
        agent = self.session().agent
        checkpoint = agent.checkpoint()
        checkpoint["model"]["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "incompatible"):
            agent.load_checkpoint(checkpoint)

    def test_binary_tensor_encoding_reduces_json_size(self):
        agent = self.session().agent
        tensor = torch.linspace(0, 1, 16384).reshape(128, 128)
        encoded = agent._encode(tensor)
        self.assertTrue(torch.equal(agent._decode(encoded), tensor))
        self.assertLess(len(json.dumps(encoded)), len(json.dumps(tensor.tolist())) / 2)

    def test_evaluation_and_inspection_preserve_pending_rollout_and_rng(self):
        session = self.session()
        session.run_batch(3)
        session.step()
        before = session.agent.checkpoint()
        policy = session.inspect_policy([0, 0])
        self.assertTrue(policy["supported"])
        self.assertAlmostEqual(sum(policy["probabilities"].values()), 1, places=6)
        with patch("torch.multinomial", wraps=torch.multinomial) as sample:
            self.assertGreater(len(session.evaluate()), 1)
            self.assertTrue(sample.called)
            sample.reset_mock()
            session.evaluate(force_greedy=True)
            sample.assert_not_called()
        self.assertEqual(before, session.agent.checkpoint())

    def test_rewards_do_not_affect_learning_and_actor_and_encoders_update(self):
        first, second = self.session(), self.session()
        observation = first.world.reset()
        first.agent.action_probabilities(observation)
        initial = {key: value.clone() for key, value in first.agent.networks.state_dict().items()}
        for episode in range(6):
            observation = first.world.reset()
            for _ in range(8):
                a = first.agent.choose_action(observation)
                self.assertEqual(a, second.agent.choose_action(observation))
                next_observation, done, _ = first.world.step(a)
                first.agent.learn(observation, a, 0., next_observation, done)
                second.agent.learn(observation, a, 12345., next_observation, done)
                observation = next_observation
                if done:
                    break
            first.agent.end_episode()
            second.agent.end_episode()
        self.assertEqual(first.agent.checkpoint(), second.agent.checkpoint())
        for prefix in ("actor.", "sa.", "goal."):
            self.assertTrue(any(not torch.equal(value, initial[key])
                                for key, value in first.agent.networks.state_dict().items()
                                if key.startswith(prefix)))
        self.assertTrue(all(torch.isfinite(p).all() for p in first.agent.networks.parameters()))

    def test_exact_discrete_advantage_and_negative_l2(self):
        agent = CPPOAgent({**SETTINGS, "cppo_normalize_advantage": False})
        states = torch.tensor([agent._state({**MAP, "position": [0, 0]})])
        with torch.no_grad():
            q = agent._all_q(states)
            self.assertTrue((q <= 0).all())
            probabilities = agent._distribution(states).probs
            advantages = torch.cat([agent._advantages(states, torch.tensor([a])) for a in range(4)])
            self.assertAlmostEqual(float((probabilities[0] * advantages).sum()), 0., places=6)
            self.assertTrue(torch.allclose(advantages, q[0] - (probabilities * q).sum()))

    def test_hindsight_respects_episode_and_rollout_boundaries(self):
        agent = CPPOAgent({**SETTINGS, "cppo_rollout_length": 4})
        # Two full windows and a partial window. Episode 1 spans a window boundary.
        episodes = torch.tensor([0, 0, 1, 1, 1, 1, 1, 2, 2, 3])
        achieved = torch.arange(len(episodes))
        for _ in range(10):
            retained, goals = agent._relabel(achieved, episodes)
            self.assertEqual(retained.tolist(), [0, 4, 8, 1, 5, 2, 6])
            self.assertTrue((goals >= retained).all())
            self.assertTrue((goals // 4 == retained // 4).all())
            self.assertTrue((episodes[retained] == episodes[goals]).all())
            self.assertEqual(goals[2].item(), 8)  # Diagonal fallback at episode end.
            self.assertEqual(goals[3].item(), 1)
        self.assertTrue((goals[retained == 2] == 3).all())

    def test_reference_critic_loss_includes_all_pairs_and_penalty(self):
        agent = self.session().agent
        sa = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        goal = torch.tensor([[1., 2.], [2., 1.]], requires_grad=True)
        logits = -((sa[:, None] - goal[None]).square().sum(-1) + 1e-6).sqrt()
        expected = (logits.logsumexp(1) - logits.diag()).mean()
        expected = expected + .1 * (logits + 1e-6).logsumexp(1).square().mean()
        actual = agent._contrastive_loss(agent._contrastive_logits(sa, goal))
        torch.testing.assert_close(actual, expected)
        for gradient, reference in zip(torch.autograd.grad(actual, (sa, goal)),
                                       torch.autograd.grad(expected, (sa, goal))):
            torch.testing.assert_close(gradient, reference)

    def test_public_energy_modes_match_pairwise_reference(self):
        sa = torch.tensor([[1., 2.], [3., 4.]])
        goal = torch.tensor([[1., 2.], [0., 0.], [2., 1.]])
        squared = (sa[:, None] - goal[None]).square().sum(dim=-1)
        dot = (sa[:, None] * goal[None]).sum(dim=-1)
        expected = {
            "norm": -(squared + 1e-6).sqrt(),
            "l2": -squared,
            "dot": dot,
            "cosine": dot / ((sa.norm(dim=-1)[:, None] + 1e-6) * (goal.norm(dim=-1)[None] + 1e-6)),
        }
        for name, logits in expected.items():
            with self.subTest(name=name):
                agent = CPPOAgent({**SETTINGS, "cppo_energy_fn": name})
                torch.testing.assert_close(agent._energy(sa[:, None], goal[None]), logits)
                torch.testing.assert_close(agent._contrastive_logits(sa, goal), logits)

    def test_public_contrastive_loss_modes(self):
        logits = torch.tensor([[1., 2.], [-5., 0.25]])
        targets = torch.arange(len(logits))
        fwd = torch.nn.functional.cross_entropy(logits, targets)
        bwd = torch.nn.functional.cross_entropy(logits.T, targets)
        binary = -torch.nn.functional.logsigmoid(logits.diag()).mean()
        binary -= torch.nn.functional.logsigmoid(-logits[~torch.eye(2, dtype=torch.bool)]).sum() / 4
        for name, expected in {"fwd_infonce": fwd, "bwd_infonce": bwd,
                               "sym_infonce": fwd + bwd, "binary_nce": binary}.items():
            with self.subTest(name=name):
                agent = CPPOAgent({**SETTINGS, "cppo_contrastive_loss_fn": name,
                                   "cppo_logsumexp_penalty_coef": 0})
                torch.testing.assert_close(agent._contrastive_loss(logits), expected)

    def test_public_rate_schedules(self):
        for name, expected in {"none": 1e-3, "linear": 0.000775,
                               "cosine": 0.0001 + .0009 * (1 + 2**-.5) / 2}.items():
            with self.subTest(name=name):
                agent = CPPOAgent({**SETTINGS, "cppo_lr_decay_type": name,
                                   "cppo_learning_rate": 1e-3, "cppo_lr_end": 1e-4})
                agent.gradient_steps = 1
                self.assertAlmostEqual(agent._decayed_rate(1e-3, 4), expected)

    def test_invalid_parameters_fail_before_training(self):
        for key, value in (("gamma", 1), ("learning_rate", float("nan")),
                           ("batch_size", 1), ("epochs", True), ("clip_coef", 0),
                           ("normalize_advantage", "false")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                CPPOAgent({**SETTINGS, "cppo_" + key: value})

    def test_default_collection_is_total_across_paper_environments(self):
        agent = CPPOAgent(DEFAULT_SETTINGS)
        self.assertEqual(agent.config["rollout_steps"], 128 * 512)
        self.assertEqual(agent.config["batch_size"], 512)
        self.assertTrue(agent.config["normalize_advantage"])

    def test_chunked_advantages_match_reference_rollout_normalization(self):
        agent = self.session().agent
        states = torch.tensor([agent._state({**MAP, "position": position})
                               for position in [[0, 0]] * 8 + [[1, 0]] * 8 + [[1, 1]] * 8])
        actions = torch.arange(len(states)) // 8
        with torch.no_grad():
            q = agent._all_q(states)
            raw = q.gather(1, actions[:, None]).squeeze(1) - (agent._distribution(states).probs * q).sum(-1)
            expected = (raw - raw.mean()) / (raw.std(unbiased=False) + 1e-8)
        actual = agent._advantages(states, actions)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        self.assertFalse(actual.requires_grad)
        self.assertGreater(abs(actual[:8].mean().item()), 1e-2)

    def test_actor_and_critic_use_separate_learning_rates(self):
        settings = {**SETTINGS, "cppo_learning_rate": 1e-4, "cppo_critic_learning_rate": 3e-4,
                    "cppo_total_updates": 1, "cppo_rollout_steps": 8}
        session = TrainingSession(MAP, settings, CPPOAgent)
        for _ in range(8):
            session.step()
        self.assertAlmostEqual(session.agent.actor_optimizer.param_groups[0]["lr"], 1e-4)
        self.assertAlmostEqual(session.agent.critic_optimizer.param_groups[0]["lr"], 3e-4)
        for _ in range(8):
            session.step()
        self.assertAlmostEqual(session.agent.actor_optimizer.param_groups[0]["lr"], 1e-7)
        self.assertAlmostEqual(session.agent.critic_optimizer.param_groups[0]["lr"], 1e-7)

    def test_metrics_distinguish_collection_and_policy_collapse(self):
        session = self.session()
        session.run_batch(4)
        metrics = session.agent.metrics
        self.assertGreaterEqual(metrics["rollout_episodes"], 1)
        self.assertEqual(metrics["rollout_steps"], SETTINGS["cppo_rollout_steps"])
        self.assertGreater(metrics["unique_achieved_goals"], 0)
        self.assertIsInstance(metrics["target_seen_in_rollout"], bool)
        self.assertGreaterEqual(metrics["policy_kl"], -1e-6)
        self.assertGreaterEqual(metrics["policy_entropy"], 0)
        self.assertLessEqual(metrics["policy_entropy"], 1.386295)


if __name__ == "__main__":
    unittest.main()
