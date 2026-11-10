"""PRTS losses, calibrated Q, advantages and PPO Actor updates."""

import math
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from agents.grag_impl.grag import GRAG, Transition, state_action_encoder


class PRTSLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.agent = GRAG({"seed": 7, "crl_gamma": 0.5, "crl_coef": 1.3})
        with torch.random.fork_rng():
            torch.manual_seed(7)
            self.agent.sa_encoder = nn.Linear(12, 3).double()
            self.agent.goal_encoder = nn.Linear(384, 3).double()
            self.agent.actor = nn.Sequential(nn.Flatten(start_dim=-3), nn.Linear(12, 4)).double()
            self.agent.critic = nn.Linear(12, 1).double()
            self.agent._goal_embedding_cache = {"A": torch.randn(384), "B": torch.randn(384)}

    def transition(self, prompt="A", position=(0, 0), action=0, success=False, end=False):
        goal = (1, 0) if prompt == "B" else (1, 1)
        state = torch.zeros(3, 2, 2, dtype=torch.float64)
        state[1, goal[0], goal[1]] = 1
        state[2, position[0], position[1]] = 1
        return Transition(state, action, 0., 0., 0., 0., success, end, prompt)

    def batch(self):
        return [self.transition(position=(0, 0), action=1),
                self.transition(position=(0, 1), action=2),
                self.transition(position=(1, 0), action=3, success=True),
                self.transition("B", position=(0, 0), action=1, success=True)]

    def reference_loss(self, transitions, remaining):
        """Evaluate the paper's sums directly, without cross_entropy or softmax."""
        prompts = list(dict.fromkeys(tr.goal_prompt for tr in transitions))
        labels = [prompts.index(tr.goal_prompt) for tr in transitions]
        states = torch.stack([tr.state for tr in transitions])
        actions = torch.tensor([tr.action for tr in transitions])
        goals = torch.stack([self.agent._goal_embedding_cache[p] for p in prompts]).double()
        with torch.no_grad():
            phi = self.agent.sa_encoder(state_action_encoder(states, actions))
            psi = self.agent.goal_encoder(goals)
            scores = [[float((p * g).sum()) for g in psi] for p in phi]
        gamma = self.agent.settings.get("crl_gamma", self.agent.gamma)
        weights = [gamma ** steps for steps in remaining]

        def logsumexp(values):
            maximum = max(values)
            return maximum + math.log(sum(math.exp(value - maximum) for value in values))

        forward = sum(-weight * (row[label] - logsumexp(row))
                      for row, label, weight in zip(scores, labels, weights)) / len(transitions)
        reverse = 0.
        for label in labels:
            positive = [index for index, task in enumerate(labels) if task == label]
            normalizer = sum(weights[index] for index in positive)
            denominator = logsumexp([row[label] for row in scores])
            reverse -= sum(weights[index] / normalizer * (scores[index][label] - denominator)
                           for index in positive) / len(transitions)
        return self.agent.settings.get("crl_coef", 1.) * (forward + reverse)

    def test_matches_prts_sums_with_repeated_goals_and_unequal_task_counts(self):
        transitions = self.batch()
        actual = self.agent._compute_crl_loss(transitions)
        self.assertEqual(actual.ndim, 0)
        self.assertAlmostEqual(actual.item(), self.reference_loss(transitions, [2, 1, 0, 0]), places=12)

    def test_excludes_failures_and_unfinished_episodes(self):
        transitions = [self.transition(), self.transition(end=True),
                       *self.batch(), self.transition(), self.transition()]
        loss = self.agent._compute_crl_loss(transitions)
        self.assertAlmostEqual(loss.item(), self.reference_loss(transitions[2:6], [2, 1, 0, 0]), places=12)

    def test_goal_change_does_not_label_previous_goal_as_successful(self):
        transitions = [self.transition(), self.transition("B"),
                       self.transition("B", position=(0, 1), success=True)]
        loss = self.agent._compute_crl_loss(transitions)
        self.assertAlmostEqual(loss.item(), self.reference_loss(transitions[1:], [1, 0]), places=12)

    def test_no_data_or_only_one_positive_skips_without_encoding_language(self):
        batches = ([], [self.transition(success=True)],
                   [self.transition(), self.transition(end=True)],
                   [self.transition(end=True), self.transition(success=True)])
        with patch("agents.grag_impl.grag.encode_language") as encode:
            for batch in batches:
                self.assertIsNone(self.agent._compute_crl_loss(batch))
            encode.assert_not_called()

    def test_gradients_reach_both_encoders_without_updating_weights(self):
        transitions = self.batch()
        self.agent.memory = transitions.copy()
        parameters = list(self.agent.sa_encoder.parameters()) + list(self.agent.goal_encoder.parameters())
        before = [parameter.detach().clone() for parameter in parameters]
        loss = self.agent._compute_crl_loss(transitions)
        for parameter, saved in zip(parameters, before):
            torch.testing.assert_close(parameter, saved, rtol=0, atol=0)
            self.assertIsNone(parameter.grad)
        loss.backward()
        for encoder in (self.agent.sa_encoder, self.agent.goal_encoder):
            gradients = [p.grad for p in encoder.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)
        for network in (self.agent.actor, self.agent.critic):
            self.assertTrue(all(p.grad is None for p in network.parameters()))
        self.assertEqual(len(self.agent.memory), len(transitions))

    def test_gradients_match_finite_differences(self):
        transitions = self.batch()
        self.agent._compute_crl_loss(transitions).backward()
        epsilon = 1e-5
        for weight in (self.agent.sa_encoder.weight, self.agent.goal_encoder.weight):
            derivative = weight.grad[0, 0].item()
            original = weight[0, 0].item()
            with torch.no_grad():
                weight[0, 0] = original + epsilon
                plus = self.agent._compute_crl_loss(transitions).item()
                weight[0, 0] = original - epsilon
                minus = self.agent._compute_crl_loss(transitions).item()
                weight[0, 0] = original
            self.assertAlmostEqual(derivative, (plus - minus) / (2 * epsilon), places=8)

    def test_language_cache_freezes_language_features_but_recomputes_goal_projection(self):
        self.agent._goal_embedding_cache.clear()
        embeddings = torch.ones(2, 384, requires_grad=True)
        with patch("agents.grag_impl.grag.encode_language", return_value=embeddings) as encode:
            with patch.object(self.agent.goal_encoder, "forward", wraps=self.agent.goal_encoder.forward) as forward:
                for _ in range(2):
                    self.agent._compute_crl_loss(self.batch()).backward()
                encode.assert_called_once_with(["A", "B"])
                self.assertEqual(forward.call_count, 2)
        self.assertIsNone(embeddings.grad)
        self.assertTrue(all(not g.requires_grad for g in self.agent._goal_embedding_cache.values()))

    def test_missing_goal_prompt_is_rebuilt_from_state(self):
        transitions = [self.transition(None), self.transition(None, position=(0, 1), success=True)]
        with patch("agents.grag_impl.grag.encode_language", return_value=torch.ones(1, 384)) as encode:
            loss = self.agent._compute_crl_loss(transitions)
            self.assertTrue(torch.isfinite(loss))
            prompts = encode.call_args.args[0]
            self.assertEqual(len(prompts), 1)
            self.assertIn("座標(1, 1)", prompts[0])
        self.assertTrue(all(tr.goal_prompt is None for tr in transitions))

    def test_default_gamma_and_coefficient_and_disabled_loss(self):
        self.agent.settings.clear()
        transitions = self.batch()
        self.assertAlmostEqual(self.agent._compute_crl_loss(transitions).item(),
                               self.reference_loss(transitions, [2, 1, 0, 0]), places=12)
        self.agent.settings["crl_coef"] = 0.
        self.assertIsNone(self.agent._compute_crl_loss(transitions))

    def test_encoder_update_changes_both_encoders_and_reduces_loss(self):
        transitions = self.batch()
        self.agent.memory = transitions.copy()
        encoders = (self.agent.sa_encoder, self.agent.goal_encoder)
        encoder_before = [[p.detach().clone() for p in encoder.parameters()] for encoder in encoders]
        other_networks = (self.agent.actor, self.agent.critic)
        other_before = [[p.detach().clone() for p in network.parameters()] for network in other_networks]
        before_loss = self.agent._compute_crl_loss(transitions)
        self.assertIsNone(self.agent._update_crl_encoders(before_loss))
        self.assertLess(self.agent._compute_crl_loss(transitions).item(), before_loss.item())
        for encoder, before in zip(encoders, encoder_before):
            self.assertTrue(any(not torch.equal(p, old) for p, old in zip(encoder.parameters(), before)))
        for network, before in zip(other_networks, other_before):
            for parameter, old in zip(network.parameters(), before):
                torch.testing.assert_close(parameter, old, rtol=0, atol=0)
                self.assertIsNone(parameter.grad)
        self.assertEqual(len(self.agent.memory), len(transitions))
        optimized = self.agent._grag_encoder_optimizer.param_groups[0]["params"]
        self.assertEqual({id(p) for p in optimized}, {id(p) for e in encoders for p in e.parameters()})
        self.assertIsNone(self.agent.optimizer)

    def test_encoder_optimizer_is_reused_and_old_gradients_are_cleared(self):
        clean = deepcopy(self.agent)
        transitions = self.batch()
        parameters = list(self.agent.sa_encoder.parameters()) + list(self.agent.goal_encoder.parameters())
        for step in range(2):
            # 汚れた勾配の有無で、同じ初期値・損失からの更新結果が変わらないこと。
            for parameter in parameters:
                parameter.grad = torch.full_like(parameter, 1000.)
            self.agent._update_crl_encoders(self.agent._compute_crl_loss(transitions))
            clean._update_crl_encoders(clean._compute_crl_loss(transitions))
            if step == 0:
                optimizer = self.agent._grag_encoder_optimizer
            self.assertIs(self.agent._grag_encoder_optimizer, optimizer)
            for parameter in parameters:
                self.assertEqual(optimizer.state[parameter]["step"].item(), step + 1)
        clean_parameters = list(clean.sa_encoder.parameters()) + list(clean.goal_encoder.parameters())
        for actual, expected in zip(parameters, clean_parameters):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_encoder_gradients_are_clipped_together_and_learning_rate_is_configurable(self):
        self.agent.settings["crl_learning_rate"] = 0.001
        loss = 1e6 * self.agent._compute_crl_loss(self.batch())
        self.agent._update_crl_encoders(loss)
        self.assertEqual(self.agent._grag_encoder_optimizer.param_groups[0]["lr"], 0.001)
        parameters = list(self.agent.sa_encoder.parameters()) + list(self.agent.goal_encoder.parameters())
        gradient_norm = torch.cat([p.grad.flatten() for p in parameters]).norm().item()
        self.assertLessEqual(gradient_norm, 0.500001)
        self.assertGreater(gradient_norm, 0.49)
        self.assertTrue(all(torch.isfinite(p).all() for p in parameters))

    def test_invalid_encoder_loss_and_learning_rate_do_not_update_weights(self):
        invalid = (None, 1., torch.ones(2, requires_grad=True), torch.tensor(1.),
                   torch.tensor(float("nan"), requires_grad=True),
                   torch.tensor(float("inf"), requires_grad=True))
        before = [p.detach().clone() for p in self.agent.sa_encoder.parameters()]
        for loss in invalid:
            with self.subTest(loss=loss), self.assertRaises((ValueError, FloatingPointError)):
                self.agent._update_crl_encoders(loss)
        for value in (-1., float("nan"), float("inf")):
            self.agent.settings["crl_learning_rate"] = value
            with self.subTest(learning_rate=value), self.assertRaises(ValueError):
                self.agent._update_crl_encoders(self.agent._compute_crl_loss(self.batch()))
        self.assertFalse(hasattr(self.agent, "_grag_encoder_optimizer"))
        for parameter, saved in zip(self.agent.sa_encoder.parameters(), before):
            torch.testing.assert_close(parameter, saved, rtol=0, atol=0)

    def test_nonfinite_gradients_prevent_step_and_next_update_can_recover(self):
        parameter = self.agent.sa_encoder.weight
        before = parameter.detach().clone()
        # 有限の損失でも、sqrt(0)の逆伝播ではNaN勾配が発生する。
        bad_loss = (parameter - parameter.detach()).square().sqrt().sum()
        with self.assertRaises(RuntimeError):
            self.agent._update_crl_encoders(bad_loss)
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
        self.assertEqual(len(self.agent._grag_encoder_optimizer.state), 0)
        self.agent._update_crl_encoders(self.agent._compute_crl_loss(self.batch()))
        self.assertTrue(torch.isfinite(parameter).all())
        self.assertFalse(torch.equal(parameter, before))

    def test_relocated_q_keeps_explicit_constant_and_gradients(self):
        transitions = self.batch()
        states = torch.stack([tr.state for tr in transitions])
        actions = torch.tensor([tr.action for tr in transitions])
        features = state_action_encoder(states, actions)
        goals = torch.stack([self.agent._goal_embedding_cache[tr.goal_prompt] for tr in transitions])
        constant = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
        q_values = self.agent._compute_crl_q(features, goals, goal_log_constant=constant)
        phi = self.agent.sa_encoder(features)
        psi = self.agent.goal_encoder(goals.double())
        torch.testing.assert_close(q_values, ((phi * psi).sum(-1) - constant).exp())
        q_values.sum().backward()
        for encoder in (self.agent.sa_encoder, self.agent.goal_encoder):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in encoder.parameters()))

    def test_grag_update_passes_calibrated_all_action_q_to_advantages(self):
        transitions = self.batch()
        self.agent.memory = transitions.copy()
        self.agent._compute_grag_advantages = Mock(wraps=self.agent._compute_grag_advantages)
        self.agent._update_ppo_actor = Mock(return_value=True)
        with patch.object(self.agent, "_compute_crl_q", wraps=self.agent._compute_crl_q) as compute_q:
            self.agent._update_grag()
            self.assertEqual(compute_q.call_count, 1)
        q_values = self.agent._compute_grag_advantages.call_args.args[1]
        self.assertEqual(q_values.shape, (len(transitions), self.agent.CRL_NUM_ACTIONS))
        self.assertFalse(q_values.requires_grad)
        self.assertEqual(self.agent.memory, [])
        advantages = self.agent._update_ppo_actor.call_args.args[1]
        with torch.no_grad():
            states = torch.stack([tr.state for tr in transitions])
            probabilities = self.agent.actor(states).softmax(-1)
            actions = torch.tensor([tr.action for tr in transitions])
            expected = q_values[torch.arange(len(transitions)), actions] - (probabilities * q_values).sum(-1)
        torch.testing.assert_close(advantages, expected)
        self.assertFalse(advantages.requires_grad)

        # 更新後の重みを使い、各行動のQと補正定数をスカラーの式で照合する。
        states = torch.stack([tr.state for tr in transitions])
        reference_features = state_action_encoder(states, torch.tensor([tr.action for tr in transitions]))
        gamma = self.agent.settings["crl_gamma"]
        remaining = [2, 1, 0, 0]
        with torch.no_grad():
            reference_phi = self.agent.sa_encoder(reference_features)
            for index, transition in enumerate(transitions):
                goal_input = self.agent._goal_embedding_cache[transition.goal_prompt].double()
                psi = self.agent.goal_encoder(goal_input)
                reference_scores = [float((phi * psi).sum()) for phi in reference_phi]
                maximum = max(reference_scores)
                log_z = maximum + math.log(sum(math.exp(s - maximum) for s in reference_scores))
                total_weight = sum(gamma ** steps for tr, steps in zip(transitions, remaining)
                                   if tr.goal_prompt == transition.goal_prompt)
                constant = log_z - math.log(total_weight) - math.log1p(-gamma)
                for action in range(self.agent.CRL_NUM_ACTIONS):
                    phi = self.agent.sa_encoder(state_action_encoder(transition.state, action))
                    expected = math.exp(float((phi * psi).sum()) - constant)
                    self.assertAlmostEqual(q_values[index, action].item(), expected, places=12)

    def test_advantages_match_known_policy_and_taken_actions_without_recomputing_q(self):
        with torch.no_grad():
            self.agent.actor[-1].weight.zero_()
            self.agent.actor[-1].bias.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]).double().log())
        q_values = torch.tensor([[1., 2., 3., 4.], [10., 10., 10., 10.],
                                 [0., 1., 0., 1.], [4., 3., 2., 1.]],
                                dtype=torch.float64, requires_grad=True)
        saved = q_values.detach().clone()
        with patch.object(self.agent, "_compute_crl_q", side_effect=AssertionError("Q was recomputed")):
            advantages = self.agent._compute_grag_advantages(self.batch(), q_values)
        torch.testing.assert_close(advantages, torch.tensor([-1., 0., 0.4, 1.], dtype=torch.float64),
                                   rtol=1e-7, atol=1e-7)
        torch.testing.assert_close(q_values, saved, rtol=0, atol=0)
        self.assertFalse(advantages.requires_grad)
        self.assertIsNone(q_values.grad)
        self.assertTrue(all(p.grad is None for p in self.agent.actor.parameters()))

    def test_advantage_minibatches_preserve_order_and_handle_last_partial_batch(self):
        self.agent.minibatch_size = 2
        transitions = [*self.batch(), self.transition(position=(1, 0), action=2)]
        q_values = torch.arange(20, dtype=torch.float64).reshape(5, 4)
        sizes = []
        handle = self.agent.actor.register_forward_pre_hook(lambda _, args: sizes.append(len(args[0])))
        try:
            actual = self.agent._compute_grag_advantages(transitions, q_values)
        finally:
            handle.remove()
        self.assertEqual(sizes, [2, 2, 1])
        with torch.no_grad():
            expected = torch.stack([
                row[tr.action] - (self.agent.actor(tr.state).softmax(-1) * row).sum()
                for tr, row in zip(transitions, q_values)
            ])
        torch.testing.assert_close(actual, expected)

    def test_empty_and_single_experience_advantages(self):
        self.assertIsNone(self.agent._compute_grag_advantages([], torch.empty(0, 4)))
        result = self.agent._compute_grag_advantages([self.transition()], torch.full((1, 4), 2.))
        self.assertEqual(result.shape, (1,))
        torch.testing.assert_close(result, torch.zeros(1))

    def test_advantage_invalid_inputs_and_nonfinite_policy_are_rejected(self):
        transitions = self.batch()
        for values in (torch.ones(4), torch.ones(4, 3), torch.ones(3, 4), torch.ones(4, 4, dtype=torch.long)):
            with self.subTest(shape=values.shape, dtype=values.dtype), self.assertRaises(ValueError):
                self.agent._compute_grag_advantages(transitions, values)
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(FloatingPointError):
                self.agent._compute_grag_advantages(transitions, torch.full((4, 4), value))
        with torch.no_grad():
            self.agent.actor[-1].weight.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            self.agent._compute_grag_advantages(transitions, torch.ones(4, 4))

    def test_actor_update_uses_fixed_advantages_and_only_changes_actor(self):
        self.agent.update_epochs = 1
        self.agent.ent_coef = 0.
        transition = self.transition(action=1)
        transition.state.requires_grad_(True)
        self.agent.memory = [transition]
        with torch.no_grad():
            before_probability = self.agent.actor(transition.state).softmax(-1)[1].item()
            transition.old_log_prob = math.log(before_probability)
        other_parameters = [p for network in (self.agent.critic, self.agent.sa_encoder, self.agent.goal_encoder)
                            for p in network.parameters()]
        before = [p.detach().clone() for p in other_parameters]
        advantages = (1 + 0 * self.agent.sa_encoder.weight.sum()).reshape(1)
        with patch.object(self.agent, "_compute_gae", side_effect=AssertionError("GAE was recomputed")):
            updated = self.agent._update_ppo_actor([transition], advantages)
        self.assertTrue(updated)
        with torch.no_grad():
            self.assertGreater(self.agent.actor(transition.state).softmax(-1)[1].item(), before_probability)
        for parameter, saved in zip(other_parameters, before):
            torch.testing.assert_close(parameter, saved, rtol=0, atol=0)
            self.assertIsNone(parameter.grad)
        self.assertIsNone(transition.state.grad)
        self.assertIs(self.agent.memory[0], transition)
        optimized = self.agent._grag_actor_optimizer.param_groups[0]["params"]
        self.assertEqual({id(p) for p in optimized}, {id(p) for p in self.agent.actor.parameters()})
        self.assertEqual(self.agent._grag_actor_optimizer.param_groups[0]["lr"], self.agent.learning_rate)

    def test_ppo_clipping_blocks_only_updates_that_improve_outside_the_clip_range(self):
        cases = ((1.5, 1., 0), (1.5, -1., -1), (0.5, -1., 0), (0.5, 1., 1))
        for ratio, advantage, direction in cases:
            with self.subTest(ratio=ratio, advantage=advantage):
                agent = deepcopy(self.agent)
                agent.update_epochs = 1
                agent.ent_coef = 0.
                with torch.no_grad():
                    agent.actor[-1].weight.zero_()
                    agent.actor[-1].bias.zero_()
                transition = self.transition(action=0)
                transition.old_log_prob = math.log(0.25 / ratio)
                agent._update_ppo_actor([transition], torch.tensor([advantage], dtype=torch.float64))
                with torch.no_grad():
                    probability = agent.actor(transition.state).softmax(-1)[0].item()
                if direction == 0:
                    self.assertEqual(probability, 0.25)
                elif direction < 0:
                    self.assertLess(probability, 0.25)
                else:
                    self.assertGreater(probability, 0.25)

    def test_actor_epochs_partial_minibatches_and_adam_state_are_reused(self):
        self.agent.update_epochs = 3
        self.agent.minibatch_size = 2
        transitions = [*self.batch(), self.transition(position=(1, 0), action=2)]
        advantages = torch.tensor([-2., -1., 0., 1., 2.], dtype=torch.float64, requires_grad=True)
        saved_advantages = advantages.detach().clone()
        sizes = []
        for call in range(2):
            with torch.no_grad():
                for transition in transitions:
                    logits = self.agent.actor(transition.state)
                    transition.old_log_prob = logits.log_softmax(-1)[transition.action].item()
            handle = self.agent.actor.register_forward_pre_hook(lambda _, args: sizes.append(len(args[0])))
            try:
                self.assertTrue(self.agent._update_ppo_actor(transitions, advantages))
            finally:
                handle.remove()
            if call == 0:
                optimizer = self.agent._grag_actor_optimizer
            self.assertIs(self.agent._grag_actor_optimizer, optimizer)
            for parameter in self.agent.actor.parameters():
                self.assertEqual(optimizer.state[parameter]["step"].item(), 9 * (call + 1))
        self.assertEqual(sizes, [2, 2, 1] * 6)
        torch.testing.assert_close(advantages, saved_advantages, rtol=0, atol=0)
        self.assertIsNone(advantages.grad)
        norm = torch.cat([p.grad.flatten() for p in self.agent.actor.parameters()]).norm().item()
        self.assertLessEqual(norm, 0.500001)

    def test_entropy_term_increases_entropy_when_normalized_advantages_are_zero(self):
        self.agent.update_epochs = 1
        self.agent.ent_coef = 0.1
        with torch.no_grad():
            self.agent.actor[-1].weight.zero_()
            self.agent.actor[-1].bias.copy_(torch.tensor([2., 0., -1., -2.]))
        transitions = [self.transition(), self.transition()]
        with torch.no_grad():
            logits = self.agent.actor(transitions[0].state)
            entropy_before = torch.distributions.Categorical(logits=logits).entropy().item()
            for transition in transitions:
                transition.old_log_prob = logits.log_softmax(-1)[transition.action].item()
        self.agent._update_ppo_actor(transitions, torch.ones(2))
        with torch.no_grad():
            entropy_after = torch.distributions.Categorical(
                logits=self.agent.actor(transitions[0].state)).entropy().item()
        self.assertGreater(entropy_after, entropy_before)

    def test_empty_or_disabled_actor_update_keeps_memory_and_skips_optimizer(self):
        self.agent.memory = self.batch()
        self.assertFalse(self.agent._update_ppo_actor([], torch.empty(0)))
        self.agent.update_epochs = 0
        self.assertFalse(self.agent._update_ppo_actor(self.agent.memory, torch.ones(4)))
        self.assertEqual(len(self.agent.memory), 4)
        self.assertFalse(hasattr(self.agent, "_grag_actor_optimizer"))

    def test_actor_update_rejects_invalid_advantages_and_old_probabilities(self):
        transitions = self.batch()
        invalid = (torch.ones(3), torch.ones(4, 1), torch.ones(4, dtype=torch.long),
                   torch.full((4,), float("nan")), torch.full((4,), float("inf")))
        for advantages in invalid:
            with self.subTest(shape=advantages.shape), self.assertRaises((ValueError, FloatingPointError)):
                self.agent._update_ppo_actor(transitions, advantages)
        transitions[0].old_log_prob = float("nan")
        with self.assertRaises(FloatingPointError):
            self.agent._update_ppo_actor(transitions, torch.ones(4))
        self.assertFalse(hasattr(self.agent, "_grag_actor_optimizer"))

    def test_complete_grag_update_trains_encoders_and_actor_and_clears_memory(self):
        transitions = self.batch()
        with torch.no_grad():
            for transition in transitions:
                logits = self.agent.actor(transition.state)
                transition.old_log_prob = logits.log_softmax(-1)[transition.action].item()
        networks = (self.agent.sa_encoder, self.agent.goal_encoder, self.agent.actor, self.agent.critic)
        before = [[p.detach().clone() for p in network.parameters()] for network in networks]
        self.agent.memory = transitions.copy()
        self.agent._update_grag()
        self.assertEqual(self.agent.memory, [])
        for network, saved in zip(networks[:3], before[:3]):
            self.assertTrue(any(not torch.equal(p, old) for p, old in zip(network.parameters(), saved)))
        for parameter, saved in zip(self.agent.critic.parameters(), before[3]):
            torch.testing.assert_close(parameter, saved, rtol=0, atol=0)
        self.assertTrue(hasattr(self.agent, "_grag_encoder_optimizer"))
        self.assertTrue(hasattr(self.agent, "_grag_actor_optimizer"))

    def test_positive_reference_check_requires_success_for_every_goal(self):
        self.assertFalse(self.agent._has_crl_positive_references([]))
        self.assertFalse(self.agent._has_crl_positive_references(
            [self.transition(), self.transition(end=True)]))
        self.assertTrue(self.agent._has_crl_positive_references(self.batch()))
        # 別試行の失敗経験が含まれていても、同じ目標の成功正例があればよい。
        self.assertTrue(self.agent._has_crl_positive_references(
            [self.transition(end=True), *self.batch()]))

    def test_grag_waits_without_training_and_resumes_after_missing_positive_arrives(self):
        transitions = self.batch()
        transitions[-1].reached_goal = False
        transitions[-1].episode_end = True
        self.agent.memory = transitions.copy()
        self.agent._compute_grag_advantages = Mock(return_value=torch.zeros(len(transitions) + 1))
        self.agent._update_ppo_actor = Mock(return_value=True)
        with patch.object(self.agent, "_compute_crl_loss", wraps=self.agent._compute_crl_loss) as compute_loss:
            with patch.object(self.agent, "_prepare_crl_q_inputs", wraps=self.agent._prepare_crl_q_inputs) as prepare_q:
                self.agent._update_grag()
                compute_loss.assert_not_called()
                prepare_q.assert_not_called()
                self.agent._compute_grag_advantages.assert_not_called()
                self.agent._update_ppo_actor.assert_not_called()
                self.assertFalse(hasattr(self.agent, "_grag_encoder_optimizer"))
                self.assertEqual([id(tr) for tr in self.agent.memory], [id(tr) for tr in transitions])

                self.agent.memory.append(self.transition("B", success=True))
                self.agent._update_grag()
                self.assertEqual(compute_loss.call_count, 1)
                self.assertEqual(prepare_q.call_count, 1)
        self.agent._update_ppo_actor.assert_called_once()
        self.assertEqual(self.agent.memory, [])

    def test_invalid_settings_and_nonfinite_logits_are_rejected(self):
        for name, value in (("crl_gamma", 0.), ("crl_gamma", 1.), ("crl_gamma", float("nan")),
                            ("crl_coef", -1.), ("crl_coef", float("inf"))):
            previous = self.agent.settings[name]
            self.agent.settings[name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.agent._compute_crl_loss(self.batch())
            self.agent.settings[name] = previous
        with torch.no_grad():
            self.agent.sa_encoder.weight.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            self.agent._compute_crl_loss(self.batch())


if __name__ == "__main__":
    unittest.main()
