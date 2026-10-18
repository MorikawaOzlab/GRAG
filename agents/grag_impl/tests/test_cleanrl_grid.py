"""CleanRL型PPOのグリッド接続と、数値計算・モデル互換性を確認する。"""

from copy import deepcopy
import importlib.util
import json
import math
import unittest

from agents.grag import GRAG
from agents.ppo_agent import PPOAgent
from agents.grag_impl.grag import _normalize_advantages, _value_loss
from gridworld.world import GridWorld
from .test_ppo import MAP


SETTINGS = {"seed": 7, "ppo_hidden_size": 8, "ppo_epochs": 1,
            "ppo_rollout_steps": 2, "ppo_minibatch_size": 2, "ppo_target_kl": None}


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is required")
class CleanRLGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.threads)

    def test_tensor_network_uses_dimensions_instead_of_environment_object(self):
        from agents.grag_impl.networks import Agent
        network = Agent(observation_size=7, action_size=3, hidden_size=8)
        states = self.torch.zeros(2, 7)
        generator = self.torch.Generator().manual_seed(5)
        action, log_probability, entropy, value = network.get_action_and_value(
            states, generator=generator)
        self.assertEqual(action.shape, (2,))
        self.assertEqual(value.shape, (2, 1))
        self.assertTrue(((action >= 0) & (action < 3)).all().item())
        before_rng = generator.get_state().clone()
        replayed = network.get_action_and_value(states, action, generator=generator)
        self.assertTrue(self.torch.equal(before_rng, generator.get_state()))
        self.assertTrue(self.torch.equal(replayed[1], log_probability))
        self.assertTrue(self.torch.equal(replayed[2], entropy))
        self.assertTrue(self.torch.equal(network.get_value(states), value))

    def test_value_clipping_matches_reference_loss_and_gradient(self):
        torch = self.torch
        values = torch.tensor([0.9, 2.0], requires_grad=True)
        returns = torch.ones(2)
        old = torch.zeros(2)
        clipped = _value_loss(values, returns, old, 0.2, True)
        self.assertAlmostEqual(clipped.item(), 0.41, places=6)
        clipped.backward()
        self.assertAlmostEqual(values.grad[0].item(), 0.0)
        self.assertAlmostEqual(values.grad[1].item(), 0.5)
        unclipped = _value_loss(values.detach(), returns, old, 0.2, False)
        self.assertAlmostEqual(unclipped.item(), 0.2525, places=6)

    def test_minibatch_normalization_handles_single_and_equal_advantages(self):
        values = self.torch.tensor([0.0, 2.0])
        normalized = _normalize_advantages(values)
        self.assertAlmostEqual(normalized[0].item(), -math.sqrt(0.5), places=6)
        self.assertAlmostEqual(normalized[1].item(), math.sqrt(0.5), places=6)
        single = _normalize_advantages(self.torch.tensor([3.0]))
        self.assertEqual(single.item(), 3.0)
        equal = _normalize_advantages(self.torch.tensor([3.0, 3.0]))
        self.assertTrue(self.torch.isfinite(equal).all().item())
        self.assertEqual(equal.tolist(), [0.0, 0.0])

    def test_cleanrl_optimizer_and_optional_schedule_resume_exactly(self):
        settings = {**SETTINGS, "ppo_learning_rate": 0.001, "ppo_anneal_lr": True,
                    "ppo_total_timesteps": 4}
        observation = GridWorld(MAP).reset()

        def advance(agent, count):
            for _ in range(count):
                action = agent.choose_action(observation)
                agent.learn(observation, action, 0, observation, False)

        original = GRAG(settings)
        advance(original, 2)
        self.assertEqual(original._optimizer.param_groups[0]["eps"], 1e-5)
        self.assertAlmostEqual(original._optimizer.param_groups[0]["lr"], 0.001)
        advance(original, 1)
        saved = json.loads(json.dumps(original.checkpoint(), allow_nan=False))
        restored = GRAG(settings)
        restored.load_checkpoint(saved)
        advance(original, 1)
        advance(restored, 1)
        self.assertAlmostEqual(original._optimizer.param_groups[0]["lr"], 0.0005)
        self.assertEqual(original.checkpoint(), restored.checkpoint())
        advance(original, 2)
        advance(restored, 2)
        self.assertEqual(original._optimizer.param_groups[0]["lr"], 0.0)
        self.assertEqual(original.checkpoint(), restored.checkpoint())

    def test_legacy_grid_model_and_adam_state_can_continue(self):
        observation = GridWorld(MAP).reset()
        legacy = PPOAgent(SETTINGS)
        for _ in range(3):
            action = legacy.choose_action(observation)
            legacy.learn(observation, action, 0, observation, False)
        saved = json.loads(json.dumps(legacy.checkpoint(), allow_nan=False))
        original = deepcopy(saved)
        restored = GRAG(SETTINGS)
        restored.load_checkpoint(saved)
        self.assertEqual(restored.action_probabilities(observation),
                         legacy.action_probabilities(observation))
        self.assertEqual(restored.export()["state_dict"], legacy.export()["state_dict"])
        self.assertEqual(saved, original)
        action = restored.choose_action(observation)
        restored.learn(observation, action, 0, observation, False)
        restored.end_episode()
        self.assertTrue(all(self.torch.isfinite(parameter).all().item()
                            for parameter in restored._model.parameters()))
        self.assertGreater(restored._updates, legacy._updates)
        json.dumps(restored.checkpoint(), allow_nan=False)

    def test_cleanrl_configuration_validation_and_defaults(self):
        agent = GRAG({})
        self.assertEqual(agent._config["ppo_learning_rate"], 2.5e-4)
        self.assertTrue(agent._config["ppo_clip_value_loss"])
        self.assertFalse(agent._config["ppo_anneal_lr"])
        self.assertIsNone(agent._config["ppo_target_kl"])
        for fields in ({"ppo_clip_value_loss": 1}, {"ppo_anneal_lr": "yes"},
                       {"ppo_total_timesteps": 0}, {"ppo_total_timesteps": True},
                       {"ppo_anneal_lr": True}):
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    GRAG(fields)
