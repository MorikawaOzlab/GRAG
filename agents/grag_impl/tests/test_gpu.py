"""Device selection, actual CUDA training, and portable checkpoint restoration."""
import json
import unittest
from unittest.mock import patch

import torch

from agents.grag import GRAG
from gridworld.world import GridWorld
from .test_grag_update import MAP, SETTINGS


def advance(agent):
    world = GridWorld(MAP)
    observation = world.reset()
    for action in ("down", "up"):
        following, reached, _ = world.step(action)
        agent.learn(observation, action, 0, following, reached)
        observation = following
    agent.end_episode()


class DeviceSelectionTests(unittest.TestCase):
    def test_auto_falls_back_to_cpu_and_explicit_cuda_fails_when_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(GRAG({}).device, torch.device("cpu"))
            with self.assertRaises(ValueError):
                GRAG({"crl_device": "cuda"})

    def test_invalid_devices_are_rejected(self):
        for value in (False, 0, "invalid", "mps", "cpu:1", "cuda:999"):
            with self.subTest(device=value), self.assertRaises(ValueError):
                GRAG({"crl_device": value})


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
class CUDATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def agent(self):
        return GRAG({**SETTINGS, "crl_device": "cuda"})

    def test_default_selects_cuda(self):
        self.assertEqual(GRAG({}).device, torch.device("cuda:0"))

    def test_network_inputs_gradients_and_adam_momentum_are_on_cuda(self):
        agent = self.agent()
        agent.initialize_networks(GridWorld(MAP).reset())
        inputs = []
        handles = [network.register_forward_pre_hook(
            lambda module, args: inputs.append(args[0].device.type))
            for network in (agent.actor, agent.sa_encoder, agent.goal_encoder)]
        try:
            advance(agent)
            observation = GridWorld(MAP).reset()
            agent.choose_action(observation)
            agent.choose_action(observation, greedy=True)
            probabilities = agent.action_probabilities(observation)
            scores = agent.crl_action_scores(observation)
        finally:
            for handle in handles:
                handle.remove()
        self.assertGreater(agent._updates, 0)
        self.assertTrue(inputs)
        self.assertEqual(set(inputs), {"cuda"})
        for network in (agent.actor, agent.sa_encoder, agent.goal_encoder):
            for parameter in network.parameters():
                self.assertEqual(parameter.device.type, "cuda")
                self.assertEqual(parameter.grad.device.type, "cuda")
                self.assertTrue(torch.isfinite(parameter).all())
        self.assertEqual(agent._log_alpha.device.type, "cuda")
        self.assertEqual(agent._log_alpha.grad.device.type, "cuda")
        for optimizer in (agent._actor_optimizer, agent._encoder_optimizer, agent._alpha_optimizer):
            for state in optimizer.state.values():
                for key in ("exp_avg", "exp_avg_sq"):
                    self.assertEqual(state[key].device.type, "cuda")
        self.assertAlmostEqual(sum(probabilities.values()), 1, places=6)
        self.assertTrue(all(isinstance(value, float) for value in scores.values()))

    def test_cuda_checkpoint_resumes_identically(self):
        agent = self.agent()
        advance(agent)
        saved = json.loads(json.dumps(agent.checkpoint(), allow_nan=False))
        restored = self.agent()
        restored.load_checkpoint(saved)
        self.assertEqual(agent.checkpoint(), restored.checkpoint())
        advance(agent)
        advance(restored)
        self.assertEqual(agent.checkpoint(), restored.checkpoint())

    def test_cpu_checkpoint_can_resume_on_gpu_and_gpu_checkpoint_on_cpu(self):
        cpu = GRAG(SETTINGS)
        advance(cpu)
        gpu = self.agent()
        gpu.load_checkpoint(json.loads(json.dumps(cpu.checkpoint())))
        self.assertEqual(gpu.device.type, "cuda")
        observation = GridWorld(MAP).reset()
        self.assertTrue(torch.allclose(
            torch.tensor(list(cpu.action_probabilities(observation).values())),
            torch.tensor(list(gpu.action_probabilities(observation).values())), atol=1e-6))
        advance(gpu)
        cpu.load_checkpoint(json.loads(json.dumps(gpu.checkpoint())))
        self.assertEqual(cpu.device.type, "cpu")
        advance(cpu)
        self.assertGreater(cpu._updates, gpu._updates)

    def test_export_can_override_saved_cuda_device_on_cpu_only_host(self):
        agent = self.agent()
        advance(agent)
        model = agent.export()
        with patch("torch.cuda.is_available", return_value=False):
            restored = GRAG.from_export(model, device="cpu")
        self.assertEqual(next(restored.actor.parameters()).device.type, "cpu")
        advance(restored)
        self.assertGreater(restored._updates, agent._updates)


if __name__ == "__main__":
    unittest.main()
