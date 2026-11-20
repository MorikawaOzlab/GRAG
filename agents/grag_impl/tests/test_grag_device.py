import unittest
from unittest.mock import patch

import torch

from agents.grag_impl.grag import GRAG


class GRAGDeviceTests(unittest.TestCase):
    def test_cpu_and_unavailable_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(GRAG({"device": "auto"}).device.type, "cpu")
            with self.assertRaisesRegex(ValueError, "CUDA"):
                GRAG({"device": "cuda"})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_online_and_periodic_updates(self):
        agent = GRAG({"device": "cuda", "seed": 7, "actions": (0, 1),
                      "grag_update_interval": 2, "target_goal_encoder": lambda obs: [0.]})
        agent.initialize_networks(1, 1)
        for index in range(2):
            observation = [float(index)]
            action = agent.choose_action(observation)
            agent.learn(observation, action, 0, [float(index + 1)], False)
            agent.end_episode()
        for network in (agent.actor, agent.sa_encoder, agent.goal_encoder):
            self.assertEqual(next(network.parameters()).device.type, "cuda")
        for optimizer in (agent.actor_optimizer, agent.contrastive_optimizer):
            self.assertEqual(next(iter(optimizer.state.values()))["exp_avg"].device.type, "cuda")
        self.assertTrue(torch.isfinite(torch.tensor(agent.policy_loss)))
        self.assertTrue(torch.isfinite(torch.tensor(agent.contrastive_loss)))


if __name__ == "__main__":
    unittest.main()
