import unittest
from unittest.mock import patch

import torch

from agents.grag_impl.grag import GRAG


class InitialCollectionTests(unittest.TestCase):
    def test_first_10000_steps_then_regular_interval(self):
        agent = GRAG({"seed": 1, "actions": (0, 1), "grag_update_interval": 1280})
        agent.initialize_networks(1, 1)
        with patch.object(agent, "_update_networks") as update:
            for step in range(11280):
                agent.pending = {"state": torch.tensor([0.]), "action": 0,
                                 "observation": [0.], "log_prob": 0.0}
                agent.learn([0.], 0, 0, [0.], False)
                if step == 9998:
                    self.assertEqual(update.call_count, 0)
                    self.assertEqual(len(agent._update_batch), 9999)
                if step == 9999:
                    self.assertEqual([len(call.args[0]) for call in update.call_args_list],
                                     [1280] * 7 + [1040])
                    self.assertEqual(agent._update_batch, [])
                if step == 11278:
                    self.assertEqual(update.call_count, 8)
            self.assertEqual(update.call_count, 9)
            self.assertEqual(len(update.call_args.args[0]), 1280)


if __name__ == "__main__":
    unittest.main()
