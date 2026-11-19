"""SGCRL integration with GridWorld's step, inspection, evaluation and resume APIs."""

import json
import unittest

import torch

from agents.sgcrl import GridWorldSGCRL
from gridworld.storage import DEFAULT_SETTINGS
from gridworld.training_runtime import TrainingSession


class SGCRLSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def session(self):
        maze = {"name": "SGCRL integration", "rows": 2, "cols": 3,
                "start": [0, 0], "goal": [1, 2], "obstacles": [[0, 1]]}
        settings = {**DEFAULT_SETTINGS, "max_steps": 8, "crl_min_replay_size": 8,
                    "crl_batch_size": 8, "crl_hidden_size": 16}
        return TrainingSession(maze, settings, GridWorldSGCRL)

    def test_json_checkpoint_mid_episode_resumes_identically(self):
        original = self.session()
        original.run_batch(4)
        original.step()
        saved = json.loads(json.dumps(original.result(checkpoint=True)))
        restored = TrainingSession.restore(saved, GridWorldSGCRL)
        for _ in range(20):
            self.assertEqual(original.step(), restored.step())
        self.assertEqual(original.agent.metrics, restored.agent.metrics)
        self.assertGreater(original.agent._updates, 0)
        for a, b in zip(original.agent.core.actor.parameters(), restored.agent.core.actor.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_evaluation_and_inspection_do_not_change_training_or_rng(self):
        session = self.session()
        session.run_batch(4)
        saved = session.result(checkpoint=True)
        control = TrainingSession.restore(saved, GridWorldSGCRL)
        before_random = session.agent.core._action_generator.get_state().clone()
        before_steps = session.agent.core._steps
        policy = session.inspect_policy([0, 0])
        self.assertTrue(policy["supported"])
        self.assertAlmostEqual(sum(policy["probabilities"].values()), 1, places=6)
        self.assertGreater(len(session.evaluate()), 1)
        self.assertEqual(session.agent.core._steps, before_steps)
        self.assertTrue(torch.equal(before_random, session.agent.core._action_generator.get_state()))
        for _ in range(12):
            self.assertEqual(session.step(), control.step())


if __name__ == "__main__":
    unittest.main()
