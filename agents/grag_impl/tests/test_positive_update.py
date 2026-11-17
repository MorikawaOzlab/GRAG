import unittest
from unittest.mock import Mock

from agents.grag_impl.grag import GRAG


class PositiveUpdateTests(unittest.TestCase):
    def setUp(self):
        self.agent = GRAG({"seed": 1, "grag_positive_update_count": 3})
        self.observation = {"rows": 2, "cols": 2, "obstacles": [],
                            "goal": [1, 1], "position": [0, 0]}
        self.agent.initialize_networks(self.observation)
        self.agent._update_grag = Mock()
        self.agent._update_grag.return_value = True
        self.agent._update_ppo = Mock(side_effect=self.agent.memory.clear)

    def step(self, success=False):
        self.agent.learn(self.observation, "right", int(success), self.observation, success)

    def test_only_successful_trajectories_count(self):
        self.step()
        self.agent.end_episode()
        self.step()
        self.step(True)
        self.agent.end_episode()
        self.agent._update_grag.assert_not_called()
        self.step(True)
        self.agent.end_episode()
        self.agent._update_grag.assert_not_called()
        self.step(True)
        self.agent._update_grag.assert_called_once()
        self.agent._update_ppo.assert_called_once()

    def test_default_threshold_and_no_success_updates(self):
        agent = GRAG({"seed": 1, "ppo_rollout_steps": 4})
        self.assertEqual(agent.positive_update_count, 8)
        agent.initialize_networks(self.observation)
        agent._update_grag = Mock(return_value=True)
        agent._update_ppo = Mock()
        for _ in range(128):
            agent.learn(self.observation, "right", 0, self.observation, False)
            agent.end_episode()
        agent._update_grag.assert_not_called()
        agent._update_ppo.assert_not_called()
        self.assertLessEqual(len(agent.memory), 4)

    def test_threshold_is_validated(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GRAG({"grag_positive_update_count": value})


if __name__ == "__main__":
    unittest.main()
