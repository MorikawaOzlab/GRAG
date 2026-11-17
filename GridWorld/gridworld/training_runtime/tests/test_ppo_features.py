import unittest
from .test_session import MAP, SETTINGS


class PPOFeaturesTests(unittest.TestCase):
    def test_ppo_inputs_are_only_three_grid_channels(self):
        from agents.ppo_agent_impl.agent import PPOAgent
        observation = {**MAP, "position": MAP["start"]}
        shape, features = PPOAgent._features(observation)
        rows, cols = shape
        self.assertEqual(len(features), 3 * rows * cols)
        for row in range(rows):
            for col in range(cols):
                index = row * cols + col
                self.assertEqual(features[index], float([row, col] in MAP["obstacles"]))
                self.assertEqual(features[rows * cols + index], float([row, col] == MAP["goal"]))
                self.assertEqual(features[2 * rows * cols + index], float([row, col] == MAP["start"]))
        agent = PPOAgent({**SETTINGS, "seed": 7})
        agent.choose_action(observation, greedy=True)
        saved = agent.export()
        self.assertEqual(saved["observation_encoding"], "grid_channels_v2")
        self.assertEqual(PPOAgent.from_export(saved).export(), saved)
        saved["observation_encoding"] = "grid_channels_v1"
        with self.assertRaises(ValueError):
            PPOAgent.from_export(saved)



if __name__ == "__main__":
    unittest.main()
