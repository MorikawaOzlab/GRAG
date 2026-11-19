import unittest

import torch

from agents.grag_impl.grag import GRAG
from agents.grag_impl.gridworld_adapter import GridWorldAdapter


class GenericActionTests(unittest.TestCase):
    def test_custom_observation_and_actions(self):
        agent = GRAG({"seed": 7, "actions": (10, 20, 30),
                      "observation_encoder": lambda obs: [obs["x"], obs["velocity"]]})
        observation = {"x": 0.5, "velocity": -1}
        agent.initialize_networks(2, 1, 3)
        with torch.no_grad():
            agent.actor[-1].weight.zero_()
            agent.actor[-1].bias.copy_(torch.tensor([0., 3., 1.]))
        self.assertEqual(agent.choose_action(observation, greedy=True), 20)
        self.assertEqual(observation, {"x": 0.5, "velocity": -1})

    def test_seed_and_shape(self):
        a, b = (GRAG({"seed": 3, "actions": (0, 1)}) for _ in range(2))
        with self.assertRaises(RuntimeError):
            a.choose_action([1, 2])
        a.initialize_networks(2, 1, 2)
        b.initialize_networks(2, 1, 2)
        def sample(agent):
            actions = []
            for _ in range(20):
                action = agent.choose_action([1, 2])
                actions.append(action)
                agent.learn([1, 2], action, 0, [1, 2], False)
            return actions
        self.assertEqual(sample(a), sample(b))
        with self.assertRaises(ValueError):
            a.choose_action([1, 2, 3])

    def test_gridworld_compatibility(self):
        agent = GRAG({"seed": 1})
        observation = {"rows": 2, "cols": 3, "obstacles": [[0, 1]],
                       "goal": [1, 2], "position": [0, 0]}
        agent.initialize_for_gridworld(observation)
        self.assertIn(agent.choose_action(observation, greedy=True),
                      ("up", "down", "left", "right"))

    def test_problem_adapter_initializes_and_encodes(self):
        observation = {"rows": 2, "cols": 3, "obstacles": [[0, 1]],
                       "goal": [1, 2], "position": [0, 0]}
        adapter = GridWorldAdapter(observation)
        state = adapter.encode_observation(observation)
        self.assertEqual(tuple(state.shape), (3, 2, 3))
        self.assertEqual(state.sum().item(), 3)
        self.assertEqual(state[0, 0, 1].item(), 1)
        self.assertEqual(state[1, 1, 2].item(), 1)
        self.assertEqual(state[2, 0, 0].item(), 1)
        agent = GRAG({"seed": 1})
        agent.initialize_networks(problem=adapter)
        self.assertEqual(agent._network_spec, (18, 2, 4))
        self.assertIn(agent.choose_action(observation), agent.actions)
        with self.assertRaises(ValueError):
            adapter.encode_observation({**observation, "rows": 3})

    def test_other_problem_adapter(self):
        class Problem:
            def network_dimensions(self, action_dim):
                return 2, 1, action_dim

            def encode_observation(self, observation):
                return [observation["x"], observation["y"]]

        agent = GRAG({"actions": (0, 1), "seed": 1})
        agent.initialize_networks(problem=Problem())
        self.assertEqual(agent._network_spec, (2, 1, 2))
        self.assertIn(agent.choose_action({"x": 1, "y": 2}), (0, 1))

    def test_initialization_preserves_weights_and_rng(self):
        agent = GRAG({"seed": 4, "actions": (0, 1)})
        rng = torch.random.get_rng_state().clone()
        agent.initialize_networks(2, 1, 2)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        actor = agent.actor
        agent.initialize_networks(2, 1, 2)
        self.assertIs(agent.actor, actor)
        with self.assertRaises(ValueError):
            agent.initialize_networks(3, 1, 2)

    def test_probabilities_match_policy_without_consuming_randomness(self):
        agent = GRAG({"seed": 7, "actions": (10, 20, 30)})
        with self.assertRaises(RuntimeError):
            agent.action_probabilities([1, 2])
        agent.initialize_networks(2, 1, 3)
        with torch.no_grad():
            agent.actor[-1].weight.zero_()
            agent.actor[-1].bias.copy_(torch.tensor([0., 1., 2.]))
        rng = agent._generator.get_state().clone()
        global_rng = torch.random.get_rng_state().clone()
        probabilities = agent.action_probabilities([1, 2])
        expected = torch.softmax(torch.tensor([0., 1., 2.]), dim=0)
        torch.testing.assert_close(torch.tensor(list(probabilities.values())), expected)
        self.assertEqual(list(probabilities), [10, 20, 30])
        self.assertEqual(agent.choose_action([1, 2], greedy=True), 30)
        self.assertTrue(torch.equal(rng, agent._generator.get_state()))
        self.assertTrue(torch.equal(global_rng, torch.random.get_rng_state()))
        with self.assertRaises(ValueError):
            agent.action_probabilities([1, 2, 3])


if __name__ == "__main__":
    unittest.main()
