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

    def test_legacy_numeric_json_checkpoint_can_still_be_restored(self):
        session = self.session()
        session.run_batch(4)
        agent = session.agent

        def legacy(value):
            if isinstance(value, torch.Tensor):
                return {"dtype": str(value.dtype).split(".")[-1], "values": value.detach().cpu().tolist()}
            if isinstance(value, dict):
                return {key: legacy(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [legacy(item) for item in value]
            return value

        with patch.object(agent, '_encode', side_effect=legacy):
            checkpoint = agent.checkpoint()
        checkpoint['model']['schema_version'] = 1
        restored = CPPOAgent(SETTINGS)
        restored.load_checkpoint(json.loads(json.dumps(checkpoint)))
        self.assertEqual(restored.checkpoint(), agent.checkpoint())
        self.assertIsNone(session.result(checkpoint=True)['q_table'])

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
        self.assertGreater(len(session.evaluate()), 1)
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
        agent = self.session().agent
        states = torch.tensor([agent._state({**MAP, "position": [0, 0]})])
        with torch.no_grad():
            q = agent._all_q(states)
            self.assertTrue((q <= 0).all())
            probabilities = agent._distribution(states).probs
            advantages = torch.cat([agent._advantages(states, torch.tensor([a])) for a in range(4)])
            self.assertAlmostEqual(float((probabilities[0] * advantages).sum()), 0., places=6)
            self.assertTrue(torch.allclose(advantages, q[0] - (probabilities * q).sum()))

    def test_hindsight_does_not_cross_episode_boundaries(self):
        agent = self.session().agent
        agent.episodes = [[{"achieved": [i, j]} for j in range(4)] for i in range(3)]
        _, goals, ids = agent._hindsight()
        for index, (goal, episode_id) in enumerate(zip(goals, ids)):
            self.assertEqual(int(goal[0]), int(episode_id))
            self.assertGreaterEqual(int(goal[1]), index % 4)

    def test_invalid_parameters_fail_before_training(self):
        for key, value in (("gamma", 1), ("learning_rate", float("nan")),
                           ("batch_size", 1), ("epochs", True), ("clip_coef", 0),
                           ("normalize_advantage", "false")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                CPPOAgent({**SETTINGS, "cppo_" + key: value})

    def test_default_collection_is_total_across_paper_environments(self):
        agent = CPPOAgent(DEFAULT_SETTINGS)
        self.assertEqual(agent.config["rollout_steps"], 128 * 512)
        self.assertEqual(agent.config["batch_size"], 256)
        self.assertTrue(agent.config["normalize_advantage"])

    def test_chunked_advantages_match_full_discrete_expectation(self):
        agent = self.session().agent
        states = torch.tensor([agent._state({**MAP, "position": position})
                               for position in ([0, 0], [1, 0], [1, 1]) * 8])
        actions = torch.arange(len(states)) % 4
        with torch.no_grad():
            q = agent._all_q(states)
            expected = q.gather(1, actions[:, None]).squeeze(1) - (agent._distribution(states).probs * q).sum(-1)
        self.assertTrue(torch.allclose(agent._advantages(states, actions), expected, atol=1e-6))

    def test_metrics_distinguish_collection_and_policy_collapse(self):
        session = self.session()
        session.run_batch(4)
        metrics = session.agent.metrics
        self.assertGreaterEqual(metrics["rollout_episodes"], 2)
        self.assertGreaterEqual(metrics["rollout_steps"], SETTINGS["cppo_rollout_steps"])
        self.assertGreater(metrics["unique_achieved_goals"], 0)
        self.assertIsInstance(metrics["target_seen_in_rollout"], bool)
        self.assertGreaterEqual(metrics["policy_kl"], -1e-6)
        self.assertGreaterEqual(metrics["policy_entropy"], 0)
        self.assertLessEqual(metrics["policy_entropy"], 1.386295)


if __name__ == "__main__":
    unittest.main()
