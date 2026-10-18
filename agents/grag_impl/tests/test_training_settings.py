"""GRAGの画面設定、乱数再現性、試行末尾のPPO更新を検証する。"""

import unittest
from unittest.mock import patch

import torch

from agents.grag import GRAG
from gridworld.storage import DEFAULT_SETTINGS
from gridworld.training_runtime import TrainingSession
from gridworld.world import GridWorld


MAP = {"name": "GRAG regression", "rows": 2, "cols": 3,
       "start": [0, 0], "goal": [1, 2], "obstacles": [[0, 1]]}


class TrainingSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_ui_settings_configure_networks_and_optimizer(self):
        agent = GRAG({**DEFAULT_SETTINGS, "ppo_learning_rate": 0.0003,
                      "ppo_hidden_size": 8})
        agent.initialize_networks(GridWorld(MAP).reset())
        self.assertEqual(agent.optimizer.param_groups[0]["lr"], 0.0003)
        self.assertEqual(agent.gamma, DEFAULT_SETTINGS["discount_factor"])
        for network in (agent.actor, agent.critic):
            self.assertEqual(network[1].out_features, 8)
            self.assertEqual(network[3].in_features, 8)
            self.assertEqual(network[3].out_features, 8)

    def test_q_learning_rate_is_not_used_as_ppo_default(self):
        self.assertEqual(GRAG(DEFAULT_SETTINGS).learning_rate, 2.5e-4)

    def test_ppo_settings_override_legacy_names(self):
        agent = GRAG({"ppo_gamma": 0.8, "gamma": 0.9, "discount_factor": 0.95,
                      "ppo_gae_lambda": 0.7, "gae_lambda": 0.8,
                      "ppo_clip_range": 0.1, "clip_coef": 0.2,
                      "ppo_entropy_coef": 0.02, "ent_coef": 0.01,
                      "ppo_value_coef": 0.4, "vf_coef": 0.5,
                      "ppo_epochs": 2, "update_epochs": 4,
                      "ppo_minibatch_size": 3, "minibatch_size": 64,
                      "ppo_rollout_steps": 5, "rollout_steps": 128})
        self.assertEqual((agent.gamma, agent.gae_lambda, agent.clip_coef,
                          agent.ent_coef, agent.vf_coef, agent.update_epochs,
                          agent.minibatch_size, agent.rollout_steps),
                         (0.8, 0.7, 0.1, 0.02, 0.4, 2, 3, 5))

    def test_seed_reproduces_interleaved_learning_without_global_rng_changes(self):
        settings = {**DEFAULT_SETTINGS, "seed": 7, "max_steps": 3,
                    "ppo_hidden_size": 8, "ppo_epochs": 2,
                    "ppo_rollout_steps": 2, "ppo_minibatch_size": 1}
        global_rng = torch.random.get_rng_state().clone()
        sessions = [TrainingSession(MAP, settings, GRAG) for _ in range(2)]
        for _ in range(8):
            self.assertEqual(sessions[0].step(), sessions[1].step())
        for name in ("actor", "critic"):
            first = getattr(sessions[0].agent, name).state_dict()
            second = getattr(sessions[1].agent, name).state_dict()
            for key in first:
                self.assertTrue(torch.equal(first[key], second[key]), key)
        self.assertTrue(torch.equal(global_rng, torch.random.get_rng_state()))

    def test_final_short_episode_updates_weights_and_clears_memory(self):
        settings = {**DEFAULT_SETTINGS, "seed": 7, "max_steps": 1,
                    "ppo_hidden_size": 8, "ppo_epochs": 1,
                    "ppo_rollout_steps": 128}
        session = TrainingSession(MAP, settings, GRAG)
        agent = session.agent
        agent.initialize_networks(session.world.observation())
        before = [p.detach().clone() for p in agent.critic.parameters()]
        session.run_batch(1)
        self.assertFalse(agent.memory)
        self.assertTrue(any(not torch.equal(a, b)
                            for a, b in zip(before, agent.critic.parameters())))
        self.assertTrue(agent.optimizer.state)
        self.assertTrue(all(state["step"].item() == 1
                            for state in agent.optimizer.state.values()))
        agent.end_episode()
        self.assertTrue(all(state["step"].item() == 1
                            for state in agent.optimizer.state.values()))

    def test_end_episode_marks_boundary_and_preserves_time_limit_bootstrap(self):
        agent = GRAG({"seed": 7, "ppo_gamma": 0.95, "ppo_epochs": 1})
        observation = GridWorld(MAP).reset()
        agent.initialize_networks(observation)
        with torch.no_grad():
            for parameter in agent.critic.parameters():
                parameter.zero_()
            agent.critic[-1].bias.fill_(2.0)
        action = agent.choose_action(observation)
        agent.learn(observation, action, 0.0, observation, False)
        compute = agent._compute_gae
        captured = []

        def capture_targets():
            self.assertTrue(agent.memory[-1].episode_end)
            self.assertEqual(agent.memory[-1].next_value, 2.0)
            result = compute()
            captured.append(result)
            return result

        with patch.object(agent, "_compute_gae", side_effect=capture_targets):
            agent.end_episode()
        self.assertAlmostEqual(captured[0][0].item(), -0.1, places=6)
        self.assertAlmostEqual(captured[0][1].item(), 1.9, places=6)
        self.assertFalse(agent.memory)


if __name__ == "__main__":
    unittest.main()
