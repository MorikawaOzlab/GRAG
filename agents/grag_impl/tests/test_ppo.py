"""PPOの終了処理、クリッピング、学習と再生の契約を検証する。"""

from copy import deepcopy
import importlib.util
import json
import math
import unittest
from unittest.mock import patch

from agents.grag import GRAG
from agents.grag_impl.grag import ACTIONS, _Transition, _compute_gae, _normalize_advantages, _value_loss
from gridworld.storage import DEFAULT_SETTINGS
from gridworld.training import train
from gridworld.world import GridWorld


MAP = {"name": "PPOテスト", "rows": 2, "cols": 3, "start": [0, 0],
       "goal": [1, 2], "obstacles": [[0, 1]]}
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class AdvantageTests(unittest.TestCase):
    def test_terminal_bootstrap_and_discounted_gae(self):
        trajectory = [
            _Transition((), 0, 0.0, 0.5, 0.0, 0.6, False),
            _Transition((), 1, 0.0, 0.6, 1.0, 100.0, True),
        ]
        advantages, returns = _compute_gae(trajectory, gamma=0.9, gae_lambda=0.8)
        self.assertAlmostEqual(advantages[1], 0.4)
        self.assertAlmostEqual(advantages[0], 0.328)
        self.assertAlmostEqual(returns[0], 0.828)
        self.assertAlmostEqual(returns[1], 1.0)

    def test_timeout_bootstraps_without_leaking_next_episode(self):
        trajectory = [
            _Transition((), 0, 0.0, 1.0, 0.0, 2.0, False, episode_end=True),
            _Transition((), 0, 0.0, 0.0, 10.0, 0.0, True),
        ]
        advantages, returns = _compute_gae(trajectory, gamma=0.9, gae_lambda=1.0)
        self.assertAlmostEqual(advantages[0], 0.8)
        self.assertAlmostEqual(returns[0], 1.8)
        self.assertAlmostEqual(returns[1], 10.0)

    def test_rollout_cutoff_bootstraps(self):
        trajectory = [_Transition((), 0, 0.0, 1.0, 0.0, 2.0, False)]
        advantages, returns = _compute_gae(trajectory, gamma=0.9, gae_lambda=0.95)
        self.assertAlmostEqual(advantages[0], 0.8)
        self.assertAlmostEqual(returns[0], 1.8)

    def test_invalid_settings_are_rejected(self):
        invalid = {
            "ppo_learning_rate": [0, -1, float("nan"), float("inf"), True],
            "ppo_gamma": [-0.1, 1.1, "0.9"],
            "ppo_gae_lambda": [-1, 2],
            "ppo_clip_range": [0, 1],
            "ppo_epochs": [0, 1.5, True],
            "ppo_minibatch_size": [0, -1],
            "ppo_rollout_steps": [0],
            "ppo_hidden_size": [0],
            "ppo_value_coef": [-1],
            "ppo_entropy_coef": [-1],
            "ppo_max_grad_norm": [0],
            "ppo_target_kl": [0, float("nan")],
            "ppo_normalize_advantage": [1],
            "ppo_clip_value_loss": [1, None, "true"],
            "ppo_anneal_lr": [1, None],
            "ppo_total_timesteps": [0, -1, True, 1.5],
            "seed": [-1, 2**32, True],
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        GRAG({name: value})

    def test_goal_methods_do_not_require_torch(self):
        observation = GridWorld(MAP).reset()
        with patch.dict("sys.modules", {"torch": None}):
            agent = GRAG({"goal_text": "ゴールに到達する"})
            self.assertFalse(agent.is_goal_reached(observation))
            self.assertEqual(agent.goal_for(observation).target, (1, 2))
            with self.assertRaisesRegex(RuntimeError, "PyTorch"):
                agent.choose_action(observation)


@unittest.skipUnless(TORCH_AVAILABLE, "ニューラルPPOのテストにはPyTorchが必要です。")
class NeuralPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.previous_threads)

    def agent(self, **overrides):
        return GRAG({
            "seed": 7, "ppo_hidden_size": 16, "ppo_epochs": 2,
            "ppo_rollout_steps": 8, "ppo_minibatch_size": 4,
            "ppo_target_kl": None, **overrides})

    def test_model_input_and_observations_are_preserved(self):
        observation = GridWorld(MAP).reset()
        before = deepcopy(observation)
        agent = self.agent()
        action = agent.choose_action(observation)
        self.assertIn(action, ACTIONS)
        self.assertEqual(observation, before)
        self.assertEqual(agent._model.actor[0].in_features, 3 * 2 * 3 + 4)
        self.assertEqual(agent._model.actor[-1].out_features, 4)
        self.assertEqual(agent._model.critic[-1].out_features, 1)

    def test_real_grid_collision_and_goal_keep_environment_rewards(self):
        world = GridWorld(MAP)
        observation = world.reset()
        agent = self.agent(ppo_rollout_steps=16)
        actions = ("up", "right", "down", "right", "right")
        expected_collisions = (True, True, False, False, False)
        for expected_action, expected_collision in zip(actions, expected_collisions):
            before = deepcopy(observation)
            # Draw a known action while retaining the real policy/log-probability path.
            with patch.object(self.torch, "multinomial", return_value=self.torch.tensor(
                    [ACTIONS.index(expected_action)])):
                action = agent.choose_action(observation)
            next_observation, reached_goal, collision = world.step(action)
            agent.learn(observation, action, float(reached_goal), next_observation, reached_goal)
            self.assertEqual(observation, before)
            self.assertEqual(collision, expected_collision)
            if collision:
                self.assertEqual(next_observation["position"], observation["position"])
            observation = next_observation
        self.assertEqual(observation["position"], MAP["goal"])
        self.assertEqual([item.reward for item in agent._rollout], [0, 0, 0, 0, 1])
        self.assertEqual([item.terminated for item in agent._rollout],
                         [False, False, False, False, True])
        self.assertEqual(agent._rollout[-1].next_value, 0)
        agent.end_episode()
        self.assertEqual(agent._steps, 5)
        self.assertEqual(agent._updates, 1)
        self.assertTrue(all(math.isfinite(value) for value in agent._last_update.values()))

    def test_grid_channels_preserve_row_column_coordinates(self):
        observation = GridWorld(MAP).reset()
        before = deepcopy(observation)
        shape, features = self.agent()._features(observation)
        self.assertEqual(shape, (2, 3))
        self.assertEqual(features[:6], (0, 1, 0, 0, 0, 0))
        self.assertEqual(features[6:12], (0, 0, 0, 0, 0, 1))
        self.assertEqual(features[12:18], (1, 0, 0, 0, 0, 0))
        self.assertEqual(features[18:], (0, 0, 1, 1))
        self.assertEqual(observation, before)

    def test_value_clipping_limits_improvement_and_preserves_corrective_gradient(self):
        for target, expected_loss, expected_gradient in ((4.0, 6.125, 0.0),
                                                          (0.0, 4.5, 3.0)):
            with self.subTest(target=target):
                values = self.torch.tensor([3.0], requires_grad=True)
                loss = _value_loss(values, self.torch.tensor([target]),
                                   self.torch.tensor([0.0]), clip_range=0.5, clip_value=True)
                self.assertAlmostEqual(loss.item(), expected_loss)
                loss.backward()
                self.assertAlmostEqual(values.grad.item(), expected_gradient)
        loss = _value_loss(self.torch.tensor([3.0]), self.torch.tensor([4.0]),
                           self.torch.tensor([0.0]), clip_range=0.5, clip_value=False)
        self.assertAlmostEqual(loss.item(), 0.5)

    def test_minibatch_normalization_handles_single_and_constant_advantages(self):
        normalized = _normalize_advantages(self.torch.tensor([1.0, 3.0]))
        self.assertAlmostEqual(normalized.mean().item(), 0.0)
        self.assertAlmostEqual(normalized.std().item(), 1.0)
        single = _normalize_advantages(self.torch.tensor([2.0]))
        self.assertEqual(single.tolist(), [2.0])
        constant = _normalize_advantages(self.torch.tensor([2.0, 2.0]))
        self.assertEqual(constant.tolist(), [0.0, 0.0])

    def test_learning_rate_schedule_resumes_at_same_step(self):
        agent = self.agent(ppo_anneal_lr=True, ppo_total_timesteps=8, ppo_rollout_steps=2)
        observation = GridWorld(MAP).reset()

        def collect(policy, steps):
            for _ in range(steps):
                action = policy.choose_action(observation)
                policy.learn(observation, action, 0, observation, False)

        collect(agent, 2)
        rate = agent.export()["settings"]["ppo_learning_rate"]
        self.assertAlmostEqual(agent._last_update["learning_rate"], rate)
        checkpoint = json.loads(json.dumps(agent.checkpoint(), allow_nan=False))
        restored = GRAG(agent.settings)
        restored.load_checkpoint(checkpoint)
        collect(agent, 2)
        collect(restored, 2)
        self.assertEqual(restored.checkpoint(), agent.checkpoint())
        self.assertAlmostEqual(agent._last_update["learning_rate"], 0.75 * rate)

    def test_positive_advantage_increases_sampled_action_probability(self):
        agent = self.agent(ppo_epochs=1, ppo_learning_rate=0.03,
                           ppo_entropy_coef=0, ppo_value_coef=0)
        observation = GridWorld(MAP).reset()
        agent._initialize_model((2, 3))
        with self.torch.no_grad():
            for parameter in agent._model.critic.parameters():
                parameter.zero_()
        action = agent.choose_action(observation)
        before = math.exp(agent._pending[2])
        agent.learn(observation, action, 1, {**observation, "position": MAP["goal"]}, True)
        agent.end_episode()
        _, features = agent._features(observation)
        with self.torch.no_grad():
            logits, _ = agent._model(self.torch.tensor(features).unsqueeze(0))
            after = logits.softmax(-1)[0, ACTIONS.index(action)].item()
        self.assertGreater(after, before)

    def test_clipping_blocks_only_improving_outside_ratio_updates(self):
        # Positive advantage clips high ratios; negative advantage clips low ratios.
        for reward, ratio in ((1, 2.0), (-1, 0.5), (1, 0.5), (-1, 2.0)):
            with self.subTest(reward=reward, ratio=ratio):
                agent = self.agent(ppo_epochs=1, ppo_entropy_coef=0,
                                   ppo_value_coef=0, ppo_normalize_advantage=False)
                observation = GridWorld(MAP).reset()
                agent._initialize_model((2, 3))
                with self.torch.no_grad():
                    for parameter in agent._model.critic.parameters():
                        parameter.zero_()
                action = agent.choose_action(observation)
                agent.learn(observation, action, reward,
                            {**observation, "position": MAP["goal"]}, True)
                agent._rollout[0].log_probability -= math.log(ratio)
                before = {name: value.clone()
                          for name, value in agent._model.actor.state_dict().items()}
                agent.end_episode()
                changed = any(not self.torch.equal(value, before[name])
                              for name, value in agent._model.actor.state_dict().items())
                should_clip = (reward > 0 and ratio > 1.2) or (reward < 0 and ratio < 0.8)
                self.assertEqual(changed, not should_clip)

    def test_rollout_limit_and_episode_tail_are_updated(self):
        agent = self.agent(ppo_rollout_steps=2)
        observation = GridWorld(MAP).reset()
        for _ in range(5):
            action = agent.choose_action(observation)
            agent.learn(observation, action, 0, observation, False)
            self.assertLess(len(agent._rollout), 2)
        self.assertEqual(agent._updates, 2)
        agent.end_episode()
        self.assertEqual(agent._updates, 3)
        self.assertEqual(agent._last_update["samples"], 1)
        self.assertEqual(agent._steps, 5)
        self.assertEqual(agent._episodes, 1)
        self.assertEqual(agent._rollout, [])

    def test_greedy_does_not_change_model_rng_or_training_buffers(self):
        agent = self.agent()
        observation = GridWorld(MAP).reset()
        action = agent.choose_action(observation)
        agent.learn(observation, action, 0, observation, False)
        agent.end_episode()
        before = agent.export()
        generator_state = agent._generator.get_state().clone()
        actions = [agent.choose_action(observation, greedy=True) for _ in range(5)]
        self.assertEqual(len(set(actions)), 1)
        self.assertEqual(agent.export(), before)
        self.assertTrue(self.torch.equal(agent._generator.get_state(), generator_state))
        self.assertIsNone(agent._pending)
        self.assertEqual(agent._rollout, [])

    def test_initialization_and_sampling_leave_global_torch_rng_unchanged(self):
        self.torch.manual_seed(1234)
        before = self.torch.random.get_rng_state().clone()
        self.agent().choose_action(GridWorld(MAP).reset())
        self.assertTrue(self.torch.equal(before, self.torch.random.get_rng_state()))

    def test_seed_reproduces_training(self):
        def run():
            return train(deepcopy(MAP), {
                **DEFAULT_SETTINGS, "seed": 11, "episodes": 3, "max_steps": 5,
                "ppo_hidden_size": 16, "ppo_epochs": 2, "ppo_minibatch_size": 4,
            }, agent_type=GRAG)
        self.assertEqual(run(), run())

    def test_export_round_trip_restores_model_and_greedy_policy(self):
        agent = self.agent()
        observation = GridWorld(MAP).reset()
        action = agent.choose_action(observation)
        agent.learn(observation, action, 1, {**observation, "position": MAP["goal"]}, True)
        agent.end_episode()
        data = json.loads(json.dumps(agent.export(), allow_nan=False))
        restored = GRAG.from_export(data)
        self.assertEqual(restored.export(), data)
        for position in ([0, 0], [1, 0], [1, 1]):
            state = {**observation, "position": position}
            self.assertEqual(restored.choose_action(state, greedy=True),
                             agent.choose_action(state, greedy=True))
        self.assertEqual(GRAG.from_export(self.agent().export())._model, None)

    def test_map_shape_and_on_policy_contract_are_enforced(self):
        observation = GridWorld(MAP).reset()
        agent = self.agent()
        with self.assertRaises(RuntimeError):
            agent.learn(observation, "up", 0, observation, False)
        action = agent.choose_action(observation)
        with self.assertRaises(RuntimeError):
            agent.choose_action(observation)
        with self.assertRaises(RuntimeError):
            agent.end_episode()
        with self.assertRaises(ValueError):
            agent.learn(observation, ACTIONS[(ACTIONS.index(action) + 1) % 4],
                        0, observation, False)
        with self.assertRaises(ValueError):
            agent.learn(observation, action, float("nan"), observation, False)
        agent.learn(observation, action, 0, observation, False)
        agent.end_episode()
        with self.assertRaises(ValueError):
            agent.choose_action({**observation, "rows": 3}, greedy=True)

    def test_small_grid_learns_and_replays_to_goal(self):
        small_map = {"name": "隣接ゴール", "rows": 2, "cols": 2,
                     "start": [0, 0], "goal": [0, 1], "obstacles": []}
        result = train(small_map, {
            **DEFAULT_SETTINGS, "seed": 7, "episodes": 200, "max_steps": 12,
            "ppo_learning_rate": 0.003, "ppo_hidden_size": 16, "ppo_epochs": 4,
            "ppo_minibatch_size": 16,
        }, agent_type=GRAG)
        self.assertEqual(len(result["history"]), 200)
        self.assertGreaterEqual(sum(item["success"] for item in result["history"][-50:]), 40)
        self.assertTrue(result["frames"][-1]["reached_goal"])
        self.assertEqual(result["q_table"]["algorithm"], "PPO")
        self.assertGreater(result["q_table"]["training"]["updates"], 0)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
