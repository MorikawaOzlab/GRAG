"""SGCRL equations, temporal sampling, inference and resumable training."""
import json
import math
import unittest
from unittest.mock import patch

import torch

from agents.grag import GRAG
from agents.grag_impl.grag import ACTIONS, actor_loss, contrastive_loss
from gridworld.storage import DEFAULT_SETTINGS
from gridworld.training_runtime import TrainingSession
from gridworld.world import GridWorld

MAP = {"name": "sgcrl", "rows": 2, "cols": 3,
       "start": [0, 0], "goal": [1, 2], "obstacles": []}
SETTINGS = {**DEFAULT_SETTINGS, "seed": 7, "max_steps": 4,
            "crl_device": "cpu",
            "crl_hidden_size": 16, "crl_repr_dim": 8,
            "crl_batch_size": 8, "crl_min_replay_size": 2}


class SingleGoalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def agent(self, **overrides):
        agent = GRAG({**SETTINGS, **overrides})
        agent.initialize_networks(GridWorld(MAP).reset())
        return agent

    def trajectory(self, agent, actions=("down", "up"), reward=0):
        world = GridWorld(MAP)
        obs = world.reset()
        for action in actions:
            nxt, reached, _ = world.step(action)
            agent.learn(obs, action, reward, nxt, reached)
            obs = nxt
        agent.end_episode()

    def test_critic_equation_matches_independent_scalar_calculation(self):
        scores = torch.tensor([[1., -1.], [0.5, 2.]], dtype=torch.float64, requires_grad=True)
        expected = sum(-row[i] + math.log(sum(math.exp(x) for x in row))
                       + 0.01 * math.log(sum(math.exp(x) for x in row)) ** 2
                       for i, row in enumerate(scores.detach().tolist())) / 2
        loss = contrastive_loss(scores)
        self.assertAlmostEqual(loss.item(), expected, places=12)
        self.assertTrue(torch.autograd.gradcheck(contrastive_loss, (scores,)))

    def test_actor_expectation_and_gradient_improve_best_action(self):
        logits = torch.zeros(1, 4, requires_grad=True)
        scores = torch.tensor([[0., 3., 1., -1.]], requires_grad=True)
        loss, entropy = actor_loss(logits, scores, torch.tensor(0.))
        self.assertAlmostEqual(loss.item(), -0.75)
        self.assertAlmostEqual(entropy.item(), math.log(4), places=6)
        loss.backward()
        self.assertEqual(logits.grad.argmin().item(), 1)
        self.assertIsNone(scores.grad)

    def test_updates_before_any_success_and_reward_does_not_affect_learning(self):
        first, second = self.agent(), self.agent()
        before = {name: [p.detach().clone() for p in getattr(first, name).parameters()]
                  for name in ("actor", "sa_encoder", "goal_encoder")}
        self.trajectory(first, reward=0)
        self.trajectory(second, reward=999)
        self.assertEqual(first.checkpoint(), second.checkpoint())
        self.assertGreater(first._updates, 0)
        self.assertTrue(all(t["next_state"] != MAP["goal"] for ep in first._episodes for t in ep))
        for name, old in before.items():
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(old, getattr(first, name).parameters())), name)

    def test_actor_trains_on_future_and_marginal_goals(self):
        agent = self.agent()
        states = torch.tensor([[0., 0.], [1., 0.]])
        actions = torch.tensor([1, 3])
        goals = torch.tensor([[1., 0.], [1., 1.]])
        with patch.object(agent, "_sample_batch", return_value=(states, actions, goals)), \
                patch.object(agent.actor, "forward", wraps=agent.actor.forward) as forward:
            agent._update()
        seen = forward.call_args.args[0]
        self.assertTrue(torch.equal(seen[:, :2], torch.cat([states, states])))
        self.assertTrue(torch.equal(seen[:, 2:], torch.cat([goals, goals.roll(1, 0)])))
        self.assertFalse((seen[:, 2:] == torch.tensor(MAP["goal"])).all(dim=-1).any())

    def test_collection_uses_final_goal_and_greedy_uses_actor(self):
        agent = self.agent()
        obs = GridWorld(MAP).reset()
        obs["position"] = [1, 0]
        with patch.object(agent.actor, "forward", return_value=torch.tensor([0., 1., 4., 2.])) as forward:
            rng = agent._generator.get_state().clone()
            self.assertEqual(agent.choose_action(obs, greedy=True), "left")
            self.assertTrue(torch.equal(rng, agent._generator.get_state()))
            self.assertEqual(forward.call_args.args[0].tolist(), [1., 0., 1., 2.])

    def test_future_sampling_is_geometric_and_does_not_cross_episodes(self):
        agent = self.agent(crl_gamma=0.5, crl_batch_size=12000)
        # Different coordinate ranges identify episode boundaries unambiguously.
        agent._episodes = [
            [{"state": [i, 0], "action": 1, "next_state": [i + 1, 0]} for i in range(4)],
            [{"state": [i, 1], "action": 1, "next_state": [i + 1, 1]} for i in range(10, 12)]]
        agent._replay_size = 6
        states, _, goals = agent._sample_batch()
        self.assertTrue((goals[:, 1] == states[:, 1]).all())
        self.assertTrue((goals[:, 0] > states[:, 0]).all())
        selected = goals[states[:, 0] == 0, 0]
        counts = torch.bincount(selected.long(), minlength=5)[1:].float() / len(selected)
        self.assertTrue(torch.allclose(counts, torch.tensor([8/15, 4/15, 2/15, 1/15]), atol=0.04))

    def test_collisions_and_revisits_are_retained(self):
        agent = self.agent()
        self.trajectory(agent, ("left", "left", "down", "up"))
        self.assertEqual(agent._replay_size, 4)
        states, _, goals = agent._sample_batch()
        self.assertTrue(any(t["state"] == t["next_state"] for t in agent._episodes[0]))
        self.assertEqual(len(states), agent.batch_size)

    def test_resume_mid_episode_matches_uninterrupted_learning_exactly(self):
        session = TrainingSession(MAP, SETTINGS, GRAG)
        for _ in range(5):
            session.step()
        saved = json.loads(json.dumps(session.result(checkpoint=True), allow_nan=False))
        restored = TrainingSession.restore(saved, GRAG)
        for _ in range(10):
            self.assertEqual(session.step(), restored.step())
        self.assertEqual(session.agent.checkpoint(), restored.agent.checkpoint())

    def test_exported_inference_and_evaluation_leave_training_unchanged(self):
        session = TrainingSession(MAP, SETTINGS, GRAG)
        session.run_batch(1)
        before = session.agent.checkpoint()
        restored = GRAG.from_export(json.loads(json.dumps(session.agent.export())))
        obs = session.world.observation()
        self.assertEqual(session.agent.action_probabilities(obs), restored.action_probabilities(obs))
        self.assertEqual(session.agent.choose_action(obs, greedy=True), restored.choose_action(obs, greedy=True))
        session.evaluate()
        self.assertEqual(before, session.agent.checkpoint())

    def test_exported_trained_model_samples_actor_instead_of_initial_uniform(self):
        agent = self.agent()
        self.trajectory(agent)
        restored = GRAG.from_export(agent.export())
        obs = GridWorld(MAP).reset()
        with patch("torch.multinomial", return_value=torch.tensor([0])) as sample:
            restored.choose_action(obs)
        probabilities = torch.tensor(list(restored.action_probabilities(obs).values()))
        self.assertTrue(torch.equal(sample.call_args.args[0], probabilities))

    def test_uninitialized_checkpoint_with_random_seed_restores_initialization(self):
        agent = GRAG({**SETTINGS, "seed": None})
        restored = GRAG({**SETTINGS, "seed": None})
        restored.load_checkpoint(json.loads(json.dumps(agent.checkpoint())))
        obs = GridWorld(MAP).reset()
        self.assertEqual(agent.choose_action(obs), restored.choose_action(obs))
        self.assertEqual(agent.checkpoint(), restored.checkpoint())

    def test_short_episode_and_empty_end_do_not_double_update(self):
        agent = self.agent()
        self.trajectory(agent)
        before = agent.checkpoint()
        agent.end_episode()
        self.assertEqual(before, agent.checkpoint())

    def test_alpha_decreases_when_entropy_exceeds_target(self):
        agent = self.agent()
        self.trajectory(agent)
        self.assertLess(agent._log_alpha.item(), 0)

    def test_replay_capacity_keeps_future_states_in_their_episode(self):
        agent = self.agent(crl_replay_capacity=3)
        self.trajectory(agent)
        self.trajectory(agent)
        self.assertEqual(agent._replay_size, 3)
        self.assertEqual([len(ep) for ep in agent._episodes], [1, 2])
        self.assertEqual(agent._episodes[0][0]["state"], [1, 0])

    def test_old_checkpoints_are_rejected(self):
        with self.assertRaises(ValueError):
            GRAG.from_export({"schema_version": 2})
        with self.assertRaises(ValueError):
            self.agent().load_checkpoint({"schema_version": 1})


if __name__ == "__main__":
    unittest.main()
