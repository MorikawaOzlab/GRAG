import unittest
from unittest.mock import patch

import torch

from agents.grag_impl.grag import GRAG
from agents.grag_impl.gridworld_adapter import GridWorldAdapter


class ContrastiveUpdateTests(unittest.TestCase):
    def test_goal_groups_average_losses_and_update_once_within_one_episode(self):
        agent = GRAG({"seed": 1, "actions": (0, 1)})
        agent.initialize_networks(1, 1)
        batch = [
            {"state": torch.tensor([float(i)]), "action": i % 2, "episode_id": 0,
             "next_state": torch.tensor([goal])}
            for i, goal in ((0, 1.), (1, 2.), (0, 2.), (2, 3.))
        ]
        reference = next(agent.sa_encoder.parameters())
        sa = torch.tensor([[0., 1., 0.], [1., 0., 1.], [2., 1., 0.]]).to(reference)
        unique_goals = torch.tensor([[1.], [2.], [3.]]).to(reference)
        with torch.no_grad():
            logits = -torch.cdist(agent.goal_encoder(unique_goals), agent.sa_encoder(sa))
            log_probs = logits.log_softmax(dim=1)
            expected_loss = -(log_probs[0, 0] + log_probs[1, :2].mean() + log_probs[2, 2]) / 3

        with patch("torch.multinomial", return_value=torch.tensor([0])):
            with patch.object(agent.contrastive_optimizer, "step", wraps=agent.contrastive_optimizer.step) as step:
                self.assertTrue(agent._update_contrastive(batch))
                self.assertEqual(step.call_count, 1)

        self.assertAlmostEqual(agent.contrastive_loss, expected_loss.item(), places=6)
        self.assertEqual(agent._contrastive_pending, [])

    def test_contrastive_epochs_reuse_immediate_goal(self):
        agent = GRAG({"seed": 1, "actions": (0, 1), "grag_contrastive_epochs": 3})
        agent.initialize_networks(1, 1)
        batch = [{"episode_id": 0, "state": torch.tensor([1.]), "action": 0,
                  "next_state": torch.tensor([3.])},
                 {"episode_id": 0, "state": torch.tensor([2.]), "action": 1,
                  "next_state": torch.tensor([4.])}]
        with patch.object(agent.goal_encoder, "forward", wraps=agent.goal_encoder.forward) as encode, \
             patch.object(agent.contrastive_optimizer, "step", wraps=agent.contrastive_optimizer.step) as step:
            self.assertTrue(agent._update_contrastive(batch))
        self.assertEqual([call.args[0][:, 0].tolist() for call in encode.call_args_list], [[3., 4.]] * 3)
        self.assertEqual(step.call_count, 3)

    def test_invalid_contrastive_epochs_are_rejected(self):
        for value in (0, -1, True, 1.5, 1001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GRAG({"grag_contrastive_epochs": value})

    def test_initial_actor_update_uses_preupdate_critic(self):
        agent = GRAG({"seed": 1, "actions": (0, 1),
                      "target_goal_encoder": lambda obs: [0.]})
        agent.initialize_networks(1, 1)
        transitions = [{"state": torch.tensor([float(i)]), "observation": [float(i)],
                        "action": i % 2, "log_prob": -0.693147,
                        "episode_id": i, "next_state": torch.tensor([float(i + 1)])}
                       for i in range(2)]
        events = []
        original_q = agent.q_values
        original_actor_step = agent.actor_optimizer.step
        original_critic_step = agent.contrastive_optimizer.step
        def q_values(*args):
            events.append("q")
            return original_q(*args)
        def actor_step(*args, **kwargs):
            events.append("actor")
            return original_actor_step(*args, **kwargs)
        def critic_step(*args, **kwargs):
            events.append("critic")
            return original_critic_step(*args, **kwargs)
        with patch.object(agent, "q_values", side_effect=q_values), \
             patch.object(agent.actor_optimizer, "step", side_effect=actor_step), \
             patch.object(agent.contrastive_optimizer, "step", side_effect=critic_step):
            agent._update_networks(transitions)
        self.assertEqual(events, ["q", "actor", "actor", "actor", "actor", "critic"])

    def test_infonce_uses_dot_products(self):
        agent = GRAG({"seed": 1, "actions": (0, 1)})
        agent.initialize_networks(1, 1)
        batch = [{"episode_id": 0, "state": torch.tensor([1.]), "action": 0,
                  "next_state": torch.tensor([3.])},
                 {"episode_id": 1, "state": torch.tensor([2.]), "action": 1,
                  "next_state": torch.tensor([4.])}]
        with patch.object(agent.sa_encoder, "forward", wraps=agent.sa_encoder.forward) as sa_forward, \
             patch.object(agent.goal_encoder, "forward", wraps=agent.goal_encoder.forward) as goal_forward:
            self.assertTrue(agent._update_contrastive(batch))
        self.assertEqual(sa_forward.call_count, 1)
        self.assertEqual(goal_forward.call_count, 1)
        self.assertTrue(torch.isfinite(torch.tensor(agent.contrastive_loss)))

    def test_hindsight_samples_future_within_episode(self):
        agent = GRAG({"seed": 1, "hindsight_gamma": 0.5})
        agent.initialize_networks(1, 1)
        transitions = [
            {"episode_id": 0, "next_state": torch.tensor([1.])},
            {"episode_id": 0, "next_state": torch.tensor([2.])},
            {"episode_id": 1, "next_state": torch.tensor([10.])},
            {"episode_id": 1, "next_state": torch.tensor([11.])},
        ]
        weights_seen = []

        def choose_last(weights, num_samples, replacement, generator):
            weights_seen.append(weights.clone())
            return torch.arange(len(weights) - 1, len(weights) - num_samples - 1, -1)

        with patch("torch.multinomial", side_effect=choose_last):
            goals, owners = agent._sample_hindsight_goals(transitions)
        self.assertEqual(goals.flatten().tolist(), [2., 1., 2., 11., 10., 11.])
        self.assertEqual(owners.tolist(), [0, 0, 1, 2, 2, 3])
        torch.testing.assert_close(weights_seen[0], torch.tensor([1., 0.5], dtype=torch.float64))
        torch.testing.assert_close(weights_seen[1], torch.tensor([1.], dtype=torch.float64))

    def test_updates_encoders_when_another_goal_supplies_negatives(self):
        agent = GRAG({"seed": 42})
        agent.UPDATE_INTERVAL = 2
        agent.INITIAL_COLLECTION_STEPS = 2
        agent.initialize_networks(problem=GridWorldAdapter({"rows": 2, "cols": 2}))

        def observation(position):
            return {"rows": 2, "cols": 2, "obstacles": [],
                    "goal": [1, 1], "position": position}

        def step(start, end):
            obs = observation(start)
            action = agent.choose_action(obs)
            agent.learn(obs, action, 0, observation(end), False)

        before_sa = [p.detach().clone() for p in agent.sa_encoder.parameters()]
        before_goal = [p.detach().clone() for p in agent.goal_encoder.parameters()]
        before_actor = [p.detach().clone() for p in agent.actor.parameters()]
        step([0, 0], [0, 1])
        step([0, 1], [1, 1])
        agent.end_episode()
        self.assertIsNotNone(agent.contrastive_loss)
        step([1, 0], [0, 0])
        step([0, 0], [1, 0])
        agent.end_episode()

        self.assertTrue(torch.isfinite(torch.tensor(agent.contrastive_loss)))
        self.assertEqual(len(agent.episodes), 2)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_sa, agent.sa_encoder.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_goal, agent.goal_encoder.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_actor, agent.actor.parameters())))


if __name__ == "__main__":
    unittest.main()
