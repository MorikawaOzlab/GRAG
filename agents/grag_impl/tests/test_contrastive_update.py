import unittest
from unittest.mock import patch

import torch

from agents.grag_impl.grag import GRAG
from agents.grag_impl.gridworld_adapter import GridWorldAdapter


class ContrastiveUpdateTests(unittest.TestCase):
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

    def test_infonce_uses_negative_euclidean_distances(self):
        agent = GRAG({"seed": 1, "actions": (0, 1)})
        agent.initialize_networks(1, 1)
        batch = [{"episode_id": 0, "state": torch.tensor([1.]), "action": 0},
                 {"episode_id": 1, "state": torch.tensor([2.]), "action": 1}]
        goals = torch.tensor([[3.], [4.]])
        inputs = torch.tensor([[1., 1., 0.], [2., 0., 1.]])
        with torch.no_grad():
            sa = agent.sa_encoder(inputs)
            goal = agent.goal_encoder(goals)
            expected = -torch.cdist(sa, goal)
        original_loss = torch.nn.functional.cross_entropy
        with patch.object(agent, "_sample_hindsight_goals", return_value=goals), \
             patch("agents.grag_impl.grag.F.cross_entropy", wraps=original_loss) as loss:
            self.assertTrue(agent._update_contrastive(batch))
        torch.testing.assert_close(loss.call_args.args[0], expected)

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

        def choose_last(weights, num_samples, generator):
            weights_seen.append(weights.clone())
            return torch.tensor([len(weights) - 1])

        with patch("torch.multinomial", side_effect=choose_last):
            goals = agent._sample_hindsight_goals(transitions)
        self.assertEqual(goals.flatten().tolist(), [2., 2., 11., 11.])
        self.assertTrue(torch.equal(weights_seen[0], torch.tensor([1., 0.5])))
        self.assertTrue(torch.equal(weights_seen[1], torch.tensor([1.])))

    def test_updates_encoders_when_another_episode_supplies_negatives(self):
        agent = GRAG({"seed": 42})
        agent.UPDATE_INTERVAL = 2
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
        self.assertIsNone(agent.contrastive_loss)
        step([1, 0], [0, 0])
        step([0, 0], [1, 0])
        agent.end_episode()

        self.assertTrue(torch.isfinite(torch.tensor(agent.contrastive_loss)))
        self.assertEqual(agent._contrastive_pending, [])
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_sa, agent.sa_encoder.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_goal, agent.goal_encoder.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_actor, agent.actor.parameters())))


if __name__ == "__main__":
    unittest.main()
