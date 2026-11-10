from copy import deepcopy
import json
import unittest

from agents.q_learning import QLearningAgent
from gridworld.storage import DEFAULT_SETTINGS
from gridworld.training_runtime import TrainingSession


MAP = {"name": "共通学習", "rows": 2, "cols": 3, "start": [0, 0],
       "goal": [1, 2], "obstacles": [[0, 1]]}
SETTINGS = {**DEFAULT_SETTINGS, "episodes": 100, "max_steps": 20}


class CountingAgent:
    def __init__(self, settings):
        self.updates = 0
        self.episodes = 0
        self.choices = 0

    def choose_action(self, observation, greedy=False):
        self.choices += 1
        return "down" if observation["position"][0] == 0 else "right"

    def learn(self, *args):
        self.updates += 1

    def end_episode(self):
        self.episodes += 1

    def export(self):
        return {"updates": self.updates, "episodes": self.episodes}


class ChangedPolicyAgent(CountingAgent):
    def choose_action(self, observation, greedy=False):
        return "up" if self.episodes else super().choose_action(observation, greedy)


class WindowAgent(CountingAgent):
    def choose_action(self, observation, greedy=False):
        return "right" if self.episodes < 100 else "down"


class SessionTests(unittest.TestCase):
    def test_split_training_matches_uninterrupted_training(self):
        continuous = TrainingSession(MAP, SETTINGS)
        continuous.run_batch(200)
        split = TrainingSession(MAP, SETTINGS)
        split.run_batch(100)
        split.evaluate()
        split.run_batch(100)
        self.assertEqual(split.history, continuous.history)
        self.assertEqual(split.visits, continuous.visits)
        self.assertEqual(split.agent.checkpoint(), continuous.agent.checkpoint())

    def test_json_checkpoint_survives_application_restart(self):
        original = TrainingSession(MAP, SETTINGS)
        original.run_batch(100)
        original.evaluate()
        saved = json.loads(json.dumps(original.result(checkpoint=True)))
        restored = TrainingSession.restore(saved)
        original.run_batch(20)
        restored.run_batch(20)
        self.assertEqual(restored.history, original.history)
        self.assertEqual(restored.agent.checkpoint(), original.agent.checkpoint())
        self.assertEqual(restored.visits, original.visits)

    def test_arbitrary_agent_keeps_same_instance_and_learning_state(self):
        session = TrainingSession(MAP, SETTINGS, CountingAgent)
        agent = session.agent
        session.run_batch(3)
        session.run_batch(2)
        self.assertIs(session.agent, agent)
        self.assertEqual(agent.episodes, 5)
        self.assertEqual(agent.updates, 15)
        self.assertEqual([item["episode"] for item in session.history], [1, 2, 3, 4, 5])

    def test_policy_evaluation_changes_no_learning_or_random_state(self):
        session = TrainingSession(MAP, SETTINGS)
        session.run_batch(5)
        state = deepcopy(session.agent.checkpoint())
        history, visits = deepcopy(session.history), deepcopy(session.visits)
        session.evaluate()
        self.assertEqual(session.agent.checkpoint(), state)
        self.assertEqual(session.history, history)
        self.assertEqual(session.visits, visits)
        generic = TrainingSession(MAP, SETTINGS, CountingAgent)
        generic.run_batch(2)
        before = deepcopy(vars(generic.agent))
        generic.evaluate()
        self.assertEqual(vars(generic.agent), before)

    def test_evaluation_uses_current_policy_instead_of_successful_training_path(self):
        session = TrainingSession(MAP, SETTINGS, ChangedPolicyAgent)
        session.run_batch(1)
        self.assertTrue(session.history[0]["success"])
        frames = session.evaluate()
        self.assertTrue(all(item["action"] == "up" for item in frames[1:]))
        self.assertFalse(frames[-1]["reached_goal"])

    def test_moving_average_window_continues_across_batches(self):
        map_data = {"name": "窓", "rows": 2, "cols": 2, "start": [0, 0],
                    "goal": [0, 1], "obstacles": []}
        session = TrainingSession(map_data, {**SETTINGS, "max_steps": 1}, WindowAgent)
        session.run_batch(100)
        session.run_batch(2)
        self.assertEqual(session.history[99]["moving_average_reward"], 1)
        self.assertEqual(session.history[100]["moving_average_reward"], .99)
        self.assertEqual(session.history[101]["moving_average_reward"], .98)
        self.assertEqual(session.total_reward, 100)

    def test_greedy_q_action_does_not_create_unvisited_states(self):
        agent = QLearningAgent(SETTINGS)
        before = agent.checkpoint()
        agent.choose_action({"position": [0, 0]}, greedy=True)
        self.assertEqual(agent.checkpoint(), before)

    def test_compact_live_result_has_latest_episode_without_model_state(self):
        session = TrainingSession(MAP, {**SETTINGS, "max_steps": 1})
        session.run_batch(1001)
        compact = session.result(compact=True)
        self.assertLessEqual(len(compact["history"]), 701)
        self.assertEqual(compact["history"][-1]["episode"], 1001)
        self.assertIsNone(compact["q_table"])
        self.assertNotIn("agent_checkpoint", compact)

    def test_episode_count_validation(self):
        session = TrainingSession(MAP, SETTINGS)
        for value in (0, -1, 1.5, True, 1000001):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    session.run_batch(value)
        self.assertEqual(session.history, [])


if __name__ == "__main__":
    unittest.main()
