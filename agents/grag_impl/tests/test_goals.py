"""目標入力の契約と、実環境の終了判定との一致を検証する。"""

from copy import deepcopy
import unittest

from agents.grag import GRAG
from agents.grag_impl.goals import LanguageGoal
from gridworld.world import GridWorld


MAP = {"name": "目標判定", "rows": 2, "cols": 3, "start": [0, 0],
       "goal": [1, 2], "obstacles": [[0, 1]]}


class GoalTests(unittest.TestCase):
    def setUp(self):
        self.world = GridWorld(deepcopy(MAP))
        self.observation = self.world.reset()

    def test_explicit_text_and_map_coordinate(self):
        agent = GRAG({"goal_text": "右下のマスへ移動する"})
        goal = agent.goal_for(self.observation)
        self.assertEqual(goal.text, "右下のマスへ移動する")
        self.assertEqual(goal.target, (1, 2))

    def test_default_text_and_explicit_none(self):
        for settings in ({}, {"goal_text": None}):
            with self.subTest(settings=settings):
                goal = GRAG(settings).goal_for(self.observation)
                self.assertEqual(goal.text, "マス [1, 2] に到達する")

    def test_goal_coordinate_is_a_snapshot(self):
        goal = GRAG({}).goal_for(self.observation)
        self.observation["goal"][0] = 0
        self.assertEqual(goal.target, (1, 2))

    def test_judge_agrees_with_world_after_collision_and_arrival(self):
        agent = GRAG({})
        self.assertFalse(agent.is_goal_reached(self.observation))
        for action in ("right", "down", "right", "right"):
            with self.subTest(action=action, position=self.world.position):
                observation, reached_goal, _ = self.world.step(action)
                self.assertEqual(agent.is_goal_reached(observation), reached_goal)
        self.assertTrue(agent.is_goal_reached(observation))

    def test_text_does_not_override_coordinate_judgement(self):
        agent = GRAG({"goal_text": "スタートに到達する"})
        self.assertFalse(agent.is_goal_reached(self.observation))
        self.observation["position"] = [1, 2]
        self.assertTrue(agent.is_goal_reached(self.observation))

    def test_goal_queries_do_not_modify_inputs_or_agent_state(self):
        agent = GRAG({"goal_text": "  ゴールに到達する  "})
        before_observation = deepcopy(self.observation)
        before_agent = deepcopy(vars(agent))
        goal = agent.goal_for(self.observation)
        agent.is_goal_reached(self.observation)
        self.assertEqual(goal.text, "  ゴールに到達する  ")
        self.assertEqual(self.observation, before_observation)
        self.assertEqual(vars(agent), before_agent)

    def test_invalid_goal_text_is_rejected(self):
        for text in ("", " \n ", 3, False, ["ゴール"]):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    GRAG({"goal_text": text})
                with self.assertRaises(ValueError):
                    LanguageGoal(text=text, target=(1, 2))

    def test_invalid_goal_coordinate_is_rejected(self):
        for target in (None, [], [1], [1, 2, 3], [-1, 2], [1.0, 2],
                       [True, 2], "1,2"):
            with self.subTest(target=target):
                observation = {**self.observation, "goal": target}
                with self.assertRaises(ValueError):
                    GRAG({}).goal_for(observation)

    def test_out_of_bounds_goal_is_rejected(self):
        for target in ([2, 2], [1, 3]):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    GRAG({}).goal_for({**self.observation, "goal": target})

    def test_invalid_map_dimensions_are_rejected(self):
        for key in ("rows", "cols"):
            for value in (None, 0, -1, True, 2.0):
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        GRAG({}).goal_for({**self.observation, key: value})

    def test_invalid_position_is_rejected(self):
        goal = GRAG({}).goal_for(self.observation)
        for position in (None, [1], [-1, 2], [1, False], [1, 2.0]):
            with self.subTest(position=position):
                with self.assertRaises(ValueError):
                    goal.is_reached({**self.observation, "position": position})


if __name__ == "__main__":
    unittest.main()
