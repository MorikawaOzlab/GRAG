"""Active training time and the first complete 100/100 success window."""

from itertools import count
from threading import Event, Thread
import json
import unittest
from unittest.mock import patch

from gridworld.training_runtime import TrainingSession
from .test_session import CountingAgent, SETTINGS


MAP = {"name": "timing", "rows": 2, "cols": 2, "start": [0, 0],
       "goal": [0, 1], "obstacles": []}


class SuccessfulAgent(CountingAgent):
    def choose_action(self, observation, greedy=False):
        return "right"

    def checkpoint(self):
        return dict(vars(self))

    def load_checkpoint(self, state):
        vars(self).update(state)


class InterruptedSuccessAgent(SuccessfulAgent):
    def choose_action(self, observation, greedy=False):
        return "down" if self.episodes in (50, 151) else "right"


class TimingTests(unittest.TestCase):
    def setUp(self):
        # Each step costs 0.25 seconds, with ~1000 seconds between steps.
        # Time spent paused, polling or displaying frames must not be counted.
        ticks = count()

        def clock():
            n = next(ticks)
            return (n // 2) * 1000 + (n % 2) * 0.25

        self.timer = patch("gridworld.training_runtime.session.perf_counter",
                           side_effect=clock)
        self.timer.start()
        self.addCleanup(self.timer.stop)

    def session(self, agent=SuccessfulAgent):
        return TrainingSession(MAP, {**SETTINGS, "max_steps": 1}, agent)

    def test_requires_100_completed_successes_and_excludes_waiting_and_evaluation(self):
        session = self.session()
        session.run_batch(99)
        self.assertIsNone(session.success_100_milestone)
        self.assertEqual(session.result(compact=True)["training_seconds"], 24.75)
        session.evaluate()
        session.inspect_policy([0, 0])
        self.assertEqual(session.training_seconds, 24.75)
        session.run_batch(1)
        expected = {"seconds": 25.0, "episode": 100, "steps": 100}
        self.assertEqual(session.result()["success_100_milestone"], expected)
        session.run_batch(5)
        self.assertEqual(session.training_seconds, 26.25)
        self.assertEqual(session.success_100_milestone, expected)

    def test_failure_resets_window_and_first_milestone_survives_later_failure(self):
        session = self.session(InterruptedSuccessAgent)
        session.MAX_CURVE_POINTS = 20
        session.run_batch(150)
        self.assertIsNone(session.success_100_milestone)
        session.run_batch(1)
        expected = {"seconds": 37.75, "episode": 151, "steps": 151}
        self.assertEqual(session.success_100_milestone, expected)
        session.run_batch(1)
        self.assertLess(session.result()["recent_success_rate"], 1)
        self.assertEqual(session.success_100_milestone, expected)

    def test_json_save_and_resume_preserves_elapsed_time_and_window(self):
        session = self.session()
        session.run_batch(60)
        saved = json.loads(json.dumps(session.result(checkpoint=True)))
        restored = TrainingSession.restore(saved, SuccessfulAgent)
        self.assertEqual(restored.training_seconds, 15)
        restored.run_batch(40)
        expected = {"seconds": 25.0, "episode": 100, "steps": 100}
        self.assertEqual(restored.success_100_milestone, expected)
        again = TrainingSession.restore(json.loads(json.dumps(restored.result(checkpoint=True))), SuccessfulAgent)
        again.run_batch(1)
        self.assertEqual(again.success_100_milestone, expected)

    def test_legacy_checkpoint_does_not_invent_missing_past_time(self):
        session = self.session()
        session.run_batch(10)
        saved = session.result(checkpoint=True)
        for name in ("training_seconds", "success_100_milestone", "timing_complete"):
            del saved[name]
        restored = TrainingSession.restore(saved, SuccessfulAgent)
        self.assertFalse(restored.timing_complete)
        restored.run_batch(100)
        self.assertEqual(restored.training_seconds, 25)
        self.assertIsNone(restored.success_100_milestone)

    def test_live_clock_is_readable_while_step_holds_session_lock(self):
        entered, release = Event(), Event()

        class BlockingAgent(SuccessfulAgent):
            def learn(self, *args):
                entered.set()
                release.wait(3)
                return super().learn(*args)

        session = self.session(BlockingAgent)
        with patch('gridworld.training_runtime.session.perf_counter') as clock:
            clock.return_value = 10.0
            worker = Thread(target=session.step)
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                clock.return_value = 15.0
                self.assertEqual(session.training_clock(), {'seconds': 5.0, 'running': True})
                self.assertEqual(session.training_seconds, 0)
                clock.return_value = 19.0
            finally:
                release.set()
                worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertEqual(session.training_clock(), {'seconds': 9.0, 'running': False})


if __name__ == "__main__":
    unittest.main()
