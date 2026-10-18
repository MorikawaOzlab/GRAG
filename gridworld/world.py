"""Grid environment. It contains no reward or learning logic."""

from gridworld.storage import validate_map


ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


class GridWorld:
    def __init__(self, map_data: dict):
        self.map = validate_map(map_data)
        self.obstacles = {tuple(p) for p in self.map["obstacles"]}
        self.position = tuple(self.map["start"])
        self.done = False

    def observation(self) -> dict:
        return {**self.map, "position": list(self.position)}

    def reset(self) -> dict:
        self.position = tuple(self.map["start"])
        self.done = False
        return self.observation()

    def step(self, action: str) -> tuple[dict, bool, bool]:
        if self.done:
            raise RuntimeError("試行が終了しています。reset()を呼んでください。")
        self.position, self.done, collision = self.transition(self.position, action)
        return self.observation(), self.done, collision

    def transition(self, position: tuple[int, int], action: str) -> tuple[tuple[int, int], bool, bool]:
        """Pure transition, also used to precompute training transitions."""
        if action not in ACTIONS:
            raise ValueError(f"不明な行動: {action}")
        dr, dc = ACTIONS[action]
        target = (position[0] + dr, position[1] + dc)
        collision = (not 0 <= target[0] < self.map["rows"]
                     or not 0 <= target[1] < self.map["cols"]
                     or target in self.obstacles)
        next_position = position if collision else target
        return next_position, next_position == tuple(self.map["goal"]), collision
