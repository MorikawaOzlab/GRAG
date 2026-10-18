"""Create a reproducible 20 x 20 maze for the map editor."""

from collections import deque
from random import Random

from gridworld.storage import save_map


SIZE = 20
START = (1, 1)
GOAL = (18, 18)
NAME = "20×20 迷路"


def build_maze() -> dict:
    rng = Random(20261012)
    corridors = {START}
    seen = {START}
    stack = [START]
    while stack:
        row, col = stack[-1]
        options = [(row + dr, col + dc, row + dr // 2, col + dc // 2)
                   for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2))
                   if 1 <= row + dr <= 17 and 1 <= col + dc <= 17
                   and (row + dr, col + dc) not in seen]
        if not options:
            stack.pop()
            continue
        next_row, next_col, between_row, between_col = rng.choice(options)
        next_cell = (next_row, next_col)
        corridors.add((between_row, between_col))
        corridors.add(next_cell)
        seen.add(next_cell)
        stack.append(next_cell)

    # Open one entrance to the lower-right goal while keeping the outer border.
    corridors.update(((17, 18), GOAL))
    obstacles = [[row, col] for row in range(SIZE) for col in range(SIZE)
                 if (row, col) not in corridors]
    return {"name": NAME, "rows": SIZE, "cols": SIZE,
            "start": list(START), "goal": list(GOAL), "obstacles": obstacles}


def shortest_path_length(map_data: dict) -> int:
    walls = {tuple(cell) for cell in map_data["obstacles"]}
    queue = deque([(START, 0)])
    seen = {START}
    while queue:
        (row, col), distance = queue.popleft()
        if (row, col) == GOAL:
            return distance
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            candidate = (row + dr, col + dc)
            if (0 <= candidate[0] < SIZE and 0 <= candidate[1] < SIZE
                    and candidate not in walls and candidate not in seen):
                seen.add(candidate)
                queue.append((candidate, distance + 1))
    raise RuntimeError("ゴールへ到達できません。")


if __name__ == "__main__":
    maze = build_maze()
    distance = shortest_path_length(maze)
    saved = save_map(maze)
    print(f"保存: {saved['name']} / {SIZE}×{SIZE} / 最短{distance}歩 / 障害物{len(saved['obstacles'])}マス")
