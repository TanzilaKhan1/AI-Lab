"""
A custom Gymnasium environment for the Stochastic Windy Gridworld.

Gymnasium ships no windy-gridworld environment, so we implement one against
the standard Gymnasium API. Q-Learning in q_learning.py interacts with this
environment only through env.step() -- it never sees windy_mdp.build_mdp(),
which is what makes it a genuine model-free learner, exactly as in the
Blackjack, DoorKey and Pig labs.

State  : (row, col), row 0 = top
Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
Reward : -1 on every step; episode ends on reaching the goal.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from windy_mdp import ROWS, COLS, START, GOAL, WIND, ACTIONS, MOVE


class WindyGridworldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 4}

    def __init__(self, max_steps=10_000, render_mode=None):
        super().__init__()
        if render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(f"Unsupported render_mode={render_mode!r}")
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(ROWS), spaces.Discrete(COLS))
        )
        self.pos = START
        self._n_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "start" in options:
            self.pos = options["start"]
        else:
            self.pos = START
        self._n_steps = 0
        return self.pos, {}

    def step(self, action):
        self._n_steps += 1
        r, c = self.pos
        dr, dc = MOVE[action]
        r1 = max(0, min(ROWS - 1, r + dr))
        c1 = max(0, min(COLS - 1, c + dc))

        w = WIND[c]     # wind strength at the column occupied BEFORE the move
        if w == 0:
            push = 0
        else:
            push = w + int(self.np_random.integers(-1, 2))   # w-1, w, or w+1

        r2 = max(0, min(ROWS - 1, r1 - push))
        self.pos = (r2, c1)

        terminated = self.pos == GOAL
        truncated = self._n_steps >= self.max_steps
        reward = -1.0
        return self.pos, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "ansi":
            rows = []
            for r in range(ROWS):
                cells = []
                for c in range(COLS):
                    pos = (r, c)
                    if pos == self.pos:
                        cells.append("A")
                    elif pos == START:
                        cells.append("S")
                    elif pos == GOAL:
                        cells.append("G")
                    else:
                        cells.append(".")
                rows.append(" ".join(cells))
            return "\n".join(rows)

        if self.render_mode == "rgb_array":
            cell = 48
            frame = np.full((ROWS * cell, COLS * cell, 3), 252, dtype=np.uint8)
            wind_colors = {
                0: np.array([238, 243, 251], dtype=np.uint8),
                1: np.array([158, 197, 244], dtype=np.uint8),
                2: np.array([42, 120, 214], dtype=np.uint8),
            }
            for r in range(ROWS):
                for c in range(COLS):
                    frame[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = wind_colors[WIND[c]]
            frame[::cell, :, :] = 252
            frame[:, ::cell, :] = 252

            def mark(pos, color):
                rr, cc = pos
                y, x = rr * cell + cell // 2, cc * cell + cell // 2
                yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
                mask = (yy - y) ** 2 + (xx - x) ** 2 <= (cell * 0.28) ** 2
                frame[mask] = np.array(color, dtype=np.uint8)

            mark(START, [82, 81, 78])
            mark(GOAL, [12, 163, 12])
            mark(self.pos, [227, 73, 72])
            return frame
        return None
