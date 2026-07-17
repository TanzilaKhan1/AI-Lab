"""
Hand-derived transition model (MDP) for the Stochastic Windy Gridworld
(Sutton & Barto, Example 6.5 / Exercise 6.10), plus tabular Value Iteration
on top of it.

Grid
----
7 rows x 10 columns, using matrix indexing: row 0 is the TOP row, row
increases downward. Start S = (row=3, col=0), goal G = (row=3, col=7)
(the classic layout). The agent takes one of 4 moves (UP/DOWN/LEFT/RIGHT).
Per the textbook convention, the wind strength is read from the column the
agent occupies BEFORE the move (not the column it lands in); the resulting
row shift is then applied to the landing column. This matches Sutton &
Barto Example 6.5 exactly: under deterministic mean winds the shortest path
from S to G is 15 steps, the standard reference value.

    column        0  1  2  3  4  5  6  7  8  9
    mean wind     0  0  0  1  1  1  2  2  1  0

In a windy column (mean wind w > 0) the ACTUAL push is stochastic:
w-1, w, or w+1, each with probability 1/3 (Exercise 6.10's "stochastic
wind"). In a calm column (w = 0) the push is deterministically 0. All
positions are clipped to stay on the grid.

State = (row, col): a genuine Markov state, since the wind at each column
is drawn i.i.d. every time the agent passes through it.

Reward is -1 on every transition (including the one that reaches the goal);
the episode ends the moment the agent reaches G. Minimizing the number of
steps to the goal is therefore exactly the same as maximizing return, for
ANY discount 0 < gamma <= 1 in a deterministic environment -- but here the
environment is genuinely stochastic (the wind branches into up to three
outcomes), so gamma is not just cosmetic: because -(1-gamma^T)/(1-gamma) is
a convex, decreasing function of the (random) path length T (equivalently the
cost (1-gamma^T)/(1-gamma) is concave and increasing in T), Jensen's
inequality means gamma weights the whole distribution of path length, not
just its mean. Two routes with the same expected length but different
variance are no longer tied once gamma < 1 -- see the gamma sweep in the report.

Unlike DoorKey (deterministic) or Pig (flat optimum away from the target),
every state here has a clearly-best action with a real margin over the
runner-up, so Q-Learning is expected to recover Value Iteration's exact
greedy policy almost everywhere -- the "clean" model-based/model-free story.
"""

import numpy as np

ROWS, COLS = 7, 10
START = (3, 0)
GOAL = (3, 7)

WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]   # mean wind strength per column

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_NAME = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}
ACTION_ARROW = {UP: "^", DOWN: "v", LEFT: "<", RIGHT: ">"}
MOVE = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


def states():
    return [(r, c) for r in range(ROWS) for c in range(COLS) if (r, c) != GOAL]


def _clip_row(r):
    return max(0, min(ROWS - 1, r))


def _clip_col(c):
    return max(0, min(COLS - 1, c))


def wind_outcomes(col):
    """List of (probability, push_amount) for the wind at this column."""
    w = WIND[col]
    if w == 0:
        return [(1.0, 0)]
    return [(1 / 3, w - 1), (1 / 3, w), (1 / 3, w + 1)]


def step_outcomes(state, action):
    """All (probability, next_state, reward, done) outcomes of taking
    `action` in `state`, under the stochastic wind."""
    r, c = state
    dr, dc = MOVE[action]
    r1, c1 = _clip_row(r + dr), _clip_col(c + dc)   # deterministic move
    outcomes = []
    for p, push in wind_outcomes(c):                # wind at the state's own column
        r2 = _clip_row(r1 - push)                    # wind blows toward row 0
        ns = (r2, c1)
        done = ns == GOAL
        outcomes.append((p, ns, -1.0, done))
    return outcomes


def build_mdp():
    """P[state][action] -> list of (prob, next_state, reward, done)."""
    P = {}
    for s in states():
        P[s] = {a: step_outcomes(s, a) for a in ACTIONS}
    return P


def value_iteration(P, gamma=1.0, theta=1e-10):
    """Standard synchronous Value Iteration. Returns V, greedy policy, and
    the per-sweep max-delta history."""
    S = list(P.keys())
    V = {s: 0.0 for s in S}
    V[GOAL] = 0.0
    history = []

    def action_value(state, action, V):
        total = 0.0
        for prob, nxt, reward, done in P[state][action]:
            total += prob * (reward if done else reward + gamma * V[nxt])
        return total

    while True:
        delta = 0.0
        for s in S:
            best = max(action_value(s, a, V) for a in ACTIONS)
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        history.append(delta)
        if delta < theta:
            break

    policy = {}
    for s in S:
        values = [action_value(s, a, V) for a in ACTIONS]
        policy[s] = ACTIONS[int(np.argmax(values))]
    return V, policy, history


def action_value_gaps(P, V, gamma):
    """Margin between the best and second-best action at every state, used
    to show the optimum is NOT flat here."""
    def av(state, action):
        total = 0.0
        for prob, nxt, reward, done in P[state][action]:
            total += prob * (reward if done else reward + gamma * V[nxt])
        return total
    gaps = {}
    for s in P:
        values = sorted((av(s, a) for a in ACTIONS), reverse=True)
        gaps[s] = values[0] - values[1]
    return gaps
