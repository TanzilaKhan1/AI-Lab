"""
Model-free tabular Q-Learning on the custom Stochastic Windy Gridworld
environment (windy_env.py).

The agent only ever sees what env.step() returns -- never the analytic model
in windy_mdp.py. The task is a standard shortest-expected-path problem
(reward -1 per step, episode ends at the goal), so a plain epsilon-greedy
schedule that decays toward a small floor is enough: every state is
reachable from the fixed start in only a handful of steps, unlike Pig's
high-turn-total states.
"""

import random

import numpy as np

from windy_env import WindyGridworldEnv
from windy_mdp import ACTIONS, START, GOAL

TRAIN_MAX_STEPS = 1_000


def train_q_learning(
    num_episodes=2_000,
    alpha=0.5,
    gamma=1.0,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    seed=42,
):
    env = WindyGridworldEnv(max_steps=TRAIN_MAX_STEPS)
    n_actions = len(ACTIONS)
    Q = {}

    def get_q(state):
        if state not in Q:
            Q[state] = np.zeros(n_actions)
        return Q[state]

    rng = random.Random(seed)
    epsilon = epsilon_start
    episode_returns = np.zeros(num_episodes)
    episode_steps = np.zeros(num_episodes)

    for ep in range(num_episodes):
        state, _ = env.reset(seed=(seed if ep == 0 else None))
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            if rng.random() < epsilon:
                action = rng.randrange(n_actions)
            else:
                action = int(np.argmax(get_q(state)))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            q = get_q(state)
            next_best = 0.0 if terminated else np.max(get_q(next_state))
            q[action] += alpha * (reward + gamma * next_best - q[action])

            state = next_state
            total_reward += reward
            steps += 1

        episode_returns[ep] = total_reward
        episode_steps[ep] = steps
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    env.close()
    policy = {s: ACTIONS[int(np.argmax(q))] for s, q in Q.items()}
    return Q, policy, episode_returns, episode_steps


def record_training_rollouts(
    snapshot_episodes,
    num_episodes=50_000,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.02,
    epsilon_decay=0.9999,
    seed=42,
    max_record_steps=80,
):
    """Train Q-Learning and keep the actual epsilon-greedy training rollout at
    selected episode numbers.

    These snapshots are intentionally behaviour-policy rollouts, not polished
    final greedy rollouts. They show the learning process itself: early random
    wandering/failures, then progressively shorter successful episodes.
    """
    wanted = set(snapshot_episodes)
    env = WindyGridworldEnv(max_steps=TRAIN_MAX_STEPS)
    n_actions = len(ACTIONS)
    Q = {}

    def get_q(state):
        if state not in Q:
            Q[state] = np.zeros(n_actions)
        return Q[state]

    rng = random.Random(seed)
    epsilon = epsilon_start
    snapshots = []

    for ep in range(num_episodes):
        state, _ = env.reset(seed=(seed if ep == 0 else None))
        record = ep in wanted
        # Knowledge going INTO this episode, for the value-function panel.
        q_snapshot = {s: q.copy() for s, q in Q.items()} if record else None
        states = [state] if record else None
        actions = [] if record else None
        total_reward = 0.0
        steps = 0
        done = False
        terminated = False

        while not done:
            if rng.random() < epsilon:
                action = rng.randrange(n_actions)
            else:
                action = int(np.argmax(get_q(state)))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            q = get_q(state)
            next_best = 0.0 if terminated else np.max(get_q(next_state))
            q[action] += alpha * (reward + gamma * next_best - q[action])

            if record and len(states) <= max_record_steps:
                states.append(next_state)
                actions.append(action)

            state = next_state
            total_reward += reward
            steps += 1

        if record:
            snapshots.append({
                "episode": ep,
                "epsilon": epsilon,
                "states": states,
                "actions": actions,
                "steps": steps,
                "return": total_reward,
                "success": bool(terminated),
                "truncated_for_video": steps > max_record_steps,
                "Q": q_snapshot,
            })

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    env.close()
    return snapshots


def evaluate_policy(policy, num_episodes=2_000, seed=123, default_action=ACTIONS[-1]):
    """Roll a greedy policy out from the fixed start in the real (stochastic)
    environment and report average return and average steps to the goal."""
    env = WindyGridworldEnv(max_steps=TRAIN_MAX_STEPS)
    steps_list = []

    for ep in range(num_episodes):
        state, _ = env.reset(seed=(seed + ep))
        done = False
        steps = 0
        while not done:
            action = policy.get(state, default_action)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        if terminated:
            steps_list.append(steps)

    env.close()
    avg_steps = float(np.mean(steps_list)) if steps_list else float("nan")
    success_rate = len(steps_list) / num_episodes
    return {
        "avg_return": -avg_steps if steps_list else float("nan"),
        "success_rate": success_rate,
        "avg_steps": avg_steps,
    }


def greedy_trajectory(policy, seed=0, max_steps=200, default_action=ACTIONS[-1]):
    """One sampled trajectory of a greedy policy from the fixed start, for
    plotting the path the agent actually takes."""
    env = WindyGridworldEnv(max_steps=max_steps)
    state, _ = env.reset(seed=seed)
    states = [state]
    actions = []
    done = False
    while not done:
        action = policy.get(state, default_action)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        states.append(state)
        actions.append(action)
    env.close()
    return states, actions
