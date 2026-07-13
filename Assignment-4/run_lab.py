"""
Main lab script: Value Iteration (model-based) vs Q-Learning (model-free) on
the Stochastic Windy Gridworld.

Run with:            python3 run_lab.py
Re-render figures:   python3 run_lab.py plot   (uses cached results)
"""

import os
import pickle
import sys
from collections import OrderedDict

from windy_mdp import (
    build_mdp, value_iteration, action_value_gaps,
    ACTIONS, START, GOAL, WIND, ROWS, COLS,
)
from q_learning import train_q_learning, evaluate_policy, greedy_trajectory, record_training_rollouts
import make_plots as mp

CACHE = "results.pkl"

GAMMA = 0.99
QL_EPISODES = 50_000
QL_ALPHA = 0.1
QL_EPS_MIN = 0.02
QL_EPS_DECAY = 0.9999

GAMMAS = [0.7, 0.9, 0.95, 0.99, 1.0]
ALPHAS = [0.02, 0.05, 0.1, 0.3, 0.5]
EPS_DECAYS = [0.999, 0.9999, 0.99995]
THETAS = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
TRAINING_VIDEO_SNAPSHOTS = [0, 25, 100, 500, 2_000, 10_000, 25_000, 49_999]
TRAINING_VIDEO_MAX_STEPS = 180
TRAINING_ROLLOUT_VERSION = 2   # v2: snapshots carry the Q-table for the value panel


def compute():
    # ---- Step 1: Model-based -- derive the wind model, run Value Iteration ----
    print("Building the Windy Gridworld transition model (derived from the wind)...")
    P = build_mdp()
    print(f"States in model: {len(P)}")
    print("Running Value Iteration...")
    V, vi_policy, vi_history = value_iteration(P, gamma=GAMMA)
    print(f"Converged in {len(vi_history)} sweeps (final delta={vi_history[-1]:.2e})")
    gaps = action_value_gaps(P, V, GAMMA)
    vi_eval = evaluate_policy(vi_policy, num_episodes=5_000, seed=1)
    print(f"VI performance: success {vi_eval['success_rate']:.3f} | "
          f"avg steps {vi_eval['avg_steps']:.3f}")
    vi_states, vi_actions = greedy_trajectory(vi_policy, seed=0)

    # ---- Step 2: Model-free -- Q-Learning on the real environment --------------
    print("\nTraining Q-Learning (model-free)...")
    _, ql_policy, ql_returns, ql_steps = train_q_learning(
        num_episodes=QL_EPISODES, alpha=QL_ALPHA, gamma=GAMMA,
        epsilon_start=1.0, epsilon_min=QL_EPS_MIN, epsilon_decay=QL_EPS_DECAY, seed=42,
    )
    ql_eval = evaluate_policy(ql_policy, num_episodes=5_000, seed=3)
    print(f"QL performance: success {ql_eval['success_rate']:.3f} | "
          f"avg steps {ql_eval['avg_steps']:.3f}")
    ql_states, ql_actions = greedy_trajectory(ql_policy, seed=0)

    # ---- Step 3: policy agreement over the full state space --------------------
    shared = [s for s in vi_policy if s in ql_policy]
    agree = sum(1 for s in shared if vi_policy[s] == ql_policy[s])
    print(f"\nPolicy agreement over shared states: {agree}/{len(shared)} "
          f"({100 * agree / len(shared):.1f}%)")

    # ---- Step 4: hyperparameter experiments -------------------------------------
    print("\nDiscount-factor (gamma) sweep on Value Iteration:")
    gamma_data = []
    for g in GAMMAS:
        Vg, pg, hg = value_iteration(P, gamma=g)
        eval_g = evaluate_policy(pg, num_episodes=3_000, seed=7)
        gamma_data.append({
            "gamma": g, "sweeps": len(hg),
            "v_start": Vg[START],
            "avg_steps": eval_g["avg_steps"],
            "success_rate": eval_g["success_rate"],
        })
        print(f"  gamma={g:<5}: {len(hg)} sweeps | V(start)={Vg[START]:.3f} | "
              f"avg steps={eval_g['avg_steps']:.3f} | success={eval_g['success_rate']:.3f}")

    print("\nConvergence-threshold (theta) sweep on Value Iteration:")
    theta_data = []
    for th in THETAS:
        Vt, pt, ht = value_iteration(P, gamma=GAMMA, theta=th)
        theta_data.append({"theta": th, "sweeps": len(ht), "v_start": Vt[START], "policy": pt})
    gold_policy = theta_data[-1]["policy"]   # tightest theta = reference policy
    for row in theta_data:
        same = sum(1 for s in gold_policy if row["policy"][s] == gold_policy[s])
        row["policy_agreement_pct"] = 100 * same / len(gold_policy)
        print(f"  theta={row['theta']:.0e}: {row['sweeps']} sweeps | V(start)={row['v_start']:.4f} | "
              f"policy agreement vs tightest theta={row['policy_agreement_pct']:.1f}%")

    print("\nQ-Learning learning-rate sweep...")
    alpha_series = OrderedDict()
    for a in ALPHAS:
        _, _, _, steps_curve = train_q_learning(
            num_episodes=QL_EPISODES, alpha=a, gamma=GAMMA,
            epsilon_start=1.0, epsilon_min=QL_EPS_MIN, epsilon_decay=QL_EPS_DECAY, seed=42,
        )
        alpha_series[f"alpha={a}"] = steps_curve

    print("Q-Learning exploration-decay sweep...")
    eps_series = OrderedDict()
    for d in EPS_DECAYS:
        _, _, _, steps_curve = train_q_learning(
            num_episodes=QL_EPISODES, alpha=QL_ALPHA, gamma=GAMMA,
            epsilon_start=1.0, epsilon_min=QL_EPS_MIN, epsilon_decay=d, seed=42,
        )
        eps_series[f"epsilon_decay={d}"] = steps_curve

    return dict(
        V=V, vi_policy=vi_policy, vi_history=vi_history, gaps=gaps,
        vi_eval=vi_eval, vi_states=vi_states, vi_actions=vi_actions,
        ql_policy=ql_policy, ql_eval=ql_eval, ql_returns=ql_returns, ql_steps=ql_steps,
        ql_states=ql_states, ql_actions=ql_actions,
        agree=agree, n_shared=len(shared),
        gamma_data=gamma_data, alpha_series=alpha_series, eps_series=eps_series,
        theta_data=theta_data,
    )


def make_figures(R):
    print("\nGenerating figures...")
    mp.plot_environment("fig_environment.png")
    mp.plot_convergence(R["vi_history"], "fig_vi_convergence.png")
    mp.plot_value_heatmap(R["V"], "fig_value_heatmap.png")
    mp.plot_gaps(R["gaps"], "fig_gaps.png")
    mp.plot_trajectories(R["vi_states"], R["ql_states"], "fig_trajectories.png")
    mp.plot_policy_agreement(R["vi_policy"], R["ql_policy"], "fig_policy_agreement.png")
    mp.plot_gamma_sweep(R["gamma_data"], "fig_gamma_sweep.png")
    mp.plot_theta_sweep(R["theta_data"], "fig_theta_sweep.png")
    mp.plot_sweep(R["alpha_series"], R["vi_eval"]["avg_steps"], window=100,
                  xlabel="Episode", ylabel="Steps to goal",
                  title="Q-Learning: effect of learning rate alpha",
                  filename="fig_alpha_sweep.png")
    mp.plot_sweep(R["eps_series"], R["vi_eval"]["avg_steps"], window=100,
                  xlabel="Episode", ylabel="Steps to goal",
                  title="Q-Learning: effect of exploration decay speed",
                  filename="fig_epsilon_sweep.png")
    mp.plot_evaluation_comparison(R["vi_eval"], R["ql_eval"], "fig_evaluation_comparison.png")
    mp.animate_learning_progress(R["ql_steps"], R["vi_eval"]["avg_steps"], "anim_learning_progress.gif")
    mp.animate_greedy_trajectory(R["ql_states"], "anim_greedy_trajectory.gif")
    mp.animate_training_rollouts(R["training_rollouts"], R["ql_steps"],
                                 R["vi_eval"]["avg_steps"], "anim_training_rollouts.gif")
    print("Saved: fig_environment, fig_vi_convergence, fig_value_heatmap, fig_gaps,")
    print("       fig_trajectories, fig_policy_agreement, fig_gamma_sweep, fig_theta_sweep,")
    print("       fig_alpha_sweep, fig_epsilon_sweep, fig_evaluation_comparison,")
    print("       anim_learning_progress, anim_greedy_trajectory, anim_training_rollouts")


def main():
    if "plot" in sys.argv and os.path.exists(CACHE):
        print(f"Loading cached results from {CACHE} (skipping training)...")
        with open(CACHE, "rb") as f:
            R = pickle.load(f)
    else:
        R = compute()
        with open(CACHE, "wb") as f:
            pickle.dump(R, f)
    if (
        "training_rollouts" not in R
        or R.get("training_rollout_max_record_steps") != TRAINING_VIDEO_MAX_STEPS
        or R.get("training_rollout_version") != TRAINING_ROLLOUT_VERSION
    ):
        print("Recording actual Q-Learning training rollouts for video...")
        R["training_rollouts"] = record_training_rollouts(
            TRAINING_VIDEO_SNAPSHOTS,
            num_episodes=QL_EPISODES,
            alpha=QL_ALPHA,
            gamma=GAMMA,
            epsilon_start=1.0,
            epsilon_min=QL_EPS_MIN,
            epsilon_decay=QL_EPS_DECAY,
            seed=42,
            max_record_steps=TRAINING_VIDEO_MAX_STEPS,
        )
        R["training_rollout_max_record_steps"] = TRAINING_VIDEO_MAX_STEPS
        R["training_rollout_version"] = TRAINING_ROLLOUT_VERSION
        with open(CACHE, "wb") as f:
            pickle.dump(R, f)
    make_figures(R)


if __name__ == "__main__":
    main()
