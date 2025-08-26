import os
from typing import List

import numpy as np

from stochmarkov.markov_chain import MarkovChain
from stochmarkov.utils import ensure_dir, set_seed
from stochmarkov.viz import (
    plot_transition_heatmap,
    plot_state_graph,
    plot_stationary_bar,
    plot_sequence_states,
)


def generate_synthetic_sequences(P: np.ndarray, n_sequences: int, length: int, seed: int = 42) -> List[List[int]]:
    set_seed(seed)
    mc = MarkovChain(P)
    sequences = []
    pi = mc.stationary()
    for _ in range(n_sequences):
        start = int(np.random.choice(mc.n_states, p=pi))
        seq = mc.sample(n_steps=length, start_state=start)
        sequences.append(seq)
    return sequences


def main():
    set_seed(123)
    out_dir = os.path.join("outputs", "markov")
    ensure_dir(out_dir)

    # True Markov chain
    state_names = ["Sunny", "Cloudy", "Rainy", "Stormy"]
    P_true = np.array(
        [
            [0.70, 0.20, 0.08, 0.02],
            [0.25, 0.55, 0.18, 0.02],
            [0.15, 0.25, 0.50, 0.10],
            [0.10, 0.10, 0.30, 0.50],
        ]
    )
    mc_true = MarkovChain(P_true, state_names=state_names)

    # Visualize ground-truth chain
    plot_transition_heatmap(mc_true.P, state_names, title="True Transition Matrix", savepath=os.path.join(out_dir, "true_P_heatmap.png"))
    plot_state_graph(mc_true.P, state_names, title="True State Graph", threshold=0.05, savepath=os.path.join(out_dir, "true_state_graph.png"))
    pi_true = mc_true.stationary()
    plot_stationary_bar(pi_true, state_names, title="True Stationary Distribution", savepath=os.path.join(out_dir, "true_stationary.png"))

    # Generate synthetic data
    sequences = generate_synthetic_sequences(mc_true.P, n_sequences=50, length=200, seed=123)

    # Fit Markov chain from data
    mc_fit = MarkovChain.fit_from_sequences(sequences, n_states=len(state_names), smoothing=1.0, state_names=state_names)

    # Compare matrices and visualize learned chain
    plot_transition_heatmap(mc_fit.P, state_names, title="Learned Transition Matrix", savepath=os.path.join(out_dir, "learned_P_heatmap.png"))
    plot_state_graph(mc_fit.P, state_names, title="Learned State Graph", threshold=0.05, savepath=os.path.join(out_dir, "learned_state_graph.png"))
    pi_fit = mc_fit.stationary()
    plot_stationary_bar(pi_fit, state_names, title="Learned Stationary Distribution", savepath=os.path.join(out_dir, "learned_stationary.png"))

    # Mixing time estimate
    t_mix_true, gap_true = mc_true.mixing_time(eps=1e-3)
    t_mix_fit, gap_fit = mc_fit.mixing_time(eps=1e-3)
    print(f"True chain: spectral gap={gap_true:.4f}, t_mix upper bound (eps=1e-3)={t_mix_true}")
    print(f"Fitted chain: spectral gap={gap_fit:.4f}, t_mix upper bound (eps=1e-3)={t_mix_fit}")

    # Sample a trajectory and plot
    traj = mc_fit.sample(n_steps=150, seed=999)
    plot_sequence_states(traj, state_names, title="Sampled State Trajectory (Learned MC)", savepath=os.path.join(out_dir, "sample_trajectory.png"))

    # Log-likelihood on a held-out sequence
    heldout = sequences[0]
    ll_true = mc_true.log_likelihood(heldout)
    ll_fit = mc_fit.log_likelihood(heldout)
    print(f"Log-likelihood (held-out) - True MC: {ll_true:.2f} | Learned MC: {ll_fit:.2f}")


if __name__ == "__main__":
    main()