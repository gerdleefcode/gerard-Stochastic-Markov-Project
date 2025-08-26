import os
from typing import List, Tuple

import numpy as np

from stochmarkov.hmm import DiscreteHMM
from stochmarkov.utils import ensure_dir, set_seed
from stochmarkov.viz import (
    plot_transition_heatmap,
    plot_emission_heatmap,
    plot_posterior_heatmap,
    plot_learning_curve,
    plot_decoding_comparison,
)

def generate_true_hmm(seed: int = 7) -> DiscreteHMM:
    set_seed(seed)
    # 3-state HMM with 6 observation symbols
    A = np.array(
        [
            [0.85, 0.10, 0.05],
            [0.15, 0.75, 0.10],
            [0.10, 0.20, 0.70],
        ]
    )
    B = np.array(
        [
            [0.60, 0.25, 0.10, 0.03, 0.01, 0.01],  # State 0 likes obs 0/1
            [0.05, 0.10, 0.65, 0.15, 0.04, 0.01],  # State 1 likes obs 2/3
            [0.02, 0.03, 0.05, 0.10, 0.35, 0.45],  # State 2 likes obs 4/5
        ]
    )
    pi = np.array([0.70, 0.20, 0.10])
    state_names = ["A", "B", "C"]
    obs_names = [f"sym{k}" for k in range(6)]
    return DiscreteHMM(n_states=3, n_obs=6, A=A, B=B, pi=pi, state_names=state_names, obs_names=obs_names)


def main():
    out_dir = os.path.join("outputs", "hmm")
    ensure_dir(out_dir)
    set_seed(2024)

    hmm_true = generate_true_hmm(seed=11)
    plot_transition_heatmap(hmm_true.A, hmm_true.state_names, title="True HMM Transition Matrix", savepath=os.path.join(out_dir, "true_A_heatmap.png"))
    plot_emission_heatmap(hmm_true.B, hmm_true.state_names, hmm_true.obs_names, title="True Emission Matrix", savepath=os.path.join(out_dir, "true_B_heatmap.png"))

    # Generate training sequences
    train_sequences: List[List[int]] = []
    true_state_sequences: List[List[int]] = []
    for _ in range(20):
        states, obs = hmm_true.sample(n_steps=300)
        train_sequences.append(obs)
        true_state_sequences.append(states)

    # Initialize a random HMM
    hmm = DiscreteHMM.random(n_states=hmm_true.n_states, n_obs=hmm_true.n_obs, alpha_trans=1.0, alpha_emit=1.0, alpha_pi=1.0)
    hmm.state_names = hmm_true.state_names[:]  # copy names for consistent labeling
    hmm.obs_names = hmm_true.obs_names[:]

    # Train with EM
    loglikes = hmm.em_baum_welch(train_sequences, n_iter=100, tol=1e-3, smoothing=1e-2, verbose=True, seed=123)
    plot_learning_curve(loglikes, savepath=os.path.join(out_dir, "em_learning_curve.png"))

    # Evaluate on a fresh long sequence
    states_true, obs = hmm_true.sample(n_steps=400, seed=999)
    gamma, ll = hmm.posterior_marginals(obs)
    states_vit, logp_vit = hmm.viterbi(obs)

    print(f"Test sequence log-likelihood (trained HMM): {ll:.2f}")
    print(f"Viterbi path log-prob: {logp_vit:.2f}")

    # Visualizations
    plot_transition_heatmap(hmm.A, hmm.state_names, title="Learned HMM Transition Matrix", savepath=os.path.join(out_dir, "learned_A_heatmap.png"))
    plot_emission_heatmap(hmm.B, hmm.state_names, hmm.obs_names, title="Learned Emission Matrix", savepath=os.path.join(out_dir, "learned_B_heatmap.png"))
    plot_posterior_heatmap(gamma, hmm.state_names, savepath=os.path.join(out_dir, "posterior_heatmap.png"))
    plot_decoding_comparison(states_true, states_vit, hmm.state_names, savepath=os.path.join(out_dir, "viterbi_vs_true.png"))


if __name__ == "__main__":
    main()