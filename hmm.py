from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .utils import normalize_rows, normalize_vector, random_stochastic_matrix, logsumexp


@dataclass
class DiscreteHMM:
    n_states: int
    n_obs: int
    A: np.ndarray  # transitions (n_states x n_states)
    B: np.ndarray  # emissions (n_states x n_obs)
    pi: np.ndarray  # initial distribution (n_states,)
    state_names: Optional[List[str]] = None
    obs_names: Optional[List[str]] = None

    def __post_init__(self):
        self.A = normalize_rows(np.asarray(self.A, dtype=float))
        self.B = normalize_rows(np.asarray(self.B, dtype=float))
        self.pi = normalize_vector(np.asarray(self.pi, dtype=float))
        if self.state_names is None:
            self.state_names = [f"H{i}" for i in range(self.n_states)]
        if self.obs_names is None:
            self.obs_names = [f"o{i}" for i in range(self.n_obs)]

    @staticmethod
    def random(n_states: int, n_obs: int, alpha_trans: float = 1.0, alpha_emit: float = 1.0, alpha_pi: float = 1.0):
        A = random_stochastic_matrix(n_states, alpha=alpha_trans)
        B = np.vstack([np.random.dirichlet(alpha_emit * np.ones(n_obs)) for _ in range(n_states)])
        pi = np.random.dirichlet(alpha_pi * np.ones(n_states))
        return DiscreteHMM(n_states=n_states, n_obs=n_obs, A=A, B=B, pi=pi)

    def sample(self, n_steps: int, seed: Optional[int] = None) -> Tuple[List[int], List[int]]:
        np.random.seed(seed)
        states = []
        obs = []
        s = int(np.random.choice(self.n_states, p=self.pi))
        states.append(s)
        o = int(np.random.choice(self.n_obs, p=self.B[s]))
        obs.append(o)
        for _ in range(1, n_steps):
            s = int(np.random.choice(self.n_states, p=self.A[states[-1]]))
            states.append(s)
            o = int(np.random.choice(self.n_obs, p=self.B[s]))
            obs.append(o)
        return states, obs

    def forward_backward_scaled(self, obs: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        # Scaled alpha-beta to avoid underflow
        T = len(obs)
        N = self.n_states
        alpha = np.zeros((T, N), dtype=float)
        beta = np.zeros((T, N), dtype=float)
        c = np.zeros(T, dtype=float)  # scaling factors

        # Initialization
        alpha[0, :] = self.pi * self.B[:, obs[0]]
        c[0] = alpha[0, :].sum()
        c[0] = c[0] if c[0] > 0 else 1.0
        alpha[0, :] /= c[0]

        # Induction
        for t in range(1, T):
            alpha[t, :] = (alpha[t - 1, :] @ self.A) * self.B[:, obs[t]]
            c[t] = alpha[t, :].sum()
            c[t] = c[t] if c[t] > 0 else 1.0
            alpha[t, :] /= c[t]

        # Initialize beta
        beta[T - 1, :] = 1.0
        # Backward
        for t in range(T - 2, -1, -1):
            beta[t, :] = (self.A @ (self.B[:, obs[t + 1]] * beta[t + 1, :]))
            beta[t, :] /= c[t + 1]

        log_likelihood = float(np.sum(np.log(c)))
        return alpha, beta, c, log_likelihood

    def viterbi(self, obs: Sequence[int]) -> Tuple[List[int], float]:
        T = len(obs)
        N = self.n_states
        # Log domain to avoid underflow
        logA = np.log(np.maximum(self.A, 1e-300))
        logB = np.log(np.maximum(self.B, 1e-300))
        logpi = np.log(np.maximum(self.pi, 1e-300))

        delta = np.full((T, N), -np.inf)
        psi = np.zeros((T, N), dtype=int)

        delta[0, :] = logpi + logB[:, obs[0]]
        for t in range(1, T):
            for j in range(N):
                vals = delta[t - 1, :] + logA[:, j]
                psi[t, j] = int(np.argmax(vals))
                delta[t, j] = np.max(vals) + logB[j, obs[t]]

        best_path_prob = float(np.max(delta[T - 1, :]))
        states = [int(np.argmax(delta[T - 1, :]))]
        for t in range(T - 2, -1, -1):
            states.append(int(psi[t + 1, states[-1]]))
        states.reverse()
        return states, best_path_prob

    def em_baum_welch(
        self,
        sequences: List[Sequence[int]],
        n_iter: int = 50,
        tol: float = 1e-4,
        smoothing: float = 1e-3,
        verbose: bool = True,
        seed: Optional[int] = None,
    ) -> List[float]:
        """
        Train via EM on a list of observation sequences (discrete symbols).
        Returns list of log-likelihoods per iteration.
        """
        np.random.seed(seed)
        loglikes = []
        last_ll = -np.inf
        for it in range(n_iter):
            # E-step accumulators
            A_num = np.zeros_like(self.A)
            A_den = np.zeros(self.n_states, dtype=float)
            B_num = np.zeros_like(self.B)
            pi_accum = np.zeros(self.n_states, dtype=float)
            total_ll = 0.0

            for obs in sequences:
                obs = list(obs)
                T = len(obs)
                if T == 0:
                    continue
                alpha, beta, c, ll = self.forward_backward_scaled(obs)
                total_ll += ll
                # Gammas
                gamma = alpha * beta
                gamma = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), 1e-15)

                # Xis
                xi_sum = np.zeros_like(self.A)
                for t in range(T - 1):
                    # shape (i,j): alpha_t(i) * A(i,j) * B(j, o_{t+1}) * beta_{t+1}(j) / c_{t+1}
                    tmp = np.outer(alpha[t, :], self.B[:, obs[t + 1]] * beta[t + 1, :])
                    xi_t = self.A * tmp  # elementwise multiply A(i,j)* ...
                    xi_t /= np.maximum(np.sum(xi_t), 1e-15)
                    xi_sum += xi_t

                # Accumulate
                pi_accum += gamma[0, :]
                A_num += xi_sum
                A_den += np.maximum(gamma[:-1, :].sum(axis=0), 1e-15)

                for k in range(self.n_obs):
                    B_num[:, k] += gamma[np.array(obs) == k, :].sum(axis=0) if k in set(obs) else 0.0

            # M-step with smoothing
            self.pi = normalize_vector(pi_accum + smoothing)
            self.A = normalize_rows(A_num + smoothing)
            self.B = normalize_rows(B_num + smoothing)

            loglikes.append(total_ll)
            if verbose:
                print(f"[EM] Iter {it+1}/{n_iter} | log-likelihood = {total_ll:.4f}")
            if it > 0 and abs(total_ll - last_ll) < tol:
                if verbose:
                    print("Converged.")
                break
            last_ll = total_ll
        return loglikes

    def posterior_marginals(self, obs: Sequence[int]) -> Tuple[np.ndarray, float]:
        alpha, beta, c, ll = self.forward_backward_scaled(obs)
        gamma = alpha * beta
        gamma = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), 1e-15)
        return gamma, ll