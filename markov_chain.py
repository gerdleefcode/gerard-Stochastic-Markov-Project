from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx

from .utils import (
    normalize_rows,
    stationary_distribution,
    mixing_time_upper_bound,
    random_stochastic_matrix,
    random_categorical,
)


@dataclass
class MarkovChain:
    P: np.ndarray
    state_names: Optional[List[str]] = None

    def __post_init__(self):
        self.P = np.asarray(self.P, dtype=float)
        n = self.P.shape[0]
        assert self.P.shape == (n, n), "P must be square"
        self.P = normalize_rows(self.P)
        if self.state_names is None:
            self.state_names = [f"S{i}" for i in range(n)]
        assert len(self.state_names) == n, "state_names length must match P"

    @property
    def n_states(self) -> int:
        return self.P.shape[0]

    @staticmethod
    def fit_from_sequences(
        sequences: List[Sequence[int]],
        n_states: Optional[int] = None,
        smoothing: float = 1.0,
        state_names: Optional[List[str]] = None,
    ) -> "MarkovChain":
        if n_states is None:
            n_states = 1 + max(max(seq) for seq in sequences)
        counts = np.zeros((n_states, n_states), dtype=float)
        for seq in sequences:
            for a, b in zip(seq[:-1], seq[1:]):
                counts[a, b] += 1.0
        P = counts + smoothing  # Laplace/Dirichlet smoothing
        P = normalize_rows(P)
        if state_names is None:
            state_names = [f"S{i}" for i in range(n_states)]
        return MarkovChain(P=P, state_names=state_names)

    @staticmethod
    def random(n_states: int, alpha: float = 1.0, state_names: Optional[List[str]] = None) -> "MarkovChain":
        P = random_stochastic_matrix(n_states, alpha=alpha)
        if state_names is None:
            state_names = [f"S{i}" for i in range(n_states)]
        return MarkovChain(P=P, state_names=state_names)

    def stationary(self) -> np.ndarray:
        return stationary_distribution(self.P)

    def mixing_time(self, eps: float = 1e-3) -> Tuple[int, float]:
        return mixing_time_upper_bound(self.P, eps=eps)

    def sample(
        self,
        n_steps: int,
        init_dist: Optional[np.ndarray] = None,
        start_state: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[int]:
        np.random.seed(seed)
        if start_state is not None:
            s = int(start_state)
        else:
            if init_dist is None:
                init_dist = self.stationary()
            init_dist = np.asarray(init_dist, dtype=float)
            init_dist = init_dist / init_dist.sum()
            s = int(np.random.choice(self.n_states, p=init_dist))
        traj = [s]
        for _ in range(n_steps - 1):
            s = int(np.random.choice(self.n_states, p=self.P[s]))
            traj.append(s)
        return traj

    def log_likelihood(self, sequence: Sequence[int]) -> float:
        seq = list(sequence)
        if len(seq) < 2:
            return 0.0
        logp = 0.0
        for a, b in zip(seq[:-1], seq[1:]):
            p = max(self.P[a, b], 1e-300)
            logp += np.log(p)
        return float(logp)

    def to_networkx_graph(self, threshold: float = 0.01) -> nx.DiGraph:
        G = nx.DiGraph()
        for i, name in enumerate(self.state_names):
            G.add_node(i, label=name)
        for i in range(self.n_states):
            for j in range(self.n_states):
                w = float(self.P[i, j])
                if w >= threshold:
                    G.add_edge(i, j, weight=w)
        return G