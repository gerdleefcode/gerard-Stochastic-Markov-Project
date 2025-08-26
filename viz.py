from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from .utils import ensure_dir


def set_style():
    sns.set_theme(style="whitegrid", context="talk", palette="deep")


def savefig(path: str, tight: bool = True, dpi: int = 150):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def plot_transition_heatmap(
    P: np.ndarray, state_names: List[str], title: str = "Transition Matrix", savepath: Optional[str] = None
):
    set_style()
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(P, annot=True, fmt=".2f", cmap="viridis", xticklabels=state_names, yticklabels=state_names)
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title(title)
    if savepath:
        savefig(savepath)


def plot_emission_heatmap(
    B: np.ndarray, state_names: List[str], obs_names: List[str], title: str = "Emission Matrix", savepath: Optional[str] = None
):
    set_style()
    plt.figure(figsize=(7, 5))
    ax = sns.heatmap(B, annot=True, fmt=".2f", cmap="magma", xticklabels=obs_names, yticklabels=state_names)
    ax.set_xlabel("Observation symbol")
    ax.set_ylabel("Hidden state")
    ax.set_title(title)
    if savepath:
        savefig(savepath)


def plot_stationary_bar(pi: np.ndarray, state_names: List[str], title: str = "Stationary Distribution", savepath: Optional[str] = None):
    set_style()
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=state_names, y=pi)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title(title)
    for i, v in enumerate(pi):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    if savepath:
        savefig(savepath)


def plot_state_graph(
    P: np.ndarray,
    state_names: List[str],
    title: str = "State Transition Graph",
    threshold: float = 0.05,
    savepath: Optional[str] = None,
):
    set_style()
    plt.figure(figsize=(7, 6))
    G = nx.DiGraph()
    n = len(state_names)
    for i, name in enumerate(state_names):
        G.add_node(i, label=name)
    for i in range(n):
        for j in range(n):
            w = float(P[i, j])
            if w >= threshold:
                G.add_edge(i, j, weight=w)

    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    widths = [2 + 8 * w for w in weights]

    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1200, edgecolors="k")
    nx.draw_networkx_labels(G, pos, labels={i: name for i, name in enumerate(state_names)})
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", width=widths, edge_color="gray")
    edge_labels = {(u, v): f"{w:.2f}" for (u, v), w in zip(G.edges(), weights)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    plt.title(title)
    plt.axis("off")
    if savepath:
        savefig(savepath)


def plot_sequence_states(
    seq: Sequence[int],
    state_names: List[str],
    title: str = "State Trajectory",
    savepath: Optional[str] = None,
):
    set_style()
    t = np.arange(len(seq))
    plt.figure(figsize=(10, 3))
    plt.step(t, seq, where="post", linewidth=2)
    plt.yticks(np.arange(len(state_names)), state_names)
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.title(title)
    if savepath:
        savefig(savepath)


def plot_decoding_comparison(
    true_states: Sequence[int],
    decoded_states: Sequence[int],
    state_names: List[str],
    title: str = "Viterbi Decoding vs True States",
    savepath: Optional[str] = None,
):
    set_style()
    t = np.arange(len(true_states))
    plt.figure(figsize=(10, 4))
    plt.step(t, true_states, where="post", label="True", linewidth=2)
    plt.step(t, decoded_states, where="post", label="Viterbi", linewidth=2, alpha=0.8)
    plt.yticks(np.arange(len(state_names)), state_names)
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.legend()
    plt.title(title)
    if savepath:
        savefig(savepath)


def plot_posterior_heatmap(
    gamma: np.ndarray,
    state_names: List[str],
    title: str = "Posterior State Probabilities",
    savepath: Optional[str] = None,
):
    set_style()
    plt.figure(figsize=(10, 4))
    ax = sns.heatmap(gamma.T, cmap="viridis", cbar=True, vmin=0, vmax=1)
    ax.set_ylabel("State")
    ax.set_xlabel("Time")
    ax.set_yticklabels(state_names, rotation=0)
    ax.set_title(title)
    if savepath:
        savefig(savepath)


def plot_learning_curve(loglikes: List[float], title: str = "EM Learning Curve", savepath: Optional[str] = None):
    set_style()
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(loglikes) + 1), loglikes, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Log-likelihood")
    plt.title(title)
    if savepath:
        savefig(savepath)