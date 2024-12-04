import numpy as np
import matplotlib.pyplot as plt
from typing import List

from flow_policy.toy.stochastic_flow_policy import StochasticFlowPolicy

def plot_probability_density_q(
        fp: StochasticFlowPolicy,
        ts: np.ndarray,
        xs: np.ndarray,
        ax: plt.Axes,
        normalize: bool=True,
        alpha: float=1,
    ):
    p = np.zeros((len(ts), len(xs)))  # (T, X)
    for i in range(len(ts)):
        for j in range(len(xs)):
            p[i,j] = fp.pdf_marginal_q(xs[j], ts[i])

    if normalize:
        p = p / p.max(axis=1, keepdims=True)  # (T, X)

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    extent = [xs[0], xs[-1], ts[0], ts[-1]]
    return ax.imshow(p, origin='lower', extent=extent, aspect='auto', alpha=alpha)

def plot_probability_density_z(
        fp: StochasticFlowPolicy,
        ts: np.ndarray,
        xs: np.ndarray,
        ax: plt.Axes,
        normalize: bool=True,
        alpha: float=1,
    ):
    p = np.zeros((len(ts), len(xs)))  # (T, X)
    for i in range(len(ts)):
        for j in range(len(xs)):
            p[i,j] = fp.pdf_marginal_z(xs[j], ts[i])
    
    if normalize:
        p = p / p.max(axis=1, keepdims=True)

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    extent = [xs[0], xs[-1], ts[0], ts[-1]]
    return ax.imshow(p, origin='lower', extent=extent, aspect='auto', alpha=alpha)


def plot_probability_density_with_trajectories_q(
        fp: StochasticFlowPolicy,
        ax: plt.Axes,
        q_starts: List[float | None],
        z_starts: List[float],
        colors: List[str],
        linewidth: float=1,
        alpha: float=0.5,
        heatmap_alpha: float=1,
        num_points_x: int=200,
        num_points_t: int=200,
        ode_steps: int=1000,
    ):
    ts = np.linspace(0, 1, num_points_t)  # (T,)
    xs = np.linspace(-1, 1, num_points_x)  # (X,)
    heatmap = plot_probability_density_q(fp, ts, xs, ax, alpha=heatmap_alpha)

    for q_start, z_start, color in zip(q_starts, z_starts, colors):
        q_start = q_start if q_start is not None else np.random.randn() * fp.σ0
        traj = fp.ode_integrate(np.array([q_start, z_start]), num_steps=ode_steps)
        xs = traj.vector_values(ts)  # (2, N+1)
        qs = xs[0]  # (N+1,)
        ax.plot(qs, ts, color=color, linewidth=linewidth, alpha=alpha)

    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Configuration (q) trajectories', size='medium')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time ⟶')

    return heatmap


def plot_probability_density_with_trajectories_z(
        fp: StochasticFlowPolicy,
        ax: plt.Axes,
        q_starts: List[float | None],
        z_starts: List[float],
        colors: List[str],
        linewidth: float=1,
        alpha: float=0.5,
        heatmap_alpha: float=1,
        num_points_x: int=200,
        num_points_t: int=200,
        ode_steps: int=1000,
    ):
    ts = np.linspace(0, 1, num_points_t)  # (T,)
    xs = np.linspace(-1, 1, num_points_x)  # (X,)
    heatmap = plot_probability_density_z(fp, ts, xs, ax, alpha=heatmap_alpha)

    for q_start, z_start, color in zip(q_starts, z_starts, colors):
        q_start = q_start if q_start is not None else np.random.randn() * fp.σ0
        traj = fp.ode_integrate(np.array([q_start, z_start]), num_steps=ode_steps)
        xs = traj.vector_values(ts)  # (2, N+1)
        zs = xs[1]  # (N+1,)
        ax.plot(zs, ts, color=color, linewidth=linewidth, alpha=alpha)

    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Latent Variable (z) trajectories', size='medium')
    ax.set_xlabel('Latent Variable (z)')
    ax.set_ylabel('Time ⟶')

    return heatmap
