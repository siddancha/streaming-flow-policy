from typing import List
import numpy as np
import matplotlib.pyplot as plt
import torch; torch.set_default_dtype(torch.double)
from torch import Tensor

from flow_policy.toy.sfps import StreamingFlowPolicyStochastic

def plot_probability_density_q(
        fp: StreamingFlowPolicyStochastic,
        ts: Tensor,
        xs: Tensor,
        ax: plt.Axes,
        normalize: bool=True,
        alpha: float=1,
    ):
    """
    Args:
        fp (StreamingFlowPolicyStochastic): Flow policy.
        ts (Tensor, dtype=float, shape=(T, X)): Time values in [0,1].
        xs (Tensor, dtype=float, shape=(T, X)): Configuration values.
        ax (plt.Axes): Axes to plot on.
        normalize (bool): Whether to normalize the probability density.
        alpha (float): Alpha value for the probability density.
    """
    p = fp.log_pdf_marginal_q(xs.unsqueeze(-1), ts).exp()  # (T, X)

    if normalize:
        p = p / p.max(dim=1, keepdims=True).values  # (T, X)

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    extent = [xs.min(), xs.max(), ts.min(), ts.max()]
    return ax.imshow(p, origin='lower', extent=extent, aspect='auto', alpha=alpha)

def plot_probability_density_z(
        fp: StreamingFlowPolicyStochastic,
        ts: Tensor,
        xs: Tensor,
        ax: plt.Axes,
        normalize: bool=True,
        alpha: float=1,
    ):
    """
    Args:
        fp (StreamingFlowPolicyStochastic): Flow policy.
        ts (Tensor, dtype=float, shape=(T, X)): Time values in [0,1].
        xs (Tensor, dtype=float, shape=(T, X)): Configuration values.
        ax (plt.Axes): Axes to plot on.
        normalize (bool): Whether to normalize the probability density.
        alpha (float): Alpha value for the probability density.
    """
    p = fp.log_pdf_marginal_z(xs.unsqueeze(-1), ts).exp()  # (T, X)

    if normalize:
        p = p / p.max(dim=1, keepdims=True).values  # (T, X)

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    extent = [xs.min(), xs.max(), ts.min(), ts.max()]
    return ax.imshow(p, origin='lower', extent=extent, aspect='auto', alpha=alpha)


def plot_probability_density_with_trajectories(
        fp: StreamingFlowPolicyStochastic,
        ax1: plt.Axes,
        ax2: plt.Axes,
        q_starts: List[float | None],
        z_starts: List[float],
        colors: List[str],
        linewidth_q: float=1,
        linewidth_z: float=1,
        alpha: float=0.5,
        heatmap_alpha: float=1,
        num_points_x: int=200,
        num_points_t: int=200,
        ode_steps: int=1000,
    ):
    ts = torch.linspace(0, 1, num_points_t, dtype=torch.double)  # (T,)
    xs = torch.linspace(-1, 1, num_points_x, dtype=torch.double)  # (X,)
    ts, xs = torch.meshgrid(ts, xs, indexing='ij')  # (T, X)

    # Plot density heatmaps in both panes.
    plot_probability_density_q(fp, ts, xs, ax1, alpha=heatmap_alpha)
    plot_probability_density_z(fp, ts, xs, ax2, alpha=heatmap_alpha)

    # Replace None with random samples from N(0, σ₀²)
    q_starts = [
        q_start if q_start is not None else np.random.randn() * fp.σ0
        for q_start in q_starts
    ]
    q_starts = torch.tensor(q_starts, dtype=torch.double).unsqueeze(-1)  # (L, 1)
    z_starts = torch.tensor(z_starts, dtype=torch.double).unsqueeze(-1)  # (L, 1)
    x_starts = torch.cat([q_starts, z_starts], dim=-1)  # (L, 2)
    list_traj = fp.ode_integrate(x_starts, num_steps=ode_steps)
    ts = np.linspace(0, 1, num_points_t)  # (T,)
    for traj, color in zip(list_traj, colors):
        xs = traj.vector_values(ts)  # (2, T+1)
        qs, zs = xs[0], xs[1]  # (T+1,)
        ax1.plot(qs, ts, color=color, linewidth=linewidth_q, alpha=alpha)
        ax2.plot(zs, ts, color=color, linewidth=linewidth_z, alpha=alpha)

    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)

    ax1.set_title('Sampled trajectories: Configuration (q)', size='medium')
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Time ⟶')

    ax2.set_title('Sampled trajectories: Latent Variable (z)', size='medium')
    ax2.set_xlabel('Latent Variable (z)')
    ax2.set_ylabel('Time ⟶')
