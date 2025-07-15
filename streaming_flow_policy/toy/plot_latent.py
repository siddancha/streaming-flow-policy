from typing import List
import numpy as np
import matplotlib.pyplot as plt
import torch; torch.set_default_dtype(torch.double)
from torch import Tensor

from streaming_flow_policy.all import StreamingFlowPolicyLatentBase

def plot_probability_density_a(
        fp: StreamingFlowPolicyLatentBase,
        t: Tensor,
        a: Tensor,
        ax: plt.Axes,
        normalize: bool=True,
        alpha: float=1,
    ):
    """
    Args:
        fp (StreamingFlowPolicyLatent): Flow policy.
        t (Tensor, dtype=float, shape=(T, X)): Time values in [0,1].
        a (Tensor, dtype=float, shape=(T, X)): Action values.
        ax (plt.Axes): Axes to plot on.
        normalize (bool): Whether to normalize the probability density.
        alpha (float): Alpha value for the probability density.
    """
    p = fp.log_pdf_marginal_a(a.unsqueeze(-1), t).exp()  # (T, X)

    if normalize:
        p = p / p.max(dim=1, keepdims=True).values  # (T, X)

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    extent = [a.min(), a.max(), t.min(), t.max()]
    return ax.imshow(p, origin='lower', extent=extent, aspect='auto', alpha=alpha)

def plot_probability_density_z(
        fp: StreamingFlowPolicyLatentBase,
        t: Tensor,
        z: Tensor,
        ax: plt.Axes,
        normalize: bool=True,
        alpha: float=1,
    ):
    """
    Args:
        fp (StreamingFlowPolicyLatent): Flow policy.
        t (Tensor, dtype=float, shape=(T, X)): Time values in [0,1].
        z (Tensor, dtype=float, shape=(T, X)): Latent variable values.
        ax (plt.Axes): Axes to plot on.
        normalize (bool): Whether to normalize the probability density.
        alpha (float): Alpha value for the probability density.
    """
    p = fp.log_pdf_marginal_z(z.unsqueeze(-1), t).exp()  # (T, X)

    if normalize:
        p = p / p.max(dim=1, keepdims=True).values  # (T, X)

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    extent = [z.min(), z.max(), t.min(), t.max()]
    return ax.imshow(p, origin='lower', extent=extent, aspect='auto', alpha=alpha)

def plot_probability_density_and_streamlines_a(
        fp: StreamingFlowPolicyLatentBase,
        ax: plt.Axes,
        num_points: int=400,
    ):
    """
    Example of how to call the function:

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    im = plot_probability_density_and_streamlines_a(fp, ax)
    plt.colorbar(im, ax=ax, label='Probability Density')
    plt.show()
    """
    t = torch.linspace(0, 1, num_points, dtype=torch.double)  # (T,)
    a = torch.linspace(-1, 1, num_points, dtype=torch.double)  # (A,)
    t, a = torch.meshgrid(t, a, indexing='ij')  # (T, A)

    # Plot log probability
    heatmap = plot_probability_density_a(fp, t, a, ax)

    # Compute the expected velocity field of a over z.
    𝔼v = fp.𝔼va_marginal(a.unsqueeze(-1), t)  # (T, A, 1)
    𝔼v = 𝔼v.squeeze(-1)  # (T, A)

    # Plot streamlines
    ax.streamplot(
        x=a[0].numpy(),
        y=t[:, 0].numpy(),
        u=𝔼v.numpy(),
        v=np.ones(𝔼v.shape), 
        color='white', density=1, linewidth=0.5, arrowsize=0.75
    )

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Probability density and flow for action a')

    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    return heatmap

def plot_probability_density_and_streamlines_z(
        fp: StreamingFlowPolicyLatentBase,
        ax: plt.Axes,
        num_points: int=400,
    ):
    """
    Example of how to call the function:

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    im = plot_probability_density_and_streamlines_a(fp, ax)
    plt.colorbar(im, ax=ax, label='Probability Density')
    plt.show()
    """
    t = torch.linspace(0, 1, num_points, dtype=torch.double)  # (T,)
    z = torch.linspace(-1, 1, num_points, dtype=torch.double)  # (Z,)
    t, z = torch.meshgrid(t, z, indexing='ij')  # (T, Z)

    # Plot log probability
    heatmap = plot_probability_density_z(fp, t, z, ax)

    # Compute the expected velocity field of a over z.
    𝔼v = fp.𝔼vz_marginal(z.unsqueeze(-1), t)  # (T, Z, 1)
    𝔼v = 𝔼v.squeeze(-1)  # (T, Z)

    # Plot streamlines
    ax.streamplot(
        x=z[0].numpy(),
        y=t[:, 0].numpy(),
        u=𝔼v.numpy(),
        v=np.ones(𝔼v.shape), 
        color='white', density=1, linewidth=0.5, arrowsize=0.75
    )

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Probability density and flow for latent variable z')

    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    return heatmap

def plot_probability_density_with_trajectories(
        fp: StreamingFlowPolicyLatentBase,
        ax1: plt.Axes,
        ax2: plt.Axes,
        a_starts: List[float | None],
        z_starts: List[float],
        colors: List[str],
        linewidth_a: float=1,
        linewidth_z: float=1,
        alpha: float=0.5,
        heatmap_alpha: float=1,
        num_points_x: int=200,
        num_points_t: int=200,
        ode_steps: int=1000,
    ):
    t = torch.linspace(0, 1, num_points_t, dtype=torch.double)  # (T,)
    x = torch.linspace(-1, 1, num_points_x, dtype=torch.double)  # (X,)
    t, x = torch.meshgrid(t, x, indexing='ij')  # (T, X)

    # Plot density heatmaps in both panes.
    plot_probability_density_a(fp, t, x, ax1, alpha=heatmap_alpha)
    plot_probability_density_z(fp, t, x, ax2, alpha=heatmap_alpha)

    # Replace None with random samples from N(0, σ₀²)
    a_starts = [
        a_start if a_start is not None else np.random.randn() * fp.σ0
        for a_start in a_starts
    ]
    a_starts = torch.tensor(a_starts, dtype=torch.double).unsqueeze(-1)  # (L, 1)
    z_starts = torch.tensor(z_starts, dtype=torch.double).unsqueeze(-1)  # (L, 1)
    x_starts = torch.cat([a_starts, z_starts], dim=-1)  # (L, 2)
    list_traj = fp.ode_integrate(x_starts, num_steps=ode_steps)
    t = np.linspace(0, 1, num_points_t)  # (T,)
    for traj, color in zip(list_traj, colors):
        x = traj.vector_values(t)  # (2, T+1)
        a, z = x[0], x[1]  # (T+1,)
        ax1.plot(a, t, color=color, linewidth=linewidth_a, alpha=alpha)
        ax2.plot(z, t, color=color, linewidth=linewidth_z, alpha=alpha)

    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)

    ax1.set_title('Sampled trajectories: Action (a)', size='medium')
    ax1.set_xlabel('Action')
    ax1.set_ylabel('Time ⟶')

    ax2.set_title('Sampled trajectories: Latent Variable (z)', size='medium')
    ax2.set_xlabel('Latent Variable (z)')
    ax2.set_ylabel('Time ⟶')
