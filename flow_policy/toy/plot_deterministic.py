from typing import List
import numpy as np
import matplotlib.pyplot as plt
import torch; torch.set_default_dtype(torch.double)
from torch import Tensor

from flow_policy.toy.sfpd import StreamingFlowPolicyDeterministic

def plot_probability_density(
        fp: StreamingFlowPolicyDeterministic,
        ts: Tensor,
        xs: Tensor,
        ax: plt.Axes,
        normalize: bool=True,
        alpha: float=1,
        aspect: str | float = 2,
    ):
    """
    Args:
        fp (StreamingFlowPolicyDeterministic): Flow policy.
        ts (Tensor, dtype=float, shape=(T, X)): Time values in [0,1].
        xs (Tensor, dtype=float, shape=(T, X)): Configuration values.
        ax (plt.Axes): Axes to plot on.
        normalize (bool): Whether to normalize the probability density.
        alpha (float): Alpha value for the probability density.
        aspect (str | float): Aspect ratio.
    """
    p = fp.log_pdf_marginal(xs.unsqueeze(-1), ts).exp()  # (T, X)

    if normalize:
        p = p / p.max(dim=1, keepdims=True).values  # (T, X)

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time ⟶')

    extent = [xs.min(), xs.max(), ts.min(), ts.max()]
    return ax.imshow(p, origin='lower', extent=extent, aspect=aspect, alpha=alpha)

def plot_probability_density_and_vector_field(
        fp: StreamingFlowPolicyDeterministic,
        ax: plt.Axes,
        num_points: int=200,
        num_quiver: int=20,
    ):
    """
    Example of how to call the function:

    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    im = plot_probability_density_and_vector_field(fp, ax)
    plt.colorbar(im, ax=ax, label='Probability Density')
    plt.show()
    """
    ts = torch.linspace(0, 1, num_points, dtype=torch.double)  # (T,)
    xs = torch.linspace(-1, 1, num_points, dtype=torch.double)  # (X,)
    ts, xs = torch.meshgrid(ts, xs, indexing='ij')  # (T, X)

    # Plot probability density
    heatmap = plot_probability_density(fp, ts, xs, ax)

    # Compute marginal velocity field.
    u = fp.u_marginal(xs.unsqueeze(-1), ts)  # (T, X, 1)
    u = u.squeeze(-1)  # (T, X)

    # Plot quiver with reduced size
    quiver_step_x = xs.shape[1] // num_quiver
    quiver_step_t = ts.shape[0] // num_quiver
    ax.quiver(
        xs[::quiver_step_t, ::quiver_step_x],
        ts[::quiver_step_t, ::quiver_step_x], 
        u[::quiver_step_t, ::quiver_step_x],
        np.ones_like(u)[::quiver_step_t, ::quiver_step_x], 
        color='white', scale=40, width=0.002, headwidth=3, headlength=4
    )

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Probability Density and Vector Field')

    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    return heatmap

def plot_probability_density_and_streamlines(
        fp: StreamingFlowPolicyDeterministic,
        ax: plt.Axes,
        num_points: int=400,
    ):
    """
    Example of how to call the function:

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    im = plot_probability_density_and_streamlines(fp, ax)
    plt.colorbar(im, ax=ax, label='Probability Density')
    plt.show()
    """
    ts = torch.linspace(0, 1, num_points, dtype=torch.double)  # (T,)
    xs = torch.linspace(-1, 1, num_points, dtype=torch.double)  # (X,)
    ts, xs = torch.meshgrid(ts, xs, indexing='ij')  # (T, X)

    # Plot log probability
    heatmap = plot_probability_density(fp, ts, xs, ax)

    # Compute marginal velocity field.
    u = fp.u_marginal(xs.unsqueeze(-1), ts)  # (T, X, 1)
    u = u.squeeze(-1)  # (T, X)

    # Plot streamlines
    ax.streamplot(
        x=xs[0].numpy(),
        y=ts[:, 0].numpy(),
        u=u.numpy(),
        v=np.ones(u.shape), 
        color='white', density=1, linewidth=0.5, arrowsize=0.5
    )

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Probability Density and Flow')

    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    return heatmap

def plot_probability_density_with_trajectories(
        fp: StreamingFlowPolicyDeterministic,
        ax: plt.Axes,
        x_starts: List[float | None],
        linewidth: float=1,
        alpha: float=0.5,
        heatmap_alpha: float=1,
    ):
    ts = torch.linspace(0, 1, 200, dtype=torch.double)  # (T,)
    xs = torch.linspace(-1, 1, 200, dtype=torch.double)  # (X,)
    ts, xs = torch.meshgrid(ts, xs, indexing='ij')  # (T, X)

    heatmap = plot_probability_density(fp, ts, xs, ax, alpha=heatmap_alpha)

    # Replace None with random samples from N(0, σ₀²)
    x_starts = [
        x_start if x_start is not None else np.random.randn() * fp.σ0
        for x_start in x_starts
    ]
    x_starts = torch.tensor(x_starts, dtype=torch.double).unsqueeze(-1)  # (L, 1)
    list_traj = fp.ode_integrate(x_starts)
    ts = np.linspace(0, 1, 200)  # (T,)
    for traj in list_traj:
        xs = traj.vector_values(ts)  # (1, T+1)
        ax.plot(xs[0], ts, color='red', linewidth=linewidth, alpha=alpha)

    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Trajectories sampled from flow')

    return heatmap
