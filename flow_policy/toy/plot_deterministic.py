import numpy as np
import matplotlib.pyplot as plt
from typing import List

from flow_policy.toy.flow_policy import StreamingFlowPolicyDeterministic

def plot_probability_density(
        fp: StreamingFlowPolicyDeterministic,
        ts: np.ndarray,
        xs: np.ndarray,
        ax: plt.Axes,
        normalize: bool=True,
        alpha: float=1,
        aspect: str | float = 2,
    ):
    p = np.zeros((len(ts), len(xs)))  # (T, X)
    for i in range(len(ts)):
        for j in range(len(xs)):
            p[i,j] = fp.pdf_marginal(xs[[j]], ts[i])

    if normalize:
        p = p / p.max(axis=1, keepdims=True)  # (T, X)

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
    ts = np.linspace(0, 1, num_points)  # (T,)
    xs = np.linspace(-1, 1, num_points)  # (X,)

    # Plot probability density
    heatmap = plot_probability_density(fp, ts, xs, ax)

    # Compute marginal velocity field.
    u = np.zeros((len(ts), len(xs)))  # (T, X)
    for i in range(len(ts)):
        for j in range(len(xs)):
            u[i,j] = fp.u_marginal(xs[[j]], ts[i])

    # Plot quiver with reduced size
    ts, xs = np.meshgrid(ts, xs, indexing='ij')  # (T, X)
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
    ts = np.linspace(0, 1, num_points)  # (T,)
    xs = np.linspace(-1, 1, num_points)  # (X,)

    # Plot log probability
    heatmap = plot_probability_density(fp, ts, xs, ax)

    # Compute marginal velocity field.
    u = np.zeros((len(ts), len(xs)))  # (T, X)
    for i in range(len(ts)):
        for j in range(len(xs)):
            u[i,j] = fp.u_marginal(xs[[j]], ts[i])

    # Plot streamlines
    ts, xs = np.meshgrid(ts, xs, indexing='ij')  # (T, X)
    ax.streamplot(x=xs[0], y=ts[:, 0], u=u, v=np.ones_like(u), 
                  color='white', density=1, linewidth=0.5, arrowsize=0.5)

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Probability Density and Flow')

    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    return heatmap

def plot_probability_density_with_trajectories(
        fp: StreamingFlowPolicyDeterministic,
        ax: plt.Axes,
        xs_start: List[float | None],
        linewidth: float=1,
        alpha: float=0.5,
        heatmap_alpha: float=1,
    ):
    ts = np.linspace(0, 1, 200)  # (T,)
    xs = np.linspace(-1, 1, 200)  # (X,)

    heatmap = plot_probability_density(fp, ts, xs, ax, alpha=heatmap_alpha)

    for x_start in xs_start:
        x_start = x_start if x_start is not None else np.random.randn() * fp.σ0
        traj = fp.ode_integrate(np.array([x_start]))
        xs = traj.vector_values(ts)
        ax.plot(xs[0], ts, color='red', linewidth=linewidth, alpha=alpha)

    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Trajectories sampled from flow')

    return heatmap
