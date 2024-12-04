from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import multivariate_normal

from pydrake.all import PiecewisePolynomial, Trajectory


class StreamingFlowPolicyBase (ABC):
    def __init__(
        self,
        dim: int,
        trajectories: List[Trajectory],
        prior: List[float],
    ):
        """
        Args:
            dim (int): Dimension of the state space.
            trajectories (List[Trajectory]): List of trajectories.
            prior (np.ndarray, dtype=float, shape=(K,)): Prior
                probabilities for each trajectory.
        """
        self.D = dim
        self.trajectories = trajectories
        self.π = np.array(prior)  # (K,)

    @abstractmethod
    def Ab(self, traj: Trajectory, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            A (np.ndarray, dtype=float, shape=(D, D)): Transition matrix.
            b (np.ndarray, dtype=float, shape=(D,)): Bias vector.
        """
        return NotImplementedError

    @abstractmethod
    def μΣ0(self, traj: Trajectory) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the mean and covariance matrix of the conditional flow at time t=0.

        Returns:
            np.ndarray, dtype=float, shape=(D,): Mean at time t=0.
            np.ndarray, dtype=float, shape=(D, D): Covariance matrix at time t=0.
        """
        return NotImplementedError

    def μΣt(self, traj: Trajectory, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the mean and covariance matrix of the conditional flows at time t.
        
        Args:
            traj (Trajectory): Demonstration trajectory.
            t (float): Time value in [0,1].
            
        Returns:
            np.ndarray, dtype=float, shape=(D,): Mean at time t.
            np.ndarray, dtype=float, shape=(D, D): Covariance matrix at time t.
        """
        μ0, Σ0 = self.μΣ0(traj)
        A, b = self.Ab(traj, t)
        μt = A @ μ0 + b
        Σt = A @ Σ0 @ A.T
        return μt, Σt

    def pdf_conditional(self, traj: Trajectory, x: np.ndarray, t: float) -> float:
        """
        Compute probability of the conditional flow at state x and time t for
        the given trajectory.

        Args:
            traj (Trajectory): Demonstration trajectory.
            x (np.ndarray, dtype=float, shape=(D,)): State values.
            t (float): Time value in [0,1].

        Returns:
            float: Probability of the conditional flow at state x and time t.
        """
        assert x.shape == (self.D,)
        μt, Σt = self.μΣt(traj, t)  # (D,) and (D, D)
        dist = multivariate_normal(mean=μt, cov=Σt)
        return dist.pdf(x)  # (D,)

    def pdf_marginal(self, x: np.ndarray, t: float) -> float:
        """
        Compute probability of the marginal flow at state x and time t.

        Args:
            x (np.ndarray, dtype=float, shape=(D,)): State values.
            t (float): Time value in [0,1].

        Returns:
            float: Probability of the marginal flow at state x and time t.
        """
        prob = 0
        for π, traj in zip(self.π, self.trajectories):
            prob += π * self.pdf_conditional(traj, x, t)
        return prob

    @abstractmethod
    def u_conditional(self, traj: Trajectory, x: np.ndarray, t: float) -> np.ndarray:
        """
        Compute the conditional velocity field for a given trajectory.

        Args:
            traj (Trajectory): Demonstration trajectory.
            x (np.ndarray, dtype=float, shape=(D,)): State values.
            t (float): Time value in [0,1].

        Returns:
            (np.ndarray, dtype=float, shape=(D,)): Velocity of the conditional flow.
        """
        raise NotImplementedError

    def u_marginal(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Args:
            x (np.ndarray, dtype=float, shape=(D,)): State values.
            t (float): Time value in [0,1].

        Returns:
            (np.ndarray, dtype=float, shape=(D,)): Marginal velocities.
        """
        likelihoods = np.hstack([self.pdf_conditional(traj, x, t) for traj in self.trajectories])  # (K,)
        velocities = np.vstack([self.u_conditional(traj, x, t) for traj in self.trajectories])  # (K, D)

        posterior = self.π * likelihoods  # (K,)
        normalizing_constant: np.ndarray = np.sum(posterior)  # (,)
        posterior = posterior / normalizing_constant  # (K,)
        posterior = posterior.reshape(-1, 1)  # (K, 1)

        us = (posterior * velocities).sum(axis=0)  # (D,)
        return us

    def ode_integrate(self, x: np.ndarray, num_steps: int = 1000) -> Trajectory:
        """
        Args:
            x (np.ndarray, dtype=float, shape=(D,)): Initial state.
            num_steps (int): Number of steps to integrate.
            
        Returns:
            Trajectory: Trajectory starting from x.
        """
        breaks = np.linspace(0.0, 1.0, num_steps + 1)  # (N+1,)
        Δt = 1.0 / num_steps
        samples = [x]
        for t in breaks[:-1]:
            u = self.u_marginal(x, t)
            x = x + Δt * u
            samples.append(x)
        samples = np.vstack(samples)  # (N+1, 2)
        return PiecewisePolynomial.FirstOrderHold(breaks, samples.T)
