from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np
import torch; torch.set_default_dtype(torch.double)
from torch import Tensor
from torch.distributions import MultivariateNormal
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
        self.π = torch.tensor(prior, dtype=torch.double)  # (K,)

    @staticmethod
    def ξt(traj: Trajectory, t: Tensor) -> Tensor:
        """
        Args:
            traj (Trajectory): Demonstration trajectory.
            t (Tensor, dtype=double, shape=(*BS)): Time values in [0,1].

        Returns:
            Tensor, dtype=double, shape=(*BS, D): Configuration values at time t.
        """
        BS = t.shape
        ξt = traj.vector_values(t.ravel().numpy())  # (D, *BS)
        ξt = torch.tensor(ξt, dtype=torch.double).reshape(-1, *BS)  # (D, *BS)
        ξt = ξt.movedim(0, -1)  # (*BS, D)
        return ξt

    @staticmethod
    def ξ̇t(traj: Trajectory, t: Tensor) -> Tensor:
        """
        Args:
            traj (Trajectory): Demonstration trajectory.
            t (Tensor, dtype=double, shape=(*BS)): Time values in [0,1].

        Returns:
            Tensor, dtype=double, shape=(*BS, D): Velocity at time t.
        """
        BS = t.shape
        traj_ξ̇ = traj.MakeDerivative()
        ξ̇t = traj_ξ̇.vector_values(t.ravel().numpy())  # (D, *BS)
        ξ̇t = torch.tensor(ξ̇t, dtype=torch.double).reshape(-1, *BS)  # (D, *BS)
        ξ̇t = ξ̇t.movedim(0, -1)  # (*BS, D)
        return ξ̇t

    @staticmethod
    def matrix_stack(grid: List[List[Tensor]]) -> Tensor:
        """
        Args:
            grid (List[List[(Tensor, dtype=double, shape=(*BS, D))]]): grid of
                tensors to be stacked into a matrix.

        Returns:
            (Tensor, dtype=double, shape=(*BS, D, D)): Stacked tensors.
        """
        # First, convert all cells into tensors.
        grid = [[cell if isinstance(cell, Tensor) else torch.tensor(cell, dtype=torch.double) for cell in row] for row in grid]

        # Compute batch shape.
        BS = max((cell.shape for row in grid for cell in row), default=())

        # Broadcast all tensors to the same shape.
        grid = [[cell.expand(*BS) for cell in row] for row in grid]

        # Stack grid into a matrix tensor.
        return torch.stack([torch.stack(row, dim=-1) for row in grid], dim=-2)  # (*BS, D, D)

    @abstractmethod
    def Ab(self, traj: Trajectory, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            traj (Trajectory): Demonstration trajectory.
            t (Tensor, dtype=double, shape=(*BS)): Time values in [0,1].

        Returns:
            A (Tensor, dtype=double, shape=(*BS, D, D)): Transition matrix.
            b (Tensor, dtype=double, shape=(*BS, D)): Bias vector.
        """
        return NotImplementedError

    @abstractmethod
    def μΣ0(self, traj: Trajectory) -> Tuple[Tensor, Tensor]:
        """
        Compute the mean and covariance matrix of the conditional flow at time t=0.

        Returns:
            Tensor, dtype=double, shape=(D,): Mean at time t=0.
            Tensor, dtype=double, shape=(D, D): Covariance matrix at time t=0.
        """
        return NotImplementedError

    def μΣt(self, traj: Trajectory, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute the mean and covariance matrix of the conditional flows at time t.

        Args:
            traj (Trajectory): Demonstration trajectory.
            t (np.ndarray, dtype=float, shape=(*BS)): Time values in [0,1].

        Returns:
            Tensor, dtype=double, shape=(*BS, D): Mean at time t.
            Tensor, dtype=double, shape=(*BS, D, D): Covariance matrix at time t.
        """
        μ0, Σ0 = self.μΣ0(traj)  # (D,) and (D, D)
        A, b = self.Ab(traj, t)  # (*BS, D, D) and (*BS, D)
        AT = A.transpose(-1, -2)  # (*BS, D, D)
        μt = A @ μ0 + b  # (*BS, D)
        Σt = A @ Σ0 @ AT  # (*BS, D, D)
        return μt, Σt  # (*BS, D) and (*BS, D, D)

    def log_pdf_conditional(self, traj: Trajectory, x: Tensor, t: Tensor) -> Tensor:
        """
        Compute log-probability of the conditional flow at state x and time t for
        the given trajectory.

        Args:
            traj (Trajectory): Demonstration trajectory.
            x (Tensor, dtype=double, shape=(*BS, D)): State values.
            t (Tensor, dtype=double, shape=(*BS)): Time values in [0,1].

        Returns:
            (Tensor, dtype=double, shape=(*BS)): Probability of the conditional
                flow at state x and time t.
        """
        assert x.shape[-1] == self.D
        μt, Σt = self.μΣt(traj, t)  # (*BS, D) and (*BS, D, D)
        dist = MultivariateNormal(loc=μt, covariance_matrix=Σt)  # BS=(*BS) ES=(D,)
        return dist.log_prob(x)  # (*BS)

    def log_pdf_marginal(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Compute log-probability of the marginal flow at state x and time t.

        Args:
            x (Tensor, dtype=double, shape=(*BS, D)): State values.
            t (Tensor, dtype=double, shape=(*BS)): Time values in [0,1].

        Returns:
            (Tensor, dtype=double, shape=(*BS)): Log-probability of the marginal
                flow at state x and time t.
        """
        log_pdf = torch.tensor(-torch.inf, dtype=torch.double)
        for π, traj in zip(self.π, self.trajectories):
            log_pdf_i = π.log() + self.log_pdf_conditional(traj, x, t)  # (*BS)
            log_pdf = torch.logaddexp(log_pdf, log_pdf_i)  # (*BS)
        return log_pdf  # (*BS)

    @abstractmethod
    def u_conditional(self, traj: Trajectory, x: Tensor, t: Tensor) -> Tensor:
        """
        Compute the conditional velocity field for a given trajectory.

        Args:
            traj (Trajectory): Demonstration trajectory.
            x (Tensor, dtype=double, shape=(*BS, D)): State values.
            t (Tensor, dtype=double, shape=(*BS)): Time values in [0,1].

        Returns:
            (Tensor, dtype=double, shape=(*BS, D)): Velocity of the conditional flow.
        """
        raise NotImplementedError

    def u_marginal(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x (Tensor, dtype=double, shape=(*BS, D)): State values.
            t (Tensor, dtype=double, shape=(*BS)): Time values in [0,1].

        Returns:
            (Tensor, dtype=double, shape=(D,)): Marginal velocities.
        """
        log_likelihoods = torch.stack([self.log_pdf_conditional(traj, x, t) for traj in self.trajectories], dim=-1)  # (*BS, K)
        velocities = torch.stack([self.u_conditional(traj, x, t) for traj in self.trajectories], dim=-2)  # (*BS, K, D)

        log_posterior = self.π.log() + log_likelihoods  # (*BS, K)
        log_partition_fn = log_posterior.logsumexp(dim=-1, keepdim=True)  # (*BS, 1)
        log_posterior = log_posterior - log_partition_fn  # (*BS, K)
        posterior = log_posterior.exp().unsqueeze(-1)  # (*BS, K, 1)
        assert posterior.isfinite().all()

        us = (posterior * velocities).sum(dim=-2)  # (*BS, D)
        assert us.isfinite().all()

        return us

    def ode_integrate(self, x: Tensor, num_steps: int = 1000) -> List[Trajectory]:
        """
        Args:
            x (Tensor, dtype=double, shape=(L, D)): Initial state.
            num_steps (int): Number of steps to integrate.
            
        Returns:
            List[Trajectory]: Trajectories starting from x.
        """
        L = x.shape[0]
        breaks = np.linspace(0.0, 1.0, num_steps + 1)  # (N+1,)
        Δt = 1.0 / num_steps
        multi_traj = [x]
        for t in breaks[:-1]:
            u = self.u_marginal(x, torch.ones([L]) * t)  # (L, D)
            x = x + Δt * u  # (L, D)
            multi_traj.append(x)
        multi_traj = torch.stack(multi_traj, dim=-2)  # (L, N+1, D)
        multi_traj = multi_traj.numpy()  # (L, N+1, D)
        return [
            PiecewisePolynomial.FirstOrderHold(breaks, traj.T)
            for traj in multi_traj
        ]
