from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np
import torch
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
        self.X = dim  # dimension of the full state
        self.trajectories = trajectories
        self.π = torch.tensor(prior, dtype=torch.get_default_dtype())  # (K,)

    @staticmethod
    def ξt(traj: Trajectory, t: Tensor) -> Tensor:
        """
        Args:
            traj (Trajectory): Demonstration trajectory.
            t (Tensor, dtype=default, shape=(*BS)): Time values in [0,1].

        Returns:
            Tensor, dtype=default, shape=(*BS, X): Action values at time t.
        """
        BS = t.shape
        ξt = traj.vector_values(t.ravel().cpu().numpy())  # (X, *BS)
        ξt = torch.tensor(ξt, dtype=torch.get_default_dtype()).reshape(-1, *BS)  # (X, *BS)
        ξt = ξt.movedim(0, -1)  # (*BS, X)
        return ξt

    @staticmethod
    def ξ̇t(traj: Trajectory, t: Tensor) -> Tensor:
        """
        Args:
            traj (Trajectory): Demonstration trajectory.
            t (Tensor, dtype=default, shape=(*BS)): Time values in [0,1].

        Returns:
            Tensor, dtype=default, shape=(*BS, X): Velocity at time t.
        """
        BS = t.shape
        traj_ξ̇ = traj.MakeDerivative()
        ξ̇t = traj_ξ̇.vector_values(t.ravel().cpu().numpy())  # (X, *BS)
        ξ̇t = torch.tensor(ξ̇t, dtype=torch.get_default_dtype()).reshape(-1, *BS)  # (X, *BS)
        ξ̇t = ξ̇t.movedim(0, -1)  # (*BS, X)
        return ξ̇t

    @staticmethod
    def block_matrix(grid: List[List[Tensor]]) -> Tensor:
        """
        Args:
            grid (List[List[(Tensor, dtype=default, shape=(*BS, D, D))]]): grid
                of tensor blocks to be concatenated into a larger matrix.

        Returns:
            (Tensor, dtype=default, shape=(*BS, M * D, N * D)): Concatenated
                block matrix.
        """
        # First, convert all cells into tensors.
        grid = [[cell if isinstance(cell, Tensor) else torch.tensor(cell, dtype=torch.get_default_dtype()) for cell in row] for row in grid]

        # Compute max shape.
        max_shape = max((cell.shape for row in grid for cell in row), default=())

        # Broadcast all tensors to the same shape.
        grid = [[cell.expand(*max_shape) for cell in row] for row in grid]

        # Stack grid into a matrix tensor.
        return torch.cat([torch.cat(row, dim=-1) for row in grid], dim=-2)  # (*BS, M * D, N * D)

    @abstractmethod
    def Ab(self, traj: Trajectory, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            traj (Trajectory): Demonstration trajectory.
            t (Tensor, dtype=default, shape=(*BS)): Time values in [0,1].

        Returns:
            A (Tensor, dtype=default, shape=(*BS, X, X)): Transition matrix.
            b (Tensor, dtype=default, shape=(*BS, X)): Bias vector.
        """
        return NotImplementedError

    @abstractmethod
    def μΣ0(self, traj: Trajectory) -> Tuple[Tensor, Tensor]:
        """
        Compute the mean and covariance matrix of the conditional flow at time t=0.

        Returns:
            Tensor, dtype=default, shape=(X,): Mean at time t=0.
            Tensor, dtype=default, shape=(X, X): Covariance matrix at time t=0.
        """
        return NotImplementedError

    def μΣt(self, traj: Trajectory, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute the mean and covariance matrix of the conditional flows at time t.

        Args:
            traj (Trajectory): Demonstration trajectory.
            t (Tensor, dtype=default, shape=(*BS)): Time values in [0,1].

        Returns:
            Tensor, dtype=default, shape=(*BS, X): Mean at time t.
            Tensor, dtype=default, shape=(*BS, X, X): Covariance matrix at time t.
        """
        μ0, Σ0 = self.μΣ0(traj)  # (X,) and (X, X)
        A, b = self.Ab(traj, t)  # (*BS, X, X) and (*BS, X)
        AT = A.transpose(-1, -2)  # (*BS, X, X)
        μt = A @ μ0 + b  # (*BS, X)
        Σt = A @ Σ0 @ AT  # (*BS, X, X)
        return μt, Σt  # (*BS, X) and (*BS, X, X)

    def log_pdf_conditional(self, traj: Trajectory, x: Tensor, t: Tensor) -> Tensor:
        """
        Compute log-probability of the conditional flow at state x and time t for
        the given trajectory.

        Args:
            traj (Trajectory): Demonstration trajectory.
            x (Tensor, dtype=default, shape=(*BS, X)): State values.
            t (Tensor, dtype=default, shape=(*BS)): Time values in [0,1].

        Returns:
            (Tensor, dtype=default, shape=(*BS)): Probability of the conditional
                flow at state x and time t.
        """
        assert x.shape[-1] == self.X
        μt, Σt = self.μΣt(traj, t)  # (*BS, X) and (*BS, X, X)
        dist = MultivariateNormal(loc=μt, covariance_matrix=Σt)  # BS=(*BS) ES=(X,)
        return dist.log_prob(x)  # (*BS)

    def log_pdf_marginal(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Compute log-probability of the marginal flow at state x and time t.

        Args:
            x (Tensor, dtype=default, shape=(*BS, X)): State values.
            t (Tensor, dtype=default, shape=(*BS)): Time values in [0,1].

        Returns:
            (Tensor, dtype=default, shape=(*BS)): Log-probability of the marginal
                flow at state x and time t.
        """
        log_pdf = torch.tensor(-torch.inf, dtype=torch.get_default_dtype())
        for π, traj in zip(self.π, self.trajectories):
            log_pdf_i = π.log() + self.log_pdf_conditional(traj, x, t)  # (*BS)
            log_pdf = torch.logaddexp(log_pdf, log_pdf_i)  # (*BS)
        return log_pdf  # (*BS)

    @abstractmethod
    def v_conditional(self, traj: Trajectory, x: Tensor, t: Tensor) -> Tensor:
        """
        Compute the conditional velocity field for a given trajectory.

        Args:
            traj (Trajectory): Demonstration trajectory.
            x (Tensor, dtype=default, shape=(*BS, X)): State values.
            t (Tensor, dtype=default, shape=(*BS)): Time values in [0,1].

        Returns:
            (Tensor, dtype=default, shape=(*BS, X)): Velocity of the conditional flow.
        """
        raise NotImplementedError

    def v_marginal(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x (Tensor, dtype=default, shape=(*BS, X)): State values.
            t (Tensor, dtype=default, shape=(*BS)): Time values in [0,1].

        Returns:
            (Tensor, dtype=default, shape=(X,)): Marginal velocities.
        """
        log_likelihoods = torch.stack([self.log_pdf_conditional(traj, x, t) for traj in self.trajectories], dim=-1)  # (*BS, K)
        velocities = torch.stack([self.v_conditional(traj, x, t) for traj in self.trajectories], dim=-2)  # (*BS, K, X)

        log_posterior = self.π.log() + log_likelihoods  # (*BS, K)
        log_partition_fn = log_posterior.logsumexp(dim=-1, keepdim=True)  # (*BS, 1)
        log_posterior = log_posterior - log_partition_fn  # (*BS, K)
        posterior = log_posterior.exp().unsqueeze(-1)  # (*BS, K, 1)
        assert posterior.isfinite().all()

        us = (posterior * velocities).sum(dim=-2)  # (*BS, X)
        assert us.isfinite().all()

        return us

    def ode_integrate(self, x: Tensor, num_steps: int = 1000) -> List[Trajectory]:
        """
        Args:
            x (Tensor, dtype=default, shape=(L, X)): Initial state.
            num_steps (int): Number of steps to integrate.
            
        Returns:
            List[Trajectory]: Trajectories starting from x.
        """
        L = x.shape[0]
        breaks = np.linspace(0.0, 1.0, num_steps + 1)  # (N+1,)
        Δt = 1.0 / num_steps
        multi_traj = [x]
        for t in breaks[:-1]:
            v = self.v_marginal(x, torch.ones([L]) * t)  # (L, X)
            x = x + Δt * v  # (L, X)
            multi_traj.append(x)
        multi_traj = torch.stack(multi_traj, dim=-2)  # (L, N+1, X)
        multi_traj = multi_traj.numpy()  # (L, N+1, X)
        return [
            PiecewisePolynomial.FirstOrderHold(breaks, traj.T)
            for traj in multi_traj
        ]
