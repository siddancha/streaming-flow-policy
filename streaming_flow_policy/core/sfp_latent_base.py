from abc import abstractmethod
from functools import cached_property
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal
from typing import List, Tuple

from pydrake.all import Trajectory

from .sfp_base import StreamingFlowPolicyBase


class StreamingFlowPolicyLatentBase (StreamingFlowPolicyBase):
    def __init__(
        self,
        dim: int,
        trajectories: List[Trajectory],
        prior: List[float],
        σ0: float,
    ):
        """
        Flow policy is an extended configuration space (q(t), z(t)) where q is
        the original trajectory and z is a noise variable that starts from
        N(0, 1).

        Args:
            trajectories (List[Trajectory]): List of trajectories.
            prior (np.ndarray, dtype=float, shape=(K,)): Prior
                probabilities for each trajectory.
            σ0 (float): Standard deviation of the Gaussian tube at time t=0.
        """
        super().__init__(
            dim = 2 * dim,  # twice the dimension because of q and z
            trajectories = trajectories,
            prior = prior,
        )
        self.σ0 = σ0

    @cached_property
    def D(self) -> int:
        return self.X // 2

    @cached_property
    def slice_q(self) -> slice:
        return slice(0, self.D)

    @cached_property
    def slice_z(self) -> slice:
        return slice(self.D, 2 * self.D)

    def μΣ0(self, traj: Trajectory) -> Tuple[Tensor, Tensor]:
        """
        Compute the mean and covariance matrix of the conditional flow at time t=0.

        Returns:
            Tensor, dtype=default, shape=(2D,): Mean at time t=0.
            Tensor, dtype=default, shape=(2D, 2D): Covariance matrix at time t=0.
        """
        I = torch.eye(self.D)  # (D, D) identity matrix
        O = torch.zeros(self.D, self.D)  # (D, D) zero matrix
        zero_vector = torch.zeros(self.D)  # (D,) zero vector

        ξ0 = self.ξt(traj, torch.tensor(0.))  # (D,)
        σ0 = self.σ0 * I  # (D, D)

        μ0 = torch.cat([ξ0, zero_vector], dim=-1)  # (2D,)
        Σ0 = self.block_matrix([
            [σ0.square(), O],
            [O,           I],
        ])  # (2D, 2D)
        return μ0, Σ0

    def μΣt_zCa(self, traj: Trajectory, t: Tensor, q: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute the mean and covariance matrix of the conditional flow of z
        given q at time t.

        Args:
            traj (Trajectory): Demonstration trajectory.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].
            a (Tensor, dtype=default, shape=(*BS, D)): Action.

        Returns:
            Tensor, dtype=default, shape=(*BS, D): Mean at time t.
            Tensor, dtype=default, shape=(*BS, D, D): Covariance matrix at time t.
        """
        μ_qz, Σ_qz = self.μΣt(traj, t)  # (*BS, 2D), (*BS, 2D, 2D)
        μq, μz = μ_qz[..., self.slice_q], μ_qz[..., self.slice_z]  # (*BS, D) and (*BS, D)
        
        Σqq = Σ_qz[..., self.slice_q, self.slice_q]  # (*BS, D, D)
        Σqz = Σ_qz[..., self.slice_q, self.slice_z]  # (*BS, D, D)
        Σzq = Σ_qz[..., self.slice_z, self.slice_q]  # (*BS, D, D)
        Σzz = Σ_qz[..., self.slice_z, self.slice_z]  # (*BS, D, D)

        # Repeated computation
        Σqq_inv = torch.inverse(Σqq)  # (*BS, D, D)

        # From https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        μ_zCq = μz + (Σzq @ Σqq_inv @ (q - μq).unsqueeze(-1)).squeeze(-1)  # (*BS, D)
        Σ_zCq = Σzz - Σzq @ Σqq_inv @ Σqz  # (*BS, D, D)

        return μ_zCq, Σ_zCq

    def log_pdf_conditional_q(self, traj: Trajectory, q: Tensor, t: Tensor) -> Tensor:
        """
        Compute log-probability of the conditional flow at configuration q and time
        t, for the given trajectory.
        
        Args:
            traj (Trajectory): Demonstration trajectory.
            q (Tensor, dtype=default, shape=(*BS, D)): Configuration.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].
            
        Returns:
            (Tensor, dtype=default, shape=(*BS)): Log-probability of the
                conditional flow at configuration q and time t.
        """
        μ_qz, Σ_qz = self.μΣt(traj, t)  # (*BS, 2D), (*BS, 2D, 2D)
        μ_q = μ_qz[..., self.slice_q]  # (*BS, D)
        Σ_q = Σ_qz[..., self.slice_q, self.slice_q]  # (*BS, D, D)
        dist = MultivariateNormal(loc=μ_q, covariance_matrix=Σ_q)  # BS=(*BS) ES=(D,)
        return dist.log_prob(q)  # (*BS)

    def log_pdf_marginal_q(self, q: Tensor, t: Tensor) -> Tensor:
        """
        Compute log-probability of the marginal flow at configuration q and time t.
        
        Args:
            q (Tensor, dtype=default, shape=(*BS, D)): Configuration.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].
            
        Returns:
            (Tensor, dtype=default, shape=(*BS)): Log-probability of the marginal
                flow at configuration q and time t.
        """
        log_pdf = torch.tensor(-torch.inf, dtype=torch.get_default_dtype())
        for π, traj in zip(self.π, self.trajectories):
            log_pdf_i = π.log() + self.log_pdf_conditional_q(traj, q, t)  # (*BS)
            log_pdf = torch.logaddexp(log_pdf, log_pdf_i)  # (*BS)
        return log_pdf  # (*BS)

    def log_pdf_conditional_z(self, traj: Trajectory, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute log-probability of the conditional flow at latent z and time t.
        
        Args:
            traj (Trajectory): Demonstration trajectory.
            z (Tensor, dtype=default, shape=(*BS, D)): Latent variable value.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].
            
        Returns:
            (Tensor, dtype=default, shape=(*BS)): Log-probability of the
                conditional flow at latent z and time t.
        """
        μ_qz, Σ_qz = self.μΣt(traj, t)  # (*BS, 2D), (*BS, 2D, 2D)
        μ_z = μ_qz[..., self.slice_z]  # (*BS, D)
        Σ_z = Σ_qz[..., self.slice_z, self.slice_z]  # (*BS, D, D)
        dist = MultivariateNormal(loc=μ_z, covariance_matrix=Σ_z)  # BS=(*BS) ES=(D,)
        return dist.log_prob(z)  # (*BS)

    def log_pdf_marginal_z(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute log-probability of the marginal flow at latent z and time t.
        
        Args:
            z (Tensor, dtype=default, shape=(*BS, D)): Latent variable value.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].
            
        Returns:
            (Tensor, dtype=default, shape=(*BS)): Log-probability of the marginal
                flow at latent z and time t.
        """
        log_pdf = torch.tensor(-torch.inf, dtype=torch.get_default_dtype())
        for π, traj in zip(self.π, self.trajectories):
            log_pdf_i = π.log() + self.log_pdf_conditional_z(traj, z, t)  # (*BS)
            log_pdf = torch.logaddexp(log_pdf, log_pdf_i)  # (*BS)
        return log_pdf  # (*BS)

    def pdf_posterior_ξCq(self, q: Tensor, t: Tensor) -> Tensor:
        """
        Compute probability p(ξ | q, t) of the posterior distribution of ξ
        given q and t.

        Args:
            q (Tensor, dtype=default, shape=(*BS, D)): Configuration.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=default, shape=(*BS, K)): p(ξ | q, t).
        """
        list_log_pdf: List[Tensor] = []
        for π, traj in zip(self.π, self.trajectories):
            log_pdf_i = π.log() + self.log_pdf_conditional_q(traj, q, t)  # (*BS)
            list_log_pdf.append(log_pdf_i)
        log_pdf = torch.stack(list_log_pdf, dim=-1)  # (*BS, K)
        return torch.softmax(log_pdf, dim=-1)  # (*BS, K)

    def pdf_posterior_ξCz(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute probability p(ξ | z, t) of the posterior distribution of ξ
        given z and t.

        Args:
            z (Tensor, dtype=default, shape=(*BS, D)): Latent variable value.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=default, shape=(*BS, K)): p(ξ | z, t).
        """
        list_log_pdf = []
        for π, traj in zip(self.π, self.trajectories):
            log_pdf_i = π.log() + self.log_pdf_conditional_z(traj, z, t)  # (*BS)
            list_log_pdf.append(log_pdf_i)
        log_pdf = torch.stack(list_log_pdf, dim=-1)  # (*BS, K)
        return torch.softmax(log_pdf, dim=-1)  # (*BS, K)

    @abstractmethod
    def 𝔼vq_conditional(self, traj: Trajectory, q: Tensor, t: Tensor) -> Tensor:
        """
        Compute the expected velocity field of q over z given q, t and a trajectory.

        Args:
            traj (Trajectory): Demonstration trajectory.
            q (Tensor, dtype=default, shape=(*BS, D)): Configuration.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=default, shape=(*BS, D)):
                expected value of vq over z given q, t and a trajectory.
        """
        raise NotImplementedError

    @abstractmethod
    def 𝔼vz_conditional(self, traj: Trajectory, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute the expected velocity field of z over q given z, t and a trajectory.

        Args:
            traj (Trajectory): Demonstration trajectory.
            z (Tensor, dtype=default, shape=(*BS, D)): Latent variable value.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=default, shape=(*BS, D)):
                expected value of vz given z, t and a trajectory.
        """
        raise NotImplementedError

    def 𝔼vq_marginal(self, q: Tensor, t: Tensor) -> Tensor:
        """
        Compute the expected velocity field of q over z given q, t.

        Args:
            q (Tensor, dtype=default, shape=(*BS, D)): Configuration.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=default, shape=(*BS, D)):
                expected value of vq given q, t.
        """
        posterior_ξ = self.pdf_posterior_ξCq(q, t)  # (*BS, K)
        posterior_ξ = posterior_ξ.unsqueeze(-2)  # (*BS, 1, K)
        𝔼vq = torch.stack([
            self.𝔼vq_conditional(traj, q, t)  # (*BS, D)
            for traj in self.trajectories
        ], dim=-1)  # (*BS, D, K)
        return (posterior_ξ * 𝔼vq).sum(dim=-1)  # (*BS, D)

    def 𝔼vz_marginal(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute the expected velocity field of z over q given z, t.

        Args:
            z (Tensor, dtype=default, shape=(*BS, D)): Latent variable value.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=default, shape=(*BS, D)):
                expected value of vz over q given z, t.
        """
        posterior_ξ = self.pdf_posterior_ξCz(z, t)  # (*BS, K)
        posterior_ξ = posterior_ξ.unsqueeze(-2)  # (*BS, 1, K)
        𝔼vz = torch.stack([
            self.𝔼vz_conditional(traj, z, t)  # (*BS, D)
            for traj in self.trajectories
        ], dim=-1)  # (*BS, D, K)
        return (posterior_ξ * 𝔼vz).sum(dim=-1)  # (*BS, D)
