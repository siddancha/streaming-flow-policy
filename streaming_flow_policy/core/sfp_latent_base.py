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
        Flow policy is an extended state space (a(t), z(t)) where a is
        the original action trajectory and z is a noise variable that starts from
        N(0, 1).

        Args:
            trajectories (List[Trajectory]): List of trajectories.
            prior (np.ndarray, dtype=float, shape=(K,)): Prior
                probabilities for each trajectory.
            σ0 (float): Standard deviation of the Gaussian tube at time t=0.
        """
        super().__init__(
            dim = 2 * dim,  # twice the dimension because of a and z
            trajectories = trajectories,
            prior = prior,
        )
        self.σ0 = σ0

    @cached_property
    def D(self) -> int:
        return self.X // 2

    @cached_property
    def slice_a(self) -> slice:
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

    def μΣt_zCa(self, traj: Trajectory, t: Tensor, a: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute the mean and covariance matrix of the conditional flow of z
        given a at time t.

        Args:
            traj (Trajectory): Demonstration trajectory.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].
            a (Tensor, dtype=default, shape=(*BS, D)): Action.

        Returns:
            Tensor, dtype=default, shape=(*BS, D): Mean at time t.
            Tensor, dtype=default, shape=(*BS, D, D): Covariance matrix at time t.
        """
        μ_az, Σ_az = self.μΣt(traj, t)  # (*BS, 2D), (*BS, 2D, 2D)
        μa, μz = μ_az[..., self.slice_a], μ_az[..., self.slice_z]  # (*BS, D) and (*BS, D)
        
        Σaa = Σ_az[..., self.slice_a, self.slice_a]  # (*BS, D, D)
        Σaz = Σ_az[..., self.slice_a, self.slice_z]  # (*BS, D, D)
        Σza = Σ_az[..., self.slice_z, self.slice_a]  # (*BS, D, D)
        Σzz = Σ_az[..., self.slice_z, self.slice_z]  # (*BS, D, D)

        # Repeated computation
        Σaa_inv = torch.inverse(Σaa)  # (*BS, D, D)

        # From https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        μ_zCa = μz + (Σza @ Σaa_inv @ (a - μa).unsqueeze(-1)).squeeze(-1)  # (*BS, D)
        Σ_zCa = Σzz - Σza @ Σaa_inv @ Σaz  # (*BS, D, D)

        return μ_zCa, Σ_zCa

    def log_pdf_conditional_a(self, traj: Trajectory, a: Tensor, t: Tensor) -> Tensor:
        """
        Compute log-probability of the conditional flow at action a and time t,
        for the given trajectory.
        
        Args:
            traj (Trajectory): Demonstration trajectory.
            a (Tensor, dtype=default, shape=(*BS, D)): Action.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].
            
        Returns:
            (Tensor, dtype=default, shape=(*BS)): Log-probability of the
                conditional flow at action a and time t.
        """
        μ_az, Σ_az = self.μΣt(traj, t)  # (*BS, 2D), (*BS, 2D, 2D)
        μ_a = μ_az[..., self.slice_a]  # (*BS, D)
        Σ_a = Σ_az[..., self.slice_a, self.slice_a]  # (*BS, D, D)
        dist = MultivariateNormal(loc=μ_a, covariance_matrix=Σ_a)  # BS=(*BS) ES=(D,)
        return dist.log_prob(a)  # (*BS)

    def log_pdf_marginal_a(self, a: Tensor, t: Tensor) -> Tensor:
        """
        Compute log-probability of the marginal flow at action a and time t.
        
        Args:
            a (Tensor, dtype=default, shape=(*BS, D)): Action.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].
            
        Returns:
            (Tensor, dtype=default, shape=(*BS)): Log-probability of the marginal
                flow at action a and time t.
        """
        log_pdf = torch.tensor(-torch.inf, dtype=torch.get_default_dtype())
        for π, traj in zip(self.π, self.trajectories):
            log_pdf_i = π.log() + self.log_pdf_conditional_a(traj, a, t)  # (*BS)
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
        μ_az, Σ_az = self.μΣt(traj, t)  # (*BS, 2D), (*BS, 2D, 2D)
        μ_z = μ_az[..., self.slice_z]  # (*BS, D)
        Σ_z = Σ_az[..., self.slice_z, self.slice_z]  # (*BS, D, D)
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

    def pdf_posterior_ξCa(self, a: Tensor, t: Tensor) -> Tensor:
        """
        Compute probability p(ξ | a, t) of the posterior distribution of ξ
        given a and t.

        Args:
            a (Tensor, dtype=default, shape=(*BS, D)): Action.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=default, shape=(*BS, K)): p(ξ | a, t).
        """
        list_log_pdf: List[Tensor] = []
        for π, traj in zip(self.π, self.trajectories):
            log_pdf_i = π.log() + self.log_pdf_conditional_a(traj, a, t)  # (*BS)
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
    def 𝔼va_conditional(self, traj: Trajectory, a: Tensor, t: Tensor) -> Tensor:
        """
        Compute the expected velocity field of a over z given a, t and a trajectory.

        Args:
            traj (Trajectory): Demonstration trajectory.
            a (Tensor, dtype=default, shape=(*BS, D)): Action.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=default, shape=(*BS, D)):
                expected value of va over z given a, t and a trajectory.
        """
        raise NotImplementedError

    @abstractmethod
    def 𝔼vz_conditional(self, traj: Trajectory, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute the expected velocity field of z over a given z, t and a trajectory.

        Args:
            traj (Trajectory): Demonstration trajectory.
            z (Tensor, dtype=default, shape=(*BS, D)): Latent variable value.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=default, shape=(*BS, D)):
                expected value of vz given z, t and a trajectory.
        """
        raise NotImplementedError

    def 𝔼va_marginal(self, a: Tensor, t: Tensor) -> Tensor:
        """
        Compute the expected velocity field of a over z given a, t.

        Args:
            a (Tensor, dtype=default, shape=(*BS, D)): Action.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=default, shape=(*BS, D)):
                expected value of va given a, t.
        """
        posterior_ξ = self.pdf_posterior_ξCa(a, t)  # (*BS, K)
        posterior_ξ = posterior_ξ.unsqueeze(-2)  # (*BS, 1, K)
        𝔼va = torch.stack([
            self.𝔼va_conditional(traj, a, t)  # (*BS, D)
            for traj in self.trajectories
        ], dim=-1)  # (*BS, D, K)
        return (posterior_ξ * 𝔼va).sum(dim=-1)  # (*BS, D)

    def 𝔼vz_marginal(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute the expected velocity field of z over a given z, t.

        Args:
            z (Tensor, dtype=default, shape=(*BS, D)): Latent variable value.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=default, shape=(*BS, D)):
                expected value of vz over a given z, t.
        """
        posterior_ξ = self.pdf_posterior_ξCz(z, t)  # (*BS, K)
        posterior_ξ = posterior_ξ.unsqueeze(-2)  # (*BS, 1, K)
        𝔼vz = torch.stack([
            self.𝔼vz_conditional(traj, z, t)  # (*BS, D)
            for traj in self.trajectories
        ], dim=-1)  # (*BS, D, K)
        return (posterior_ξ * 𝔼vz).sum(dim=-1)  # (*BS, D)
