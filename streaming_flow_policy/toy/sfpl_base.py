from abc import abstractmethod
import numpy as np
import torch; torch.set_default_dtype(torch.double)
from torch import Tensor
from torch.distributions import MultivariateNormal
from typing import List, Tuple

from pydrake.all import Trajectory

from streaming_flow_policy.toy.sfp_base import StreamingFlowPolicyBase


class StreamingFlowPolicyLatentBase (StreamingFlowPolicyBase):
    def __init__(
        self,
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
        super().__init__(dim=2, trajectories=trajectories, prior=prior)
        self.σ0 = σ0

    def μΣ0(self, traj: Trajectory) -> Tuple[Tensor, Tensor]:
        """
        Compute the mean and covariance matrix of the conditional flow at time t=0.

        Returns:
            Tensor, dtype=double, shape=(*BS, 2): Mean at time t=0.
            Tensor, dtype=double, shape=(*BS, 2, 2): Covariance matrix at time t=0.
        """
        ξ0 = traj.value(0).item()
        σ0 = self.σ0
        μ0 = torch.tensor([ξ0, 0], dtype=torch.double)  # (2,)
        Σ0 = torch.tensor([[np.square(σ0), 0], [0, 1]], dtype=torch.double)  # (2, 2)
        return μ0, Σ0

    def μΣt_zCq(self, traj: Trajectory, t: Tensor, q: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute the mean and covariance matrix of the conditional flow of z
        given q at time t.

        Args:
            traj (Trajectory): Demonstration trajectory.
            t (Tensor, dtype=double, shape=(*BS)): Time value in [0,1].
            q (Tensor, dtype=double, shape=(*BS, 1)): Configuration.

        Returns:
            Tensor, dtype=double, shape=(*BS, 1): Mean at time t.
            Tensor, dtype=double, shape=(*BS, 1, 1): Covariance matrix at time t.
        """
        μ_qz, Σ_qz = self.μΣt(traj, t)  # (*BS, 2), (*BS, 2, 2)
        μq, μz = μ_qz[..., 0:1], μ_qz[..., 1:2]  # (*BS, 1)
        
        Σqq = Σ_qz[..., 0:1, 0:1]  # (*BS, 1, 1)
        Σqz = Σ_qz[..., 0:1, 1:2]  # (*BS, 1, 1)
        Σzq = Σ_qz[..., 1:2, 0:1]  # (*BS, 1, 1)
        Σzz = Σ_qz[..., 1:2, 1:2]  # (*BS, 1, 1)

        # Repeated computation
        Σqq_inv = torch.inverse(Σqq)  # (*BS, 1, 1)

        # From https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        μ_zCq = μz + (Σzq @ Σqq_inv @ (q - μq).unsqueeze(-1)).squeeze(-1)  # (*BS, 1)
        Σ_zCq = Σzz - Σzq @ Σqq_inv @ Σqz  # (*BS, 1, 1)

        return μ_zCq, Σ_zCq

    def log_pdf_conditional_q(self, traj: Trajectory, q: Tensor, t: Tensor) -> Tensor:
        """
        Compute log-probability of the conditional flow at configuration q and time
        t, for the given trajectory.
        
        Args:
            traj (Trajectory): Demonstration trajectory.
            q (Tensor, dtype=double, shape=(*BS, 1)): Configuration.
            t (Tensor, dtype=double, shape=(*BS)): Time value in [0,1].
            
        Returns:
            (Tensor, dtype=double, shape=(*BS)): Log-probability of the
                conditional flow at configuration q and time t.
        """
        μ_qz, Σ_qz = self.μΣt(traj, t)  # (*BS, 2), (*BS, 2, 2)
        μ_q = μ_qz[..., 0:1]  # (*BS, 1)
        Σ_q = Σ_qz[..., 0:1, 0:1]  # (*BS, 1, 1)
        dist = MultivariateNormal(loc=μ_q, covariance_matrix=Σ_q)  # BS=(*BS) ES=(1,)
        return dist.log_prob(q)  # (*BS)

    def log_pdf_marginal_q(self, q: Tensor, t: Tensor) -> Tensor:
        """
        Compute log-probability of the marginal flow at configuration q and time t.
        
        Args:
            q (Tensor, dtype=double, shape=(*BS, 1)): Configuration.
            t (Tensor, dtype=double, shape=(*BS)): Time value in [0,1].
            
        Returns:
            (Tensor, dtype=double, shape=(*BS)): Log-probability of the marginal
                flow at configuration q and time t.
        """
        log_pdf = torch.tensor(-torch.inf, dtype=torch.double)
        for π, traj in zip(self.π, self.trajectories):
            log_pdf_i = π.log() + self.log_pdf_conditional_q(traj, q, t)  # (*BS)
            log_pdf = torch.logaddexp(log_pdf, log_pdf_i)  # (*BS)
        return log_pdf  # (*BS)

    def log_pdf_conditional_z(self, traj: Trajectory, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute log-probability of the conditional flow at latent z and time t.
        
        Args:
            traj (Trajectory): Demonstration trajectory.
            z (Tensor, dtype=double, shape=(*BS, 1)): Latent variable value.
            t (Tensor, dtype=double, shape=(*BS)): Time value in [0,1].
            
        Returns:
            (Tensor, dtype=double, shape=(*BS)): Log-probability of the
                conditional flow at latent z and time t.
        """
        μ_qz, Σ_qz = self.μΣt(traj, t)  # (*BS, 2), (*BS, 2, 2)
        μ_z = μ_qz[..., 1:2]  # (*BS, 1)
        Σ_z = Σ_qz[..., 1:2, 1:2]  # (*BS, 1, 1)
        dist = MultivariateNormal(loc=μ_z, covariance_matrix=Σ_z)  # BS=(*BS) ES=(1,)
        return dist.log_prob(z)  # (*BS)

    def log_pdf_marginal_z(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute log-probability of the marginal flow at latent z and time t.
        
        Args:
            z (Tensor, dtype=double, shape=(*BS, 1)): Latent variable value.
            t (Tensor, dtype=double, shape=(*BS)): Time value in [0,1].
            
        Returns:
            (Tensor, dtype=double, shape=(*BS)): Log-probability of the marginal
                flow at latent z and time t.
        """
        log_pdf = torch.tensor(-torch.inf, dtype=torch.double)
        for π, traj in zip(self.π, self.trajectories):
            log_pdf_i = π.log() + self.log_pdf_conditional_z(traj, z, t)  # (*BS)
            log_pdf = torch.logaddexp(log_pdf, log_pdf_i)  # (*BS)
        return log_pdf  # (*BS)

    def pdf_posterior_ξCq(self, q: Tensor, t: Tensor) -> Tensor:
        """
        Compute probability p(ξ | q, t) of the posterior distribution of ξ
        given q and t.

        Args:
            q (Tensor, dtype=double, shape=(*BS, 1)): Configuration.
            t (Tensor, dtype=double, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=double, shape=(*BS, K)): p(ξ | q, t).
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
            z (Tensor, dtype=double, shape=(*BS, 1)): Latent variable value.
            t (Tensor, dtype=double, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=double, shape=(*BS, K)): p(ξ | z, t).
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
            q (Tensor, dtype=double, shape=(*BS, 1)): Configuration.
            t (Tensor, dtype=double, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=double, shape=(*BS, 1)):
                expected value of vq over z given q, t and a trajectory.
        """
        raise NotImplementedError

    @abstractmethod
    def 𝔼vz_conditional(self, traj: Trajectory, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute the expected velocity field of z over q given z, t and a trajectory.

        Args:
            traj (Trajectory): Demonstration trajectory.
            z (Tensor, dtype=double, shape=(*BS, 1)): Latent variable value.
            t (Tensor, dtype=double, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=double, shape=(*BS, 1)):
                expected value of vz given z, t and a trajectory.
        """
        raise NotImplementedError

    def 𝔼vq_marginal(self, q: Tensor, t: Tensor) -> Tensor:
        """
        Compute the expected velocity field of q over z given q, t.

        Args:
            q (Tensor, dtype=double, shape=(*BS, 1)): Configuration.
            t (Tensor, dtype=double, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=double, shape=(*BS, 1)):
                expected value of vq given q, t.
        """
        posterior_ξ = self.pdf_posterior_ξCq(q, t)  # (*BS, K)
        posterior_ξ = posterior_ξ.unsqueeze(-2)  # (*BS, 1, K)
        𝔼vq = torch.stack([
            self.𝔼vq_conditional(traj, q, t)  # (*BS, 1)
            for traj in self.trajectories
        ], dim=-1)  # (*BS, 1, K)
        return (posterior_ξ * 𝔼vq).sum(dim=-1)  # (*BS, 1)

    def 𝔼vz_marginal(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute the expected velocity field of z over q given z, t.

        Args:
            z (Tensor, dtype=double, shape=(*BS, 1)): Latent variable value.
            t (Tensor, dtype=double, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=double, shape=(*BS, 1)):
                expected value of vz over q given z, t.
        """
        posterior_ξ = self.pdf_posterior_ξCz(z, t)  # (*BS, K)
        posterior_ξ = posterior_ξ.unsqueeze(-2)  # (*BS, 1, K)
        𝔼vz = torch.stack([
            self.𝔼vz_conditional(traj, z, t)  # (*BS, 1)
            for traj in self.trajectories
        ], dim=-1)  # (*BS, 1, K)
        return (posterior_ξ * 𝔼vz).sum(dim=-1)  # (*BS, 1)
