import numpy as np
from scipy.stats import multivariate_normal
import torch; torch.set_default_dtype(torch.double)
from torch import Tensor
from torch.distributions import MultivariateNormal
from typing import List, Tuple

from pydrake.all import Trajectory

from streaming_flow_policy.toy.sfp_base import StreamingFlowPolicyBase


class StreamingFlowPolicyStochasticStabilizing (StreamingFlowPolicyBase):
    def __init__(
        self,
        trajectories: List[Trajectory],
        prior: List[float],
        σ0: float,
        σ1: float,
        k: float = 0.0,
    ):
        """
        Flow policy is an extended configuration space (q(t), z(t)) where q is
        the original trajectory and z is a noise variable that starts from
        N(0, 1).

        Let ξ(t) be the demonstration trajectory.
        Define constant σᵣ = √(σ₁²exp(2k) - σ₀²).
        Note that σ₁²exp(2k) = σ₀² + σᵣ².

        Conditional flow:
        • At time t=0, we sample:
            • q₀ ~ N(ξ(0), σ₀²)
            • z₀ ~ N(0, 1)

        • Flow trajectory at time t:
            • q(t) = ξ(t) + (q₀ - ξ(0) + σᵣtz₀) exp(-kt)
            • z(t) = (1 - (1-σ₁)t)z₀ + tξ(t)
              • z starts from a pure noise sample z₀ that drifts towards the
              trajectory. Therefore, z(t) is uncorrelated with q at t=0, but
              eventually becomes very informative of the trajectory.

        Args:
            trajectories (List[Trajectory]): List of trajectories.
            prior (np.ndarray, dtype=float, shape=(K,)): Prior
                probabilities for each trajectory.
            σ0 (float): Standard deviation of the Gaussian tube at time t=0.
            σ1 (float): Standard deviation of the Gaussian tube at time t=1.
        """
        super().__init__(dim=2, trajectories=trajectories, prior=prior)

        assert 0 <= σ0 <= σ1, "σ0 must be less than or equal to σ1"
        self.σ0 = σ0
        self.σ1 = σ1
        self.k = k

        # Residual standard deviation: √(σ₁²exp(2k) - σ₀²)
        self.σr = np.sqrt(np.square(σ1) * np.exp(2 * k) - np.square(σ0))

    def Ab(self, traj: Trajectory, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            traj (Trajectory): Demonstration trajectory.
            t (Tensor, dtype=double, shape=(*BS)): Time value in [0,1].

        Returns:
            A (Tensor, dtype=double, shape=(*BS, 2, 2)): Transition matrix.
            b (Tensor, dtype=double, shape=(*BS, 2)): Bias vector.
        """
        σ1 = self.σ1  # (,)
        σr = self.σr  # (,)
        k = self.k  # (,)

        ξ0: float = traj.value(0).item()
        ξt = self.ξt(traj, t)[..., 0]  # (*BS)
        αt = torch.exp(-k * t)  # (*BS)

        b = torch.stack([ξt - ξ0 * αt, t * ξt], dim=-1)  # (*BS, 2)
        A = self.matrix_stack([
            [αt,     σr * t * αt],
            [0, 1 - (1 - σ1) * t],
        ])  # (*BS, 2, 2)
        return A, b

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
        μ_zCq = μz + Σzq @ Σqq_inv @ (q - μq)  # (*BS, 1)
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

    def v_conditional(self, traj: Trajectory, x: Tensor, t: Tensor) -> Tensor:
        """
        Compute the conditional velocity field for a given trajectory.

        • Flow trajectory at time t:
            • q(t) = ξ(t) + (q₀ - ξ(0) + σᵣtz₀) exp(-kt)
            • z(t) = (1 - (1-σ₁)t)z₀ + tξ(t)

        • Conditional velocity field:
            • First, given q(t) and z(t), we want to compute q₀ and z₀.
                • z₀ = (z(t) - tξ(t)) / (1 - (1-σ₁)t)
                • q₀ = ξ(0) + (q(t) - ξ(t)) exp(kt) - σᵣtz₀
            • Then, we compute the velocity for the conditional flow.
                • vq(q, z, t) = ξ̇(t) -k(q₀ - ξ(0) + σᵣtz₀)exp(-kt) + σᵣz₀exp(-kt)
                • vz(q, z, t) = ξ(t) + tξ̇(t) - (1-σ₁)z₀
            • Plugging (z₀, q₀) into the velocity gives us the velocity field:
                • vq(q, z, t) = ξ̇(t) - k(q - ξ(t)) + σᵣexp(-kt) / (1 - (1-σ₁)t) * (z - tξ(t))
                • vz(q, z, t) = ξ(t) + tξ̇(t) - (1-σ₁) / (1 - (1-σ₁)t) * (z - tξ(t))

        Args:
            traj (Trajectory): Demonstration trajectory.
            x (Tensor, dtype=double, shape=(*BS, 2)): State values.
            t (Tensor, dtype=double, shape=(*BS)): Time value in [0,1].
            
        Returns:
            (Tensor, dtype=double, shape=(*BS, 2)): Velocity of conditional flow.
        """
        σ1 = self.σ1
        σr = self.σr
        k = self.k

        qt = x[..., 0:1]  # (*BS, 1)
        zt = x[..., 1:2]  # (*BS, 1)
        ξt = self.ξt(traj, t)  # (*BS, 1)
        ξ̇t = self.ξ̇t(traj, t)  # (*BS, 1)
        t = t.unsqueeze(-1)  # (*BS, 1)
        αt = torch.exp(-k * t)  # (*BS, 1)

        # Invert zt to get z0
        z0 = (zt - t * ξt) / (1 - (1 - σ1) * t)  # (*BS, 1)

        # Compute velocity field
        vq = ξ̇t - k * (qt - ξt) + σr * z0 * αt  # (*BS, 1)
        vz = ξt + t * ξ̇t - (1 - σ1) * z0  # (*BS, 1)

        return torch.cat([vq, vz], dim=-1)  # (*BS, 2)

    def 𝔼vq_conditional(self, traj: Trajectory, q: Tensor, t: Tensor) -> Tensor:
        """
        Compute the expected velocity field of q over z given q, t and a trajectory.

        The velocity field is given by:
            • vq(q, z, t) = ξ̇(t) - k(q - ξ(t)) + σᵣexp(-kt) / (1 - (1-σ₁)t) * (z - tξ(t))
        
        Therefore, the expected velocity under N(μ_z|q, Σ_z|q) is given by:
            • 𝔼[vq(q, z, t)] = ξ̇(t) - k(q - ξ(t)) + σᵣexp(-kt) / (1 - (1-σ₁)t) * (μ_z|q - tξ(t))

        Args:
            traj (Trajectory): Demonstration trajectory.
            q (Tensor, dtype=double, shape=(*BS, 1)): Configuration.
            t (Tensor, dtype=double, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=double, shape=(*BS, 1)):
                expected value of vq over z given q, t and a trajectory.
        """
        σ1 = self.σ1
        σr = self.σr
        k = self.k

        μ_zCq, Σ_zCq = self.μΣt_zCq(traj, t, q)  # (*BS, 1), (*BS, 1, 1)

        ξt = self.ξt(traj, t)  # (*BS, 1)
        ξ̇t = self.ξ̇t(traj, t)  # (*BS, 1)
        t = t.unsqueeze(-1)  # (*BS, 1)
        αt = torch.exp(-k * t)  # (*BS, 1)

        # Expected z0 given q
        μ_z0Cq = (μ_zCq - t * ξt) / (1 - (1 - σ1) * t)  # (*BS, 1)

        # Compute expected velocity field
        𝔼vq = ξ̇t - k * (q - ξt) + σr * μ_z0Cq * αt  # (*BS, 1)
        return 𝔼vq  # (*BS, 1)

    def 𝔼vz_conditional(self, traj: Trajectory, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute the expected velocity field of z over q given z, t and a trajectory.

        The velocity field is given by:
            • vz(q, z, t) = ξ(t) + tξ̇(t) - (1-σ₁) / (1 - (1-σ₁)t) * (z - tξ(t))
        
        Therefore, the expected velocity under N(μ_z|q, Σ_z|q) is given by:
            • 𝔼[vz(q, z, t)] = ξ(t) + tξ̇(t) - (1-σ₁) / (1 - (1-σ₁)t) * (z - tξ(t))

        NOTE: the velocity field of z does not depend on q. So we need not
        compute the distribution of q given z.

        Args:
            traj (Trajectory): Demonstration trajectory.
            z (Tensor, dtype=double, shape=(*BS, 1)): Latent variable value.
            t (Tensor, dtype=double, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=double, shape=(*BS, 1)):
                expected value of vz given z, t and a trajectory.
        """
        σ1 = self.σ1

        ξt = self.ξt(traj, t)  # (*BS, 1)
        ξ̇t = self.ξ̇t(traj, t)  # (*BS, 1)
        t = t.unsqueeze(-1)  # (*BS, 1)

        # Invert zt to get z0
        z0 = (z - t * ξt) / (1 - (1 - σ1) * t)  # (*BS, 1)

        # Compute expected velocity field
        𝔼vz = ξt + t * ξ̇t - (1 - σ1) * z0  # (*BS, 1)
        return 𝔼vz

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
