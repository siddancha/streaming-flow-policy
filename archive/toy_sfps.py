import numpy as np
from scipy.stats import multivariate_normal
import torch; torch.set_default_dtype(torch.double)
from torch import Tensor
from torch.distributions import MultivariateNormal
from typing import List, Tuple

from pydrake.all import Trajectory

from streaming_flow_policy.toy.sfp_base import StreamingFlowPolicyBase


class StreamingFlowPolicyStochastic (StreamingFlowPolicyBase):
    def __init__(
        self,
        trajectories: List[Trajectory],
        prior: List[float],
        σ0: float,
        σ1: float,
    ):
        """
        Flow policy is an extended state space (a(t), z(t)) where a is
        the original action trajectory and z is a noise variable that starts
        from N(0, 1).

        Let ξ(t) be the demonstration trajectory.
        Define constant σᵣ = √(σ₁² - σ₀²). Note that σ₁² = σ₀² + σᵣ².

        Conditional flow:
        • At time t=0, we sample:
            • a₀ ~ N(ξ(0), σ₀)
            • z₀ ~ N(0, 1)

        • Flow trajectory at time t:
            • a(t) = a₀ + (ξ(t) - ξ(0)) + (σᵣt) z₀
            • z(t) = (1 - (1-σ₁)t)z₀ + tξ(t)
              • z starts from a pure noise sample z₀ that drifts towards the
              trajectory. Therefore, z(t) is uncorrelated with a at t=0, but
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

        # Residual standard deviation: √(σ₁² - σ₀²)
        self.σr = np.sqrt(np.square(σ1) - np.square(σ0))

    def Ab(self, traj: Trajectory, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            A (Tensor, dtype=double, shape=(*BS, 2, 2)): Transition matrix.
            b (Tensor, dtype=double, shape=(*BS, 2)): Bias vector.
        """
        ξ0: float = traj.value(0).item()
        ξt = self.ξt(traj, t)[..., 0]  # (*BS)
        σ1 = self.σ1  # (,)
        σr = self.σr  # (,)

        b = torch.stack([ξt - ξ0, t * ξt], dim=-1)  # (*BS, 2)
        A = self.matrix_stack([
            [1,           σr * t],
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

    def log_pdf_conditional_a(self, traj: Trajectory, a: Tensor, t: Tensor) -> Tensor:
        """
        Compute log-probability of the conditional flow at action a and time
        t, for the given trajectory.
        
        Args:
            traj (Trajectory): Demonstration trajectory.
            a (Tensor, dtype=double, shape=(*BS, 1)): Action.
            t (Tensor, dtype=double, shape=(*BS)): Time value in [0,1].
            
        Returns:
            (Tensor, dtype=double, shape=(*BS)): Log-probability of the
                conditional flow at action a and time t.
        """
        μ_az, Σ_az = self.μΣt(traj, t)  # (*BS, 2), (*BS, 2, 2)
        μ_a = μ_az[..., 0:1]  # (*BS, 1)
        Σ_a = Σ_az[..., 0:1, 0:1]  # (*BS, 1, 1)
        dist = MultivariateNormal(loc=μ_a, covariance_matrix=Σ_a)  # BS=(*BS) ES=(1,)
        return dist.log_prob(a)  # (*BS)

    def log_pdf_marginal_a(self, a: Tensor, t: Tensor) -> Tensor:
        """
        Compute log-probability of the marginal flow at action a and time t.
        
        Args:
            a (Tensor, dtype=double, shape=(*BS, 1)): Action.
            t (Tensor, dtype=double, shape=(*BS)): Time value in [0,1].
            
        Returns:
            (Tensor, dtype=double, shape=(*BS)): Log-probability of the marginal
                flow at action a and time t.
        """
        log_pdf = torch.tensor(-torch.inf, dtype=torch.double)
        for π, traj in zip(self.π, self.trajectories):
            log_pdf_i = π.log() + self.log_pdf_conditional_a(traj, a, t)  # (*BS)
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
        μ_az, Σ_az = self.μΣt(traj, t)  # (*BS, 2), (*BS, 2, 2)
        μ_z = μ_az[..., 1:2]  # (*BS, 1)
        Σ_z = Σ_az[..., 1:2, 1:2]  # (*BS, 1, 1)
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

    def v_conditional(self, traj: Trajectory, x: Tensor, t: Tensor) -> Tensor:
        """
        Compute the conditional velocity field for a given trajectory.

        • Flow trajectory at time t:
            • a(t) = a₀ + (ξ(t) - ξ(0)) + (σᵣt) z₀
            • z(t) = (1 - (1-σ₁)t)z₀ + tξ(t)

        • Conditional velocity field:
            • First, given a(t) and z(t), we want to compute a₀ and z₀.
                • z₀ = (z(t) - tξ(t)) / (1 - (1-σ₁)t)
                • a₀ = a(t) - (ξ(t) - ξ(0)) - (σᵣt) z₀
            • Then, we compute the velocity field for the conditional flow.
                • va(a, z, t) = ξ̇(t) + σᵣz₀
                • vz(a, z, t) = ξ(t) + tξ̇(t) - (1-σ₁)z₀

        Args:
            traj (Trajectory): Demonstration trajectory.
            x (Tensor, dtype=double, shape=(*BS, 2)): State values.
            t (Tensor, dtype=double, shape=(*BS)): Time value in [0,1].
            
        Returns:
            (Tensor, dtype=double, shape=(*BS, 2)): Velocity of conditional flow.
        """
        zt = x[..., 1:2]  # (*BS, 1)
        ξt = self.ξt(traj, t)  # (*BS, 1)
        ξ̇t = self.ξ̇t(traj, t)  # (*BS, 1)
        t = t.unsqueeze(-1)  # (*BS, 1)
        σ1 = self.σ1
        σr = self.σr

        # Invert the flow and transform (at, zt) to (a0, z0)
        z0 = (zt - t * ξt) / (1 - (1 - σ1) * t)  # (*BS, 1)

        # Compute velocity of the trajectory starting from (a0, z0) at t
        va = ξ̇t + σr * z0  # (*BS, 1)
        vz = ξt + t * ξ̇t - (1 - σ1) * z0  # (*BS, 1)

        return torch.cat([va, vz], dim=-1)  # (*BS, 2)
