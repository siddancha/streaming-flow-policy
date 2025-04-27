import numpy as np
import torch
from torch import Tensor
from typing import List, Tuple

from pydrake.all import Trajectory

from streaming_flow_policy.core.sfpl_base import StreamingFlowPolicyLatentBase


class StreamingFlowPolicyLatent (StreamingFlowPolicyLatentBase):
    def __init__(
        self,
        dim: int,
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
        Define constant σᵣ = √(σ₁² - σ₀²exp(-2k)).
        Note that σ₁² = σ₀²exp(-2k) + σᵣ².

        Conditional flow:
        • At time t=0, we sample:
            • q₀ ~ N(ξ(0), σ₀²)
            • z₀ ~ N(0, 1)

        • Flow trajectory at time t:
            • q(t) = ξ(t) + (q₀ - ξ(0)) exp(-kt) + σᵣtz₀
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
        super().__init__(dim=dim, trajectories=trajectories, prior=prior, σ0=σ0)

        self.σ1 = σ1
        self.k = k

        # Residual standard deviation: √(σ₁² - σ₀²exp(-2k))
        assert 0 <= σ0 * np.exp(-k) <= σ1, "σ1 is too small relative to σ0"
        self.σr = np.sqrt(np.square(σ1) - np.square(σ0) * np.exp(-2 * k))

    def Ab(self, traj: Trajectory, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            traj (Trajectory): Demonstration trajectory.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].

        Returns:
            A (Tensor, dtype=default, shape=(*BS, 2D, 2D)): Transition matrix.
            b (Tensor, dtype=default, shape=(*BS, 2D)): Bias vector.
        """
        I = torch.eye(self.D)  # (D, D)
        O = torch.zeros(self.D, self.D)  # (D, D)
        
        σ1 = self.σ1  # (,)
        σr = self.σr  # (,)
        k = self.k  # (,)

        ξ0 = self.ξt(traj, torch.tensor(0.))  # (D,)
        ξt = self.ξt(traj, t)  # (*BS, D)
        αt = torch.exp(-k * t)  # (*BS)

        # Compute b
        t = t.unsqueeze(-1)  # (*BS, 1)
        αt = αt.unsqueeze(-1)  # (*BS, 1)
        b = torch.cat([ξt - ξ0 * αt, t * ξt], dim=-1)  # (*BS, 2D)

        # Compute A
        t = t.unsqueeze(-1) * I  # (*BS, D, D)
        αt = αt.unsqueeze(-1) * I  # (*BS, D, D)
        A = self.block_matrix([
            [αt,           σr * t],
            [ O, 1 - (1 - σ1) * t],
        ])  # (*BS, 2D, 2D)
        return A, b

    def v_conditional(self, traj: Trajectory, x: Tensor, t: Tensor) -> Tensor:
        """
        Compute the conditional velocity field for a given trajectory.

        • Flow trajectory at time t:
            • q(t) = ξ(t) + (q₀ - ξ(0)) exp(-kt) + σᵣtz₀
            • z(t) = (1 - (1-σ₁)t)z₀ + tξ(t)

        • Conditional velocity field:
            • First, given q(t) and z(t), we want to compute q₀ and z₀.
                • z₀ = (z(t) - tξ(t)) / (1 - (1-σ₁)t)
                • q₀ = ξ(0) + (q(t) - ξ(t) - σᵣtz₀) exp(kt)
            • Then, we compute the velocity for the conditional flow.
                • vq(q, z, t) = ξ̇(t) -k(q₀ - ξ(0))exp(-kt) + σᵣz₀
                • vz(q, z, t) = ξ(t) + tξ̇(t) - (1-σ₁)z₀
            • Plugging (z₀, q₀) into the velocity gives us the velocity field:
                • vq(q, z, t) = ξ̇(t) - k(q - ξ(t)) + σᵣ(1 + kt) / (1 - (1-σ₁)t) * (z - tξ(t))
                • vz(q, z, t) = ξ(t) + tξ̇(t) - (1-σ₁) / (1 - (1-σ₁)t) * (z - tξ(t))

        Args:
            traj (Trajectory): Demonstration trajectory.
            x (Tensor, dtype=default, shape=(*BS, 2D)): State values.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].
            
        Returns:
            (Tensor, dtype=default, shape=(*BS, 2D)): Velocity of conditional flow.
        """
        σ1 = self.σ1
        σr = self.σr
        k = self.k

        qt = x[..., self.slice_q]  # (*BS, D)
        zt = x[..., self.slice_z]  # (*BS, D)
        ξt = self.ξt(traj, t)  # (*BS, D)
        ξ̇t = self.ξ̇t(traj, t)  # (*BS, D)
        t = t.unsqueeze(-1)  # (*BS, 1)

        # Invert zt to get z0
        z0 = (zt - t * ξt) / (1 - (1 - σ1) * t)  # (*BS, D)

        # Compute velocity field
        vq = ξ̇t - k * (qt - ξt) + σr * (1 + k * t) * z0  # (*BS, D)
        vz = ξt + t * ξ̇t - (1 - σ1) * z0  # (*BS, D)

        return torch.cat([vq, vz], dim=-1)  # (*BS, 2D)

    def 𝔼vq_conditional(self, traj: Trajectory, q: Tensor, t: Tensor) -> Tensor:
        """
        Compute the expected velocity field of q over z given q, t and a trajectory.

        The velocity field is given by:
            • vq(q, z, t) = ξ̇(t) - k(q - ξ(t)) + σᵣ(1 + kt) / (1 - (1-σ₁)t) * (z - tξ(t))
        
        Therefore, the expected velocity under N(μ_z|q, Σ_z|q) is given by:
            • 𝔼[vq(q, z, t)] = ξ̇(t) - k(q - ξ(t)) + σᵣ(1 + kt) / (1 - (1-σ₁)t) * (μ_z|q - tξ(t))

        Args:
            traj (Trajectory): Demonstration trajectory.
            q (Tensor, dtype=default, shape=(*BS, D)): Configuration.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=default, shape=(*BS, D)):
                expected value of vq over z given q, t and a trajectory.
        """
        σ1 = self.σ1
        σr = self.σr
        k = self.k

        μ_zCq, Σ_zCq = self.μΣt_zCq(traj, t, q)  # (*BS, D), (*BS, D, D)

        ξt = self.ξt(traj, t)  # (*BS, D)
        ξ̇t = self.ξ̇t(traj, t)  # (*BS, D)
        t = t.unsqueeze(-1)  # (*BS, 1)

        # Expected z0 given q
        μ_z0Cq = (μ_zCq - t * ξt) / (1 - (1 - σ1) * t)  # (*BS, D)

        # Compute expected velocity field
        𝔼vq = ξ̇t - k * (q - ξt) + σr * (1 + k * t) * μ_z0Cq  # (*BS, D)
        return 𝔼vq

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
            z (Tensor, dtype=default, shape=(*BS, D)): Latent variable value.
            t (Tensor, dtype=default, shape=(*BS)): Time value in [0,1].

        Returns:
            (Tensor, dtype=default, shape=(*BS, D)):
                expected value of vz given z, t and a trajectory.
        """
        σ1 = self.σ1

        ξt = self.ξt(traj, t)  # (*BS, D)
        ξ̇t = self.ξ̇t(traj, t)  # (*BS, D)
        t = t.unsqueeze(-1)  # (*BS, 1)

        # Invert zt to get z0
        z0 = (z - t * ξt) / (1 - (1 - σ1) * t)  # (*BS, D)

        # Compute expected velocity field
        𝔼vz = ξt + t * ξ̇t - (1 - σ1) * z0  # (*BS, D)
        return 𝔼vz
