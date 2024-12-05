from typing import List, Tuple
import numpy as np
import torch; torch.set_default_dtype(torch.double)
from torch import Tensor

from pydrake.all import Trajectory

from flow_policy.toy.sfp_base import StreamingFlowPolicyBase


class StreamingFlowPolicyDeterministic (StreamingFlowPolicyBase):
    def __init__(
        self,
        trajectories: List[Trajectory],
        prior: List[float],
        σ0: float,
        k: float = 0.0,
    ):
        """
        Flow policy with stabilizing conditional flow.

        Let q̃(t) be the demonstration trajectory. And its velocity be ṽ(t).

        Conditional flow:
        • At time t=0, we sample:
            • q₀ ~ N(q̃(0), σ₀)

        • Conditional velocity field at (x, t) is:
            • u(x, t) = -k(q(t) - q̃(t)) + ṽ(t)

        • Flow trajectory at time t:
            • q(t) - q̃(t) = (q₀ - q̃(0)) exp(-kt)
              • The error from the trajectory decreases exponentially with time.

        Args:
            trajectories (List[Trajectory]): List of trajectories.
            prior (np.ndarray, dtype=float, shape=(K,)): Prior
                probabilities for each trajectory.
            sigma (float): Standard deviation of the Gaussian distribution.
            gain (float): Gain for stabilizing the conditional flow around a
                demonstration trajectory.

        """
        super().__init__(dim=1, trajectories=trajectories, prior=prior)

        assert σ0 >= 0.0
        assert k >= 0.0

        self.σ0 = σ0
        self.k = k

    def Ab(self, traj: Trajectory, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            traj (Trajectory): Demonstration trajectory.
            t (Tensor, dtype=double, shape=(*BS)): Time values in [0,1].

        Returns:
            A (Tensor, dtype=double, shape=(*BS, 1, 1)): Transition matrix.
            b (Tensor, dtype=double, shape=(*BS, 1)): Bias vector.
        """
        q̃0: float = traj.value(0).item()
        q̃t = self.q̃t(traj, t)  # (*BS, 1)
        k = self.k

        exp_neg_kt = torch.exp(-k * t)  # (*BS)
        A = exp_neg_kt.unsqueeze(-1).unsqueeze(-1)  # (*BS, 1, 1)
        b = q̃t - (q̃0 * exp_neg_kt).unsqueeze(-1)  # (*BS, 1)
        return A, b

    def μΣ0(self, traj: Trajectory) -> Tuple[Tensor, Tensor]:
        """
        Compute the mean and covariance matrix of the conditional flow at time t=0.

        Returns:
            Tensor, dtype=double, shape=(1,): Mean at time t=0.
            Tensor, dtype=double, shape=(1, 1): Covariance matrix at time t=0.
        """
        q̃0: float = traj.value(0).item()
        σ0 = self.σ0
        μ0 = torch.tensor([q̃0], dtype=torch.double)  # (1,)
        Σ0 = torch.tensor([[np.square(σ0)]], dtype=torch.double)  # (1, 1)
        return μ0, Σ0

    def u_conditional(self, traj: Trajectory, x: Tensor, t: Tensor) -> Tensor:
        """
        Compute the conditional velocity field for a given trajectory.

        • Conditional velocity field at (x, t) is:
            • u(x, t) = -k(q(t) - q̃(t)) + ṽ(t)

        • Flow trajectory at time t:
            • q(t) - q̃(t) = (q₀ - q̃(0)) exp(-kt)

        Args:
            traj (Trajectory): Demonstration trajectory.
            x (Tensor, dtype=double, shape=(*BS, 1)): Configuration values.
            t (Tensor, dtype=double, shape=(*BS)): Time values in [0,1].

        Returns:
            Tensor, dtype=double, shape=(*BS, 1): Velocity of the conditional flow.
        """
        qt = x  # (*BS, 1)
        q̃t = self.q̃t(traj, t)  # (*BS, 1)
        ṽt = self.ṽt(traj, t)  # (*BS, 1)
        k = self.k

        u = -k * (qt - q̃t) + ṽt  # (*BS, 1)
        return u
