from typing import List, Tuple
import numpy as np
import torch; torch.set_default_dtype(torch.double)
from torch import Tensor

from pydrake.all import Trajectory

from streaming_flow_policy.core.sfp_base import StreamingFlowPolicyBase


class StreamingFlowPolicyCSpace (StreamingFlowPolicyBase):
    def __init__(
        self,
        trajectories: List[Trajectory],
        prior: List[float],
        σ0: float,
        k: float = 0.0,
    ):
        """
        Flow policy with stabilizing conditional flow.

        Let ξ(t) be the demonstration trajectory. And its velocity be ξ̇(t).

        Conditional flow:
        • At time t=0, we sample:
            • q₀ ~ N(ξ(0), σ₀)

        • Conditional velocity field at (x, t) is:
            • u(x, t) = -k(q(t) - ξ(t)) + ξ̇(t)

        • Flow trajectory at time t:
            • q(t) - ξ(t) = (q₀ - ξ(0)) exp(-kt)
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
        ξ0: float = traj.value(0).item()
        ξt = self.ξt(traj, t)[..., 0]  # (*BS)
        k = self.k

        exp_neg_kt = torch.exp(-k * t)  # (*BS)
        A = self.matrix_stack([[exp_neg_kt]])  # (*BS, 1, 1)
        b = (ξt - ξ0 * exp_neg_kt).unsqueeze(-1)  # (*BS, 1)
        return A, b

    def μΣ0(self, traj: Trajectory) -> Tuple[Tensor, Tensor]:
        """
        Compute the mean and covariance matrix of the conditional flow at time t=0.

        Returns:
            Tensor, dtype=double, shape=(1,): Mean at time t=0.
            Tensor, dtype=double, shape=(1, 1): Covariance matrix at time t=0.
        """
        ξ0: float = traj.value(0).item()
        σ0 = self.σ0
        μ0 = torch.tensor([ξ0], dtype=torch.double)  # (1,)
        Σ0 = torch.tensor([[np.square(σ0)]], dtype=torch.double)  # (1, 1)
        return μ0, Σ0

    def v_conditional(self, traj: Trajectory, x: Tensor, t: Tensor) -> Tensor:
        """
        Compute the conditional velocity field for a given trajectory.

        • Conditional velocity field at (x, t) is:
            • u(x, t) = -k(q(t) - ξ(t)) + ξ̇(t)

        • Flow trajectory at time t:
            • q(t) - ξ(t) = (q₀ - ξ(0)) exp(-kt)

        Args:
            traj (Trajectory): Demonstration trajectory.
            x (Tensor, dtype=double, shape=(*BS, 1)): State values.
            t (Tensor, dtype=double, shape=(*BS)): Time values in [0,1].

        Returns:
            (Tensor, dtype=double, shape=(*BS, 1)): Velocity of conditional flow.
        """
        qt = x  # (*BS, 1)
        ξt = self.ξt(traj, t)  # (*BS, 1)
        ξ̇t = self.ξ̇t(traj, t)  # (*BS, 1)
        k = self.k

        u = -k * (qt - ξt) + ξ̇t  # (*BS, 1)
        return u
