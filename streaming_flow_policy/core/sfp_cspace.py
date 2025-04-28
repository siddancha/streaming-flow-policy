from typing import List, Tuple
import torch
from torch import Tensor

from pydrake.all import Trajectory

from streaming_flow_policy.core.sfp_base import StreamingFlowPolicyBase


class StreamingFlowPolicyCSpace (StreamingFlowPolicyBase):
    def __init__(
        self,
        dim: int,
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
            dim (int): Dimension of the state space.
            trajectories (List[Trajectory]): List of trajectories.
            prior (np.ndarray, dtype=float, shape=(K,)): Prior
                probabilities for each trajectory.
            sigma (float): Standard deviation of the Gaussian distribution.
            gain (float): Gain for stabilizing the conditional flow around a
                demonstration trajectory.

        """
        super().__init__(dim=dim, trajectories=trajectories, prior=prior)

        assert σ0 >= 0.0
        assert k >= 0.0

        self.σ0 = σ0
        self.k = k

    def Ab(self, traj: Trajectory, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            traj (Trajectory): Demonstration trajectory.
            t (Tensor, dtype=default, shape=(*BS)): Time values in [0,1].

        Returns:
            A (Tensor, dtype=default, shape=(*BS, X, X)): Transition matrix.
            b (Tensor, dtype=default, shape=(*BS, X)): Bias vector.
        """
        k = self.k

        ξ0 = self.ξt(traj, torch.tensor(0.))  # (X,)
        ξt = self.ξt(traj, t)  # (*BS, X)
        αt = torch.exp(-k * t)  # (*BS)

        # Compute b
        t = t.unsqueeze(-1)  # (*BS, 1)
        αt = αt.unsqueeze(-1)  # (*BS, 1)
        b = ξt - ξ0 * αt  # (*BS, X)

        # Compute A
        αt = αt.unsqueeze(-1)  # (*BS, 1, 1)
        I = torch.eye(self.X)  # (X, X)
        A = αt * I  # (*BS, X, X)
        return A, b

    def μΣ0(self, traj: Trajectory) -> Tuple[Tensor, Tensor]:
        """
        Compute the mean and covariance matrix of the conditional flow at time t=0.

        Returns:
            Tensor, dtype=default, shape=(X,): Mean at time t=0.
            Tensor, dtype=default, shape=(X, X): Covariance matrix at time t=0.
        """
        I = torch.eye(self.X)  # (X, X)

        ξ0 = self.ξt(traj, torch.tensor(0.))  # (X,)
        σ0 = self.σ0 * I  # (X, X)

        μ0 = ξ0  # (X,)
        Σ0 = σ0.square()  # (X, X)
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
            x (Tensor, dtype=default, shape=(*BS, X)): State values.
            t (Tensor, dtype=default, shape=(*BS)): Time values in [0,1].

        Returns:
            (Tensor, dtype=default, shape=(*BS, X)): Velocity of conditional flow.
        """
        qt = x  # (*BS, X)
        ξt = self.ξt(traj, t)  # (*BS, X)
        ξ̇t = self.ξ̇t(traj, t)  # (*BS, X)
        k = self.k

        v = ξ̇t - k * (qt - ξt)  # (*BS, X)
        return v
