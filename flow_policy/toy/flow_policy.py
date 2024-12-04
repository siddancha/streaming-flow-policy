from typing import List, Tuple
import numpy as np

from pydrake.all import Trajectory

from flow_policy.toy.base_policy import StreamingFlowPolicyBase


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

    def Ab(self, traj: Trajectory, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            A (np.ndarray, dtype=float, shape=(1, 1)): Transition matrix.
            b (np.ndarray, dtype=float, shape=(1,)): Bias vector.
        """
        q̃0 = traj.value(0).item()
        q̃t = traj.value(t).item()
        k = self.k

        exp_neg_kt = np.exp(-k * t)
        A = np.array([[exp_neg_kt]])  # (1, 1)
        b = np.array([q̃t - q̃0 * exp_neg_kt])  # (1,)
        return A, b

    def μΣ0(self, traj: Trajectory) -> np.ndarray:
        """
        Compute the mean and covariance matrix of the conditional flow at time t=0.

        Returns:
            np.ndarray, dtype=float, shape=(1,): Mean at time t=0.
            np.ndarray, dtype=float, shape=(1, 1): Covariance matrix at time t=0.
        """
        q̃0 = traj.value(0).item()
        σ0 = self.σ0
        μ0 = np.array([q̃0])  # (1,)
        Σ0 = np.array([[np.square(σ0)]])  # (1, 1)
        return μ0, Σ0

    def u_conditional(self, traj: Trajectory, x: np.ndarray, t: float) -> np.ndarray:
        """
        Compute the conditional velocity field for a given trajectory.

        • Conditional velocity field at (x, t) is:
            • u(x, t) = -k(q(t) - q̃(t)) + ṽ(t)

        • Flow trajectory at time t:
            • q(t) - q̃(t) = (q₀ - q̃(0)) exp(-kt)

        Args:
            traj (Trajectory): Demonstration trajectory.
            x (np.ndarray, dtype=float, shape=(1,)): Configuration value.
            t (float): Time value in [0,1].
            
        Returns:
            (np.ndarray, dtype=float, shape=(1)): Velocity of the conditional flow.
        """
        qt = x  # (1,)
        q̃t = traj.value(t).item()
        ṽt = traj.EvalDerivative(t).item()
        k = self.k

        u = -k * (qt - q̃t) + ṽt  # (1,)
        return u
