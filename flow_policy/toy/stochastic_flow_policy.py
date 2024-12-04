import numpy as np
from scipy.stats import multivariate_normal
from typing import List, Tuple

from pydrake.all import Trajectory

from flow_policy.toy.base_policy import StreamingFlowPolicyBase


class StreamingFlowPolicyStochastic (StreamingFlowPolicyBase):
    def __init__(
        self,
        trajectories: List[Trajectory],
        prior: List[float],
        σ0: float,
        σ1: float,
    ):
        """
        Flow policy is an extended configuration space (q(t), z(t)) where q is
        the original trajectory and z is a noise variable that starts from
        N(0, 1).

        Let q̃(t) be the demonstration trajectory.
        Define constant σᵣ = √(σ₁² - σ₀²). Note that σ₁² = σ₀² + σᵣ².

        Conditional flow:
        • At time t=0, we sample:
            • q₀ ~ N(q̃(0), σ₀)
            • z₀ ~ N(0, 1)

        • Flow trajectory at time t:
            • q(t) = q₀ + (q̃(t) - q̃(0)) + (σᵣt) z₀
            • z(t) = (1 - (1-σ₁)t)z₀ + tq̃(t)
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

        # Residual standard deviation: √(σ₁² - σ₀²)
        self.σr = np.sqrt(np.square(σ1) - np.square(σ0))

    def Ab(self, traj: Trajectory, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            A (np.ndarray, dtype=float, shape=(2, 2)): Transition matrix.
            b (np.ndarray, dtype=float, shape=(2,)): Bias vector.
        """
        σ1 = self.σ1  # (,)
        σr = self.σr  # (,)
        q̃0 = traj.value(0).item()
        q̃t = traj.value(t).item()

        b = np.array([q̃t - q̃0, t * q̃t])
        A = np.array([
            [1,           σr * t],
            [0, 1 - (1 - σ1) * t],
        ])
        return A, b

    def μΣ0(self, traj: Trajectory) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the mean and covariance matrix of the conditional flow at time t=0.

        Returns:
            np.ndarray, dtype=float, shape=(2,): Mean at time t=0.
            np.ndarray, dtype=float, shape=(2, 2): Covariance matrix at time t=0.
        """
        q̃0 = traj.value(0).item()
        σ0 = self.σ0
        μ0 = np.array([q̃0, 0])  # (2,)
        Σ0 = np.array([[np.square(σ0), 0], [0, 1]])  # (2, 2)
        return μ0, Σ0

    def pdf_conditional_q(self, traj: Trajectory, q: float, t: float) -> float:
        """
        Compute probability of the conditional flow at configuration q and time
        t, for the given trajectory.
        
        Args:
            traj (Trajectory): Demonstration trajectory.
            q (float): Configuration.
            t (float): Time value in [0,1].
            
        Returns:
            float: Probability of the conditional flow at state x and time t.
        """
        assert isinstance(q, float)
        μ_qz, Σ_qz = self.μΣt(traj, t)
        μ_q, Σ_q = μ_qz[0], Σ_qz[0, 0]
        dist = multivariate_normal(mean=μ_q, cov=Σ_q)
        return dist.pdf(q)

    def pdf_marginal_q(self, q: float, t: float) -> float:
        """
        Compute probability of the marginal flow at configuration q and time t.
        
        Args:
            q (float): Configuration.
            t (float): Time value in [0,1].
            
        Returns:
            float: Probability of the marginal flow at configuration q and time t.
        """
        assert isinstance(q, float)
        prob = 0
        for π, traj in zip(self.π, self.trajectories):
            prob += π * self.pdf_conditional_q(traj, q, t)
        return prob

    def pdf_conditional_z(self, traj: Trajectory, z: float, t: float) -> float:
        """
        Compute probability of the conditional flow at z and time t, for each
        of the K trajectories.
        
        Args:
            traj (Trajectory): Demonstration trajectory.
            z (float): Latent variable value.
            t (float): Time value in [0,1].
            
        Returns:
            float: Probability of the conditional flow at z and time t.
        """
        assert isinstance(z, float)
        μ_qz, Σ_qz = self.μΣt(traj, t)
        μ_z, Σ_z = μ_qz[1], Σ_qz[1, 1]
        dist = multivariate_normal(mean=μ_z, cov=Σ_z)
        return dist.pdf(z)

    def pdf_marginal_z(self, z: float, t: float) -> float:
        """
        Compute probability of the marginal flow at z and time t.
        
        Args:
            z (float): Latent variable value.
            t (float): Time value in [0,1].
            
        Returns:
            float: Probability of the marginal flow at z and time t.
        """
        assert isinstance(z, float)
        prob = 0
        for π, traj in zip(self.π, self.trajectories):
            prob += π * self.pdf_conditional_z(traj, z, t)
        return prob

    def u_conditional(self, traj: Trajectory, x: np.ndarray, t: float) -> np.ndarray:
        """
        Compute the conditional velocity field for a given trajectory.

        • Flow trajectory at time t:
            • q(t) = q₀ + (q̃(t) - q̃(0)) + (σᵣt) z₀
            • z(t) = (1 - (1-σ₁)t)z₀ + tq̃(t)

        • Conditional velocity field:
            • First, given q(t) and z(t), we want to compute q₀ and z₀.
                • z₀ = (z(t) - tq̃(t)) / (1 - (1-σ₁)t)
                • q₀ = q(t) - (q̃(t) - q̃(0)) - (σᵣt) z₀
            • Then, we compute the velocity field for the conditional flow.
                • uq(q, z, t) = ṽ(t) + σᵣz₀
                • uz(q, z, t) = q̃(t) + tṽ(t) - (1-σ₁)z₀

        Args:
            traj (Trajectory): Demonstration trajectory.
            x (np.ndarray, dtype=float, shape=(2,)): State values.
            t (float): Time value in [0,1].
            
        Returns:
            (np.ndarray, dtype=float, shape=(2)): Velocity of the conditional flow.
        """
        qt, zt = x
        q̃t = traj.value(t).item()
        σ1 = self.σ1
        σr = self.σr
    
        ṽt = traj.EvalDerivative(t).item()

        # Invert the flow and transform (qt, zt) to (q0, z0)
        z0 = (zt - t * q̃t) / (1 - (1 - σ1) * t)

        # Compute velocity of the trajectory starting from (q0, z0) at t
        uq = ṽt + σr * z0
        uv = q̃t + t * ṽt - (1 - σ1) * z0

        return np.array([uq, uv])
