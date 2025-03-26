from typing import List

import torch
from torch import Tensor

from streaming_flow_policy.traj import Trajectory

class QuinticHermitePolynomial:
    def __init__(self, t0: float, Δt: float, x0: float, m0: float, a0: float, 
                 x1: float, m1: float, a1: float):
        self.t0 = t0
        self.Δt = Δt
        self.x0 = x0
        self.m0 = m0
        self.a0 = a0
        self.x1 = x1
        self.m1 = m1
        self.a1 = a1

        self.coefficient_matrix = torch.tensor([
            [ 1,  0,  0, -10,  15,  -6],
            [ 0,  1,  0,  -6,   8,  -3], 
            [ 0,  0,  0.5, -1.5, 1.5, -0.5],
            [ 0,  0,  0,  10, -15,   6],
            [ 0,  0,  0,  -4,   7,  -3],
            [ 0,  0,  0,  0.5, -1.0,  0.5]
        ], dtype=torch.float)  # (6, 6)

    def x(self, t: Tensor) -> Tensor:
        """
        Returns the values of the quintic hermite polynomials
        
        Args:
            t (Tensor, dtype=float, shape=(*BS,)): Time points.
        Returns:
            Tensor, dtype=float, shape=(*BS,): Values of the polynomial.
        """
        # Rescale time to [0, 1]
        t = (t - self.t0) / self.Δt  # (*BS,)

        # Compute powers of t from t⁰ to t⁵
        time_powers = t[..., None] ** torch.arange(6)  #(*BS, 6)

        # Multiply coefficient matrix by powers to get polynomial terms
        hh = self.coefficient_matrix @ time_powers.T  # (*BS, 6)

        return (
            hh[..., 0] * self.x0 + \
            hh[..., 1] * self.m0 * self.Δt + \
            hh[..., 2] * self.a0 * self.Δt**2 + \
            hh[..., 3] * self.x1 + \
            hh[..., 4] * self.m1 * self.Δt + \
            hh[..., 5] * self.a1 * self.Δt**2
        )  # (*BS,)

    def ẋ(self, t: Tensor) -> Tensor:
        """
        Returns the derivative of the polynomial with respect to t.
        Uses torch autograd for automatic differentiation.
        
        Args:
            t (Tensor, dtype=float, shape=(*BS,)): Time points.
        Returns:
            Tensor, dtype=float, shape=(*BS,): Values of the derivative.
        """
        t = t.detach().requires_grad_(True)
        
        # Compute x(t)
        x_t = self.x(t)  # (*BS)
        
        # Compute gradient
        ẋ = torch.autograd.grad(x_t.sum(), t)[0]  # (*BS)

        return ẋ.detach()

class QuinticHermiteInterpolation:
    def __init__(
            self,
            seed_trajectory: Trajectory,
            initial_slope: float | None = None,
            final_slope: float | None = None,
            initial_accel: float | None = None,
            final_accel: float | None = None,
    ):
        seed_x, seed_t = seed_trajectory.x, seed_trajectory.t
        slopes = (seed_x[1:] - seed_x[:-1]) / (seed_t[1:] - seed_t[:-1])
        
        initial_slope = slopes[0] if initial_slope is None else initial_slope
        final_slope = slopes[-1] if final_slope is None else final_slope

        slopes = torch.cat([
            torch.tensor([initial_slope], device=slopes.device),
            (slopes[1:] + slopes[:-1]) / 2,
            torch.tensor([final_slope], device=slopes.device)
        ])

        # Deduce accelerations from seed trajectory
        accelerations = (slopes[1:] - slopes[:-1]) / (seed_t[1:] - seed_t[:-1])
        
        # Use deduced accelerations if not provided by user
        initial_accel = accelerations[0] if initial_accel is None else initial_accel
        final_accel = accelerations[-1] if final_accel is None else final_accel

        # Append initial and final accelerations
        accelerations = torch.cat([
            torch.tensor([initial_accel], device=accelerations.device),
            accelerations,
            torch.tensor([final_accel], device=accelerations.device)
        ])

        self.times: List[float] = seed_t
        self.polynomials: List[QuinticHermitePolynomial] = []
        
        for i in range(len(seed_t) - 1):
            poly = QuinticHermitePolynomial(
                t0 = seed_t[i],
                Δt = seed_t[i+1] - seed_t[i],
                x0 = seed_x[i],
                m0 = slopes[i],
                a0 = accelerations[i],
                x1 = seed_x[i+1],
                m1 = slopes[i+1],
                a1 = accelerations[i+1],
            )
            self.polynomials.append(poly)

    def x(self, ts: Tensor) -> Tensor:
        """
        Args:
            t (Tensor, dtype=float, shape=(N,)): Time.
        Returns:
            Tensor, dtype=float, shape=(N,)): Values of the polynomial.
        """
        indices: List[int] = torch.searchsorted(self.times[1:], ts)  # (N,)
        return torch.tensor([self.polynomials[index].x(t).item() for (index, t) in zip(indices, ts)])

    def ẋ(self, ts: Tensor) -> Tensor:
        """
        Args:
            ts (Tensor, dtype=float, shape=(N,)): Time.
        Returns:
            Tensor, dtype=float, shape=(N,)): Values of the derivative.
        """
        indices: List[int] = torch.searchsorted(self.times[1:], ts)  # (N,)
        return torch.tensor([self.polynomials[index].ẋ(t).item() for (index, t) in zip(indices, ts)])

    def MakeTrajectory(self, ts: Tensor) -> Trajectory:
        return Trajectory(x=self.x(ts), t=ts)
