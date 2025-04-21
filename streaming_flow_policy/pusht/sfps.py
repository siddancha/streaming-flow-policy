from typing import Dict, Optional
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from pydrake.all import PiecewisePolynomial
from torchdyn.core import NeuralODE

from streaming_flow_policy.pusht.dp_state_notebook.base_policy import Policy


class StreamingFlowPolicyStochastic (Policy):
    def __init__(self,
                 velocity_net: nn.Module,
                 action_dim: int,
                 σ0: float = 0.0,
                 σ1: float = 0.0,
                 pred_horizon: int = 16,
                 device: torch.device = 'cuda',
        ):
        """
        Conditional flow:
        • At time t=0, we sample:
            • q₀ ~ N(ξ(0), σ₀)
            • z₀ ~ N(0, 1)

        • Flow trajectory at time t:
            • q(t) = q₀ + (ξ(t) - ξ(0)) + (σᵣt) z₀
            • z(t) = (1 - (1-σ₁)t)z₀ + tξ(t)
              • z starts from a pure noise sample z₀ that drifts towards the
              trajectory. Therefore, z(t) is uncorrelated with q at t=0, but
              eventually becomes very informative of the trajectory.

        • Conditional velocity field:
            • vq(q, z, t) = ξ̇(t) + σᵣz₀
            • vz(q, z, t) = ξ(t) + tξ̇(t) - (1-σ₁)z₀

        Args:
            velocity_net (nn.Module): velocity network
            action_dim (int): action dimension
            pred_horizon (int): prediction horizon
            σ0 (float): standard deviation of conditional probability flow at t=0.
            σ1 (float): standard deviation of conditional probability flow at t=1.
            device (torch.device): device
        """
        super().__init__()
        assert 0 <= σ0 <= σ1, "σ0 must be less than or equal to σ1"
        σr = np.sqrt(np.square(σ1) - np.square(σ0))

        self.velocity_net = velocity_net
        self.action_dim = action_dim
        self.device = device

        # Register pred_horizon and sigma as buffers if provided
        self.register_buffer('pred_horizon', torch.tensor(pred_horizon, dtype=torch.int32))
        self.register_buffer('σ0', torch.tensor(σ0, dtype=torch.float32))
        self.register_buffer('σ1', torch.tensor(σ1, dtype=torch.float32))
        self.register_buffer('σr', torch.tensor(σr, dtype=torch.float32))
        self.pred_horizon: Tensor; self.σ0: Tensor; self.σ1: Tensor; self.σr: Tensor

    def TransformTrainingDatum(self, datum: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Args:
            datum (Dict[str, np.ndarray]):
                'obs' (np.ndarray, shape=(OBS_HORIZON, OBS_DIM), dtype=np.float32)
                'action' (np.ndarray, shape=(PRED_HORIZON, ACTION_DIM), dtype=np.float32)

        Returns:
            Dict[str, np.ndarray]:
                'obs' (np.ndarray, shape=(OBS_HORIZON, OBS_DIM), dtype=np.float32)
                'q' (np.ndarray, shape=(1, ACTION_DIM), dtype=np.float32): configuration
                'z' (np.ndarray, shape=(1, ACTION_DIM), dtype=np.float32): latent variable
                'vq' (np.ndarray, shape=(1, ACTION_DIM), dtype=np.float32): target q-velocity
                'vz' (np.ndarray, shape=(1, ACTION_DIM), dtype=np.float32): target z-velocity
                't' (np.ndarray, shape=(,), dtype=np.float32): time
        """
        obs, action = datum['obs'], datum['action']
        OBS_HORIZON, OBS_DIM = obs.shape
        PRED_HORIZON, ACTION_DIM = action.shape
        assert PRED_HORIZON == self.pred_horizon.item()
        assert OBS_HORIZON == 2  # logic currently only works for history of length 2

        # Ensure that the first action matches the last observation
        # This may not happen when the sequence starts from the beginning of an
        # episode and both obs and action are duplicated. Then, hack this by
        # setting the first action to the last observation. However, the right
        # thing to do is:
        # TODO (Sid): Set the first action correctly when creating the dataset.
        if not np.all(action[0] == obs[-1, :2]):
            action = action.copy()
            action[0] = obs[-1, :2]

        # Create a trajectory from the action sequence.
        traj_times = np.linspace(0, 1, PRED_HORIZON)  # (PRED_HORIZON,)
        traj_positions = action  # (PRED_HORIZON, ACTION_DIM)
        traj: PiecewisePolynomial = PiecewisePolynomial.FirstOrderHold(
            traj_times, traj_positions.T,
        )

        time = np.float32(np.random.rand())  # (,)
        ξt = traj.value(time).T  # (1, ACTION_DIM)
        ξ̇t = traj.EvalDerivative(time).T  # (1, ACTION_DIM)
        σ0 = self.σ0.item()
        σ1 = self.σ1.item()
        σr = self.σr.item()

        # Sample z0 from N(0, 1)
        z0 = np.random.randn(1, ACTION_DIM)  # (1, ACTION_DIM)

        # Sample qt
        ε_q0 =  σ0 * np.random.randn(1, ACTION_DIM)  # (1, ACTION_DIM)
        qt = ξt + ε_q0 + σr * time * z0  # (1, ACTION_DIM)

        # Sample zt
        zt = (1 - (1-σ1) * time) * z0 + time * ξt

        # Compute conditional flow
        vq = ξ̇t + σr * z0  # (1, ACTION_DIM)
        vz = ξt + time * ξ̇t - (1 - σ1) * z0  # (1, ACTION_DIM)

        return {
            'obs': obs,  # (OBS_HORIZON, OBS_DIM)
            'q': qt.astype(np.float32),  # (1, ACTION_DIM)
            'z': zt.astype(np.float32),  # (1, ACTION_DIM)
            'vq': vq.astype(np.float32),  # (1, ACTION_DIM)
            'vz': vz.astype(np.float32),  # (1, ACTION_DIM)
            't': time,  # (,)
        }

    @torch.enable_grad()
    def Loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            batch (Dict[str, Tensor]):
                'obs' (Tensor, shape=(B, OBS_HORIZON, OBS_DIM))
                'q' (Tensor, shape=(B, 1, ACTION_DIM))
                'z' (Tensor, shape=(B, 1, ACTION_DIM), dtype=np.float32): latent variable
                'vq' (Tensor, shape=(B, 1, ACTION_DIM), dtype=np.float32): target q-velocity
                'vz' (Tensor, shape=(B, 1, ACTION_DIM), dtype=np.float32): target z-velocity
                't' (Tensor, shape=(B,)): time

        Returns:
            Tensor (shape=(,), dtype=torch.float32): loss
        """
        # device transfer
        obs = batch['obs'].to(self.device)  # (B, OBS_HORIZON, OBS_DIM)
        q = batch['q'].to(self.device)  # (B, 1, ACTION_DIM)
        z = batch['z'].to(self.device)  # (B, 1, ACTION_DIM)
        vq = batch['vq'].to(self.device)  # (B, 1, ACTION_DIM)
        vz = batch['vz'].to(self.device)  # (B, 1, ACTION_DIM)
        t = batch['t'].to(self.device)  # (B,)
        B = obs.shape[0]

        # observation as FiLM conditioning
        obs_cond = obs.flatten(start_dim=1)  # (B, OBS_HORIZON * OBS_DIM)

        # concatenate q and z
        x = torch.cat((q, z), dim=-2)  # (B, 2, ACTION_DIM)
        v_target = torch.cat((vq, vz), dim=-2)  # (B, 2, ACTION_DIM)

        # predict the velocity
        v_pred = self.velocity_net(
            sample=x, timestep=t, global_cond=obs_cond
        )  # (B, 2, ACTION_DIM)

        # L2 loss
        loss = nn.functional.mse_loss(v_pred, v_target)  # (,)
        return loss

    @torch.inference_mode()
    def __call__(self,
                 nobs: Tensor,
                 num_actions: Optional[int] = None,
                 integration_steps_per_action: int = 6,
    ) -> Tensor:
        """
        Args:
            nobs (Tensor, shape=(OBS_HORIZON, OBS_DIM)): normalized observations
            num_actions (Optional[int]): number of actions to predict
            integration_steps_per_action (int): number of integration steps per action

        Returns:
            Tensor (shape=(1, NUM_ACTIONS, ACTION_DIM)): predicted actions
        """
        obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)  # (1, OBS_HORIZON * OBS_DIM)
        ode_solver = NeuralODE(
            vector_field=VectorFieldWrapper(self.velocity_net, obs_cond),
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )

        # Set to current configuration.
        q0 = nobs[-1, :2]  # (ACTION_DIM,)

        # Sample latent variable -- this is the stochastic step.
        z0 = torch.randn_like(q0)  # (ACTION_DIM,)

        # Integration time steps.
        num_actions = num_actions or self.pred_horizon.item()
        assert 1 <= num_actions <= self.pred_horizon.item()
        num_future_actions = num_actions - 1
        t_max = num_future_actions / (self.pred_horizon.item() - 1)
        total_integration_steps = 1 + num_future_actions * integration_steps_per_action
        t_span = torch.linspace(0, t_max, total_integration_steps)
        select_action_indices = np.arange(
            0,
            total_integration_steps,
            integration_steps_per_action,
        )

        x0 = torch.stack((q0, z0), dim=0)  # (2, ACTION_DIM)
        traj = ode_solver.trajectory(x=x0, t_span=t_span)  # (K, 2, ACTION_DIM)

        naction = traj[select_action_indices, 0, :]  # (NUM_ACTIONS, ACTION_DIM)
        naction = naction.unsqueeze(0)  # (1, NUM_ACTIONS, ACTION_DIM)
        return naction


class VectorFieldWrapper (nn.Module):
    """Wraps model to torchdyn compatible format."""
    def __init__(self, model: nn.Module, obs_cond: Tensor):
        super().__init__()
        self.model = model
        self.obs_cond = obs_cond

    def forward(self, t: Tensor, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Args:
            t (Tensor, shape=(,), dtype=float): time
            x (Tensor, shape=(2, ACTION_DIM), dtype=float): position

        Returns:
            Tensor (shape=(2, ACTION_DIM), dtype=float): velocity
        """
        x = x.unsqueeze(0)  # (1, 2, ACTION_DIM)
        v: Tensor = self.model(
            sample=x,
            timestep=t.repeat(x.shape[0]),
            global_cond=self.obs_cond,
        )  # (1, 2, ACTION_DIM)
        v = v.squeeze(0)  # (2, ACTION_DIM)
        return v
