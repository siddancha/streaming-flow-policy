from typing import Dict, Optional
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from pydrake.all import PiecewisePolynomial
from torchdyn.core import NeuralODE

from streaming_flow_policy.pusht.dp_state_notebook.base_policy import Policy


class StreamingFlowPolicyDeterministic (Policy):
    def __init__(self,
                 velocity_net: nn.Module,
                 action_dim: int,
                 pred_horizon: int = 16,
                 sigma: float = 0.0,
                 device: torch.device = 'cuda',
        ):
        """
        Args:
            velocity_net (nn.Module): velocity network
            action_dim (int): action dimension
            pred_horizon (int): prediction horizon
            sigma (float): standard deviation of the Gaussian noise
            device (torch.device): device
        """
        super().__init__()
        self.velocity_net = velocity_net
        self.action_dim = action_dim
        self.device = device

        # Register pred_horizon and sigma as buffers if provided
        self.register_buffer('pred_horizon', torch.tensor(pred_horizon, dtype=torch.int32))
        self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float32))
        self.pred_horizon: Tensor; self.sigma: Tensor

    def TransformTrainingDatum(self, datum: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Args:
            datum (Dict[str, np.ndarray]):
                'obs' (np.ndarray, shape=(OBS_HORIZON, OBS_DIM), dtype=np.float32)
                'action' (np.ndarray, shape=(PRED_HORIZON, ACTION_DIM), dtype=np.float32)

        Returns:
            Dict[str, np.ndarray]:
                'obs' (np.ndarray, shape=(OBS_HORIZON, OBS_DIM), dtype=np.float32)
                'x' (np.ndarray, shape=(ACTION_DIM,), dtype=np.float32): position
                'u' (np.ndarray, shape=(ACTION_DIM,), dtype=np.float32): velocity
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
        x = traj.value(time).T  # (1, ACTION_DIM)
        u = traj.EvalDerivative(time).T  # (1, ACTION_DIM)

        # Add noise to position
        x = x + self.sigma.item() * np.random.randn(*x.shape)  # (1, ACTION_DIM)
        x = x.astype(np.float32)  # (1, ACTION_DIM)

        return {
            'obs': obs,  # (OBS_HORIZON, OBS_DIM)
            'x': x.astype(np.float32),  # (1, ACTION_DIM)
            'u': u.astype(np.float32),  # (1, ACTION_DIM)
            't': time,  # (,)
        }

    @torch.enable_grad()
    def Loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            batch (Dict[str, Tensor]):
                'obs' (Tensor, shape=(B, OBS_HORIZON, OBS_DIM))
                'x' (Tensor, shape=(B, 1, ACTION_DIM))
                'u' (Tensor, shape=(B, 1, ACTION_DIM))
                't' (Tensor, shape=(B,))

        Returns:
            Tensor (shape=(,), dtype=torch.float32): loss
        """
        # device transfer
        obs = batch['obs'].to(self.device)  # (B, OBS_HORIZON, OBS_DIM)
        x = batch['x'].to(self.device)  # (B, 1, ACTION_DIM)
        u = batch['u'].to(self.device)  # (B, 1, ACTION_DIM)
        t = batch['t'].to(self.device)  # (B,)
        B = obs.shape[0]

        # observation as FiLM conditioning
        obs_cond = obs.flatten(start_dim=1)  # (B, OBS_HORIZON * OBS_DIM)

        # predict the velocity
        u_pred = self.velocity_net(
            sample=x, timestep=t, global_cond=obs_cond
        )  # (B, ACTION_DIM)

        # L2 loss
        loss = nn.functional.mse_loss(u_pred, u)  # (,)
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

        x0 = nobs[-1, :2]  # (2,)

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

        traj = ode_solver.trajectory(x=x0, t_span=t_span)  # (K, 2)

        naction = traj[select_action_indices]  # (NUM_ACTIONS, 2)
        naction = naction.unsqueeze(0)  # (1, NUM_ACTIONS, 2)
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
            x (Tensor, shape=(ACTION_DIM,), dtype=float): position

        Returns:
            Tensor (shape=(ACTION_DIM,), dtype=float): velocity
        """
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, ACTION_DIM)
        v: Tensor = self.model(
            sample=x,
            timestep=t.repeat(x.shape[0]),
            global_cond=self.obs_cond,
        )  # (1, 1, ACTION_DIM)
        v = v.flatten()  # (ACTION_DIM,)
        return v
