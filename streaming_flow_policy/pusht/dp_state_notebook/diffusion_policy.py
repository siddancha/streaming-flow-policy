import torch
from torch import Tensor
import torch.nn as nn

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from streaming_flow_policy.pusht.dp_state_notebook.base_policy import Policy


class DiffusionPolicy (Policy):
    def __init__(self,
                 noise_pred_net: nn.Module,
                 num_diffusion_iters: int,
                 pred_horizon: int,
                 action_dim: int,
                 device: torch.device,
        ):
        self.noise_pred_net = noise_pred_net
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.device = device
        self.num_diffusion_iters = num_diffusion_iters

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            # the choice of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )

    def __call__(self, nobs: Tensor) -> Tensor:
        """
        Args:
            nobs (Tensor, shape=(OBS_HORIZON, OBS_DIM)): normalized observations

        Returns:
            Tensor (shape=(1, NUM_ACTIONS, ACTION_DIM)): predicted actions
        """
        obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)  # (1, OBS_HORIZON * OBS_DIM)
        B = obs_cond.shape[0]

        # initialize action from Gaussian noise
        noisy_action = torch.randn(
            (B, self.pred_horizon, self.action_dim), device=self.device)
        naction = noisy_action

        # init scheduler
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = self.noise_pred_net(
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        return naction
