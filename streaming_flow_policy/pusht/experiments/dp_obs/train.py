"""
All functions in this file have been copied from the Diffusion Policy repo, in
particular, this notebook (diffusion_policy_state_pusht_demo.ipynb):
https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B
"""
# diffusion policy import
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from streaming_flow_policy.pusht.dataset import PushTStateDatasetWithNextObsAsAction
from streaming_flow_policy.pusht.dp_state_notebook.network import ConditionalUnet1D


"""
|o|o|                             observations: 2
| |a|a|a|a|a|a|a|a|               actions executed: 8
|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
"""

# =============================================================================
# Parameters
# =============================================================================

pred_horizon = 16
obs_horizon = 2
action_horizon = 8
obs_dim = 5
action_dim = 2
num_epochs = 100
num_diffusion_iters = 100
save_path = f"models/pusht_dp_obs_{num_epochs}ep.pth"

# =============================================================================

# create dataset from file
dataset = PushTStateDatasetWithNextObsAsAction(
    pred_horizon=16,
    obs_horizon=2,
    action_horizon=8,
)

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    num_workers=1,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process after each epoch
    persistent_workers=True
)

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# device transfer
device = torch.device('cuda')
_ = noise_pred_net.to(device)

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(parameters=noise_pred_net.parameters(), power=0.75)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(),
    lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # data normalized in dataset
                # device transfer
                nobs = nbatch['obs'].to(device)
                naction = nbatch['action'].to(device)
                B = nobs.shape[0]

                # observation as FiLM conditioning
                obs_cond = nobs[:,:obs_horizon,:]  # (B, OBS_HORIZON, OBS_DIM)
                obs_cond = obs_cond.flatten(start_dim=1)  # (B, OBS_HORIZON * OBS_DIM)

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps)

                # predict the noise residual
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond)

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(noise_pred_net.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)
        tglobal.set_postfix(loss=np.mean(epoch_loss))

# Weights of the EMA model
# is used for inference
ema_noise_pred_net = noise_pred_net
ema.copy_to(ema_noise_pred_net.parameters())

# Save model
torch.save(ema_noise_pred_net.state_dict(), save_path)
print(f"Saved model to {save_path}.")
