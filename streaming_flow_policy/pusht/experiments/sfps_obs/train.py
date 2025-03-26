"""
All functions in this file have been copied from the Diffusion Policy repo, in
particular, this notebook (diffusion_policy_state_pusht_demo.ipynb):
https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B
"""
import numpy as np
import torch
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from streaming_flow_policy.pusht.dataset import PushTStateDatasetWithNextObsAsAction
from streaming_flow_policy.pusht.dp_state_notebook.network import ConditionalUnet1D
from streaming_flow_policy.pusht.sfps import StreamingFlowPolicyStochastic


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
σ0 = 0.1  # 0.05
σ1 = 0.1  # 0.5
num_epochs = 1000
batch_size = 1024
save_path = f"models/pusht_sfps_obs_{num_epochs}ep.pth"

# =============================================================================

# create network object
velocity_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon,
    fc_timesteps=2,
)

# device transfer
device = torch.device('cuda')
_ = velocity_net.to(device)

policy = StreamingFlowPolicyStochastic(
    velocity_net=velocity_net,
    action_dim=action_dim,
    pred_horizon=pred_horizon,
    σ0=σ0, σ1=σ1,
    device=device,
)

# create dataset from file
dataset = PushTStateDatasetWithNextObsAsAction(
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon,
    transform_datum_fn=policy.TransformTrainingDatum,
)

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=1,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process after each epoch
    persistent_workers=True
)

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(parameters=velocity_net.parameters(), power=0.75)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=velocity_net.parameters(),
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
                # L2 loss
                loss = policy.Loss(nbatch)

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(velocity_net.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)
        tglobal.set_postfix(loss=np.mean(epoch_loss))

# Weights of the EMA model
# is used for inference
ema_velocity_net = policy.velocity_net
ema.copy_to(ema_velocity_net.parameters())

# Save model
torch.save(policy.state_dict(), save_path)
print(f"Saved model to {save_path}.")
