"""
2D Toy Comparison: DP  vs  SFP  vs  SFP + EqM-E
=================================================

Trains three models on the bimodal 2D toy dataset, then draws a side-by-side
comparison plot showing how well each method covers both modes.

Usage
-----
    cd <repo-root>
    python -m streaming_flow_policy.pusht.experiments.toy_2d.train_and_plot

Output
------
    toy_2d_comparison.png  — comparison figure saved to the working directory.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from streaming_flow_policy.pusht.dataset_2d_toy import ToyDataset2D, generate_trajectory
from streaming_flow_policy.pusht.dp_state_notebook.network import ConditionalUnet1D
from streaming_flow_policy.pusht.dp_state_notebook.dataset import (
    normalize_data, unnormalize_data,
)
from streaming_flow_policy.pusht.dp_state_notebook.diffusion_policy import DiffusionPolicy
from streaming_flow_policy.pusht.sfps import StreamingFlowPolicyStochastic

# ============================================================================
# Hyper-parameters
# ============================================================================

PRED_HORIZON     = 16
OBS_HORIZON      = 2
ACTION_HORIZON   = 8
OBS_DIM          = 2
ACTION_DIM       = 2

# Training
NUM_EPOCHS       = 500
BATCH_SIZE       = 256
LR               = 1e-4
WEIGHT_DECAY     = 1e-6
WARMUP_STEPS     = 200

# SFP
SIGMA_0          = 0.0
SIGMA_1          = 0.1
EQME_LAMBDA      = 0.01   # weight for EqM-E regulariser

# Diffusion Policy
NUM_DIFFUSION_ITERS = 100

# Evaluation
N_ROLLOUTS       = 50   # number of generated trajectories per method

DOWN_DIMS        = [128, 256, 512]   # smaller net for 2D toy

# Device selection — torchdyn NeuralODE works reliably on CPU/CUDA.
# MPS can be used for DP but may be unstable for ODE solvers.
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

print(f"Using device: {DEVICE}")

# ============================================================================
# Helper: build starting observation from the first frame of both modes
# ============================================================================

def make_start_obs(stats, device):
    """Return a normalized (OBS_HORIZON, OBS_DIM) obs starting at (0, 0)."""
    start_pos = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)  # (2, 2)
    nobs = normalize_data(start_pos, stats['obs'])
    return torch.from_numpy(nobs).to(device)


# ============================================================================
# Helper: run N rollouts of an SFP policy
# ============================================================================

def rollout_sfp(policy, nobs, n_rollouts, stats):
    """Return unnormalized trajectories shape (n_rollouts, PRED_HORIZON, 2)."""
    trajs = []
    for _ in range(n_rollouts):
        with torch.no_grad():
            naction = policy(nobs, num_actions=PRED_HORIZON)  # (1, T, 2)
        traj = unnormalize_data(
            naction.squeeze(0).cpu().numpy(), stats['action']
        )  # (T, 2)
        trajs.append(traj)
    return np.stack(trajs)  # (N, T, 2)


# ============================================================================
# Helper: run N rollouts of a DP policy
# ============================================================================

def rollout_dp(policy, nobs, n_rollouts, stats):
    """Return unnormalized trajectories shape (n_rollouts, PRED_HORIZON, 2)."""
    trajs = []
    for _ in range(n_rollouts):
        with torch.no_grad():
            naction = policy(nobs)  # (1, T, 2)
        traj = unnormalize_data(
            naction.squeeze(0).cpu().numpy(), stats['action']
        )  # (T, 2)
        trajs.append(traj)
    return np.stack(trajs)  # (N, T, 2)


# ============================================================================
# Training helpers
# ============================================================================

def make_sfp_policy(eqme_lambda=0.0):
    net = ConditionalUnet1D(
        input_dim=ACTION_DIM,
        global_cond_dim=OBS_DIM * OBS_HORIZON,
        down_dims=DOWN_DIMS,
        fc_timesteps=2,   # SFP processes [a, z] — a 2-step sequence
    ).to(DEVICE)
    policy = StreamingFlowPolicyStochastic(
        velocity_net=net,
        action_dim=ACTION_DIM,
        σ0=SIGMA_0,
        σ1=SIGMA_1,
        pred_horizon=PRED_HORIZON,
        eqme_lambda=eqme_lambda,
        device=DEVICE,
    )
    return policy, net


def train_sfp(policy, net, dataset, label):
    """Train an SFP (or SFP+EqM-E) model. Returns the EMA-averaged policy."""
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == 'cuda'),
    )

    ema = EMAModel(parameters=net.parameters(), power=0.75)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_scheduler = get_scheduler(
        'cosine',
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=len(dataloader) * NUM_EPOCHS,
    )

    with tqdm(range(NUM_EPOCHS), desc=label) as tglobal:
        for _ in tglobal:
            epoch_losses = []
            for nbatch in dataloader:
                loss = policy.Loss(nbatch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                ema.step(net.parameters())
                epoch_losses.append(loss.item())
            tglobal.set_postfix(loss=np.mean(epoch_losses))

    ema.copy_to(net.parameters())
    return policy


def make_dp_policy():
    net = ConditionalUnet1D(
        input_dim=ACTION_DIM,
        global_cond_dim=OBS_DIM * OBS_HORIZON,
        down_dims=DOWN_DIMS,
    ).to(DEVICE)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=NUM_DIFFUSION_ITERS,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )
    policy = DiffusionPolicy(
        noise_pred_net=net,
        num_diffusion_iters=NUM_DIFFUSION_ITERS,
        pred_horizon=PRED_HORIZON,
        action_dim=ACTION_DIM,
        device=DEVICE,
    )
    return policy, net, noise_scheduler


def train_dp(policy, net, noise_scheduler, dataset, label):
    """Train a Diffusion Policy model. Returns the EMA-averaged policy."""
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == 'cuda'),
    )

    ema = EMAModel(parameters=net.parameters(), power=0.75)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_scheduler = get_scheduler(
        'cosine',
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=len(dataloader) * NUM_EPOCHS,
    )

    with tqdm(range(NUM_EPOCHS), desc=label) as tglobal:
        for _ in tglobal:
            epoch_losses = []
            for nbatch in dataloader:
                nobs    = nbatch['obs'].to(DEVICE)     # (B, OBS_HORIZON, OBS_DIM)
                naction = nbatch['action'].to(DEVICE)  # (B, PRED_HORIZON, ACTION_DIM)
                B = nobs.shape[0]

                obs_cond = nobs[:, :OBS_HORIZON, :].flatten(start_dim=1)
                noise    = torch.randn_like(naction)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=DEVICE,
                ).long()
                noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
                noise_pred    = net(noisy_actions, timesteps, global_cond=obs_cond)
                loss          = nn.functional.mse_loss(noise_pred, noise)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                ema.step(net.parameters())
                epoch_losses.append(loss.item())
            tglobal.set_postfix(loss=np.mean(epoch_losses))

    ema.copy_to(net.parameters())
    return policy


# ============================================================================
# Main
# ============================================================================

def main():
    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    # DP dataset: raw {obs, action} (no transform_datum_fn)
    dp_dataset = ToyDataset2D(
        pred_horizon=PRED_HORIZON,
        obs_horizon=OBS_HORIZON,
        action_horizon=ACTION_HORIZON,
    )
    stats = dp_dataset.stats  # shared normalisation stats

    # SFP datasets: apply TransformTrainingDatum inside __getitem__
    sfp_policy_tmp, _ = make_sfp_policy(eqme_lambda=0.0)
    sfp_dataset = ToyDataset2D(
        pred_horizon=PRED_HORIZON,
        obs_horizon=OBS_HORIZON,
        action_horizon=ACTION_HORIZON,
        transform_datum_fn=sfp_policy_tmp.TransformTrainingDatum,
    )
    # Re-use the same data for SFP+EqM-E; TransformTrainingDatum is stateless
    # so we can share the dataset object.

    # ------------------------------------------------------------------
    # Train DP
    # ------------------------------------------------------------------
    print("\n=== Training DP baseline ===")
    dp_policy, dp_net, dp_scheduler = make_dp_policy()
    train_dp(dp_policy, dp_net, dp_scheduler, dp_dataset, label="DP")

    # ------------------------------------------------------------------
    # Train SFP (no EqM-E)
    # ------------------------------------------------------------------
    print("\n=== Training SFP (no EqM-E) ===")
    sfp_policy, sfp_net = make_sfp_policy(eqme_lambda=0.0)
    # Recreate dataset with the new policy's transform function
    sfp_dataset_plain = ToyDataset2D(
        pred_horizon=PRED_HORIZON,
        obs_horizon=OBS_HORIZON,
        action_horizon=ACTION_HORIZON,
        transform_datum_fn=sfp_policy.TransformTrainingDatum,
    )
    train_sfp(sfp_policy, sfp_net, sfp_dataset_plain, label="SFP")

    # ------------------------------------------------------------------
    # Train SFP + EqM-E
    # ------------------------------------------------------------------
    print(f"\n=== Training SFP + EqM-E (λ={EQME_LAMBDA}) ===")
    sfp_eqme_policy, sfp_eqme_net = make_sfp_policy(eqme_lambda=EQME_LAMBDA)
    sfp_dataset_eqme = ToyDataset2D(
        pred_horizon=PRED_HORIZON,
        obs_horizon=OBS_HORIZON,
        action_horizon=ACTION_HORIZON,
        transform_datum_fn=sfp_eqme_policy.TransformTrainingDatum,
    )
    train_sfp(sfp_eqme_policy, sfp_eqme_net, sfp_dataset_eqme, label="SFP+EqM-E")

    # ------------------------------------------------------------------
    # Evaluate: generate rollouts from the shared starting observation
    # ------------------------------------------------------------------
    nobs = make_start_obs(stats, DEVICE)  # (OBS_HORIZON, OBS_DIM)

    print("\nGenerating rollouts …")
    dp_trajs   = rollout_dp(dp_policy,   nobs, N_ROLLOUTS, stats)
    sfp_trajs  = rollout_sfp(sfp_policy, nobs, N_ROLLOUTS, stats)
    eqme_trajs = rollout_sfp(sfp_eqme_policy, nobs, N_ROLLOUTS, stats)

    # Ground-truth demo trajectories (unnormalized)
    demo_s     = generate_trajectory('s',     T=PRED_HORIZON)
    demo_anti  = generate_trajectory('anti_s', T=PRED_HORIZON)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True, sharey=True)
    fig.suptitle("2D Toy Comparison: Bimodal Coverage", fontsize=14, fontweight='bold')

    titles  = ["Ground Truth Demos", "DP baseline", "SFP", f"SFP + EqM-E (λ={EQME_LAMBDA})"]
    c_s     = '#2196F3'   # blue  — S-curve
    c_anti  = '#F44336'   # red   — anti-S-curve
    c_gen   = '#9E9E9E'   # grey  — generated

    for ax, title in zip(axes, titles):
        # Always draw the true demos as reference
        ax.plot(demo_s[:, 0],    demo_s[:, 1],    color=c_s,   lw=2, label='S-curve demo')
        ax.plot(demo_anti[:, 0], demo_anti[:, 1], color=c_anti, lw=2, label='Anti-S demo')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('x')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # DP generated
    for traj in dp_trajs:
        axes[1].plot(traj[:, 0], traj[:, 1], color=c_gen, alpha=0.3, lw=0.8)

    # SFP generated
    for traj in sfp_trajs:
        axes[2].plot(traj[:, 0], traj[:, 1], color=c_gen, alpha=0.3, lw=0.8)

    # SFP + EqM-E generated
    for traj in eqme_trajs:
        axes[3].plot(traj[:, 0], traj[:, 1], color=c_gen, alpha=0.3, lw=0.8)

    axes[0].set_ylabel('y')

    # Legend
    legend_elements = [
        mpatches.Patch(color=c_s,   label='S-curve demo'),
        mpatches.Patch(color=c_anti, label='Anti-S demo'),
        mpatches.Patch(color=c_gen,  label=f'Generated ({N_ROLLOUTS} samples)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = 'toy_2d_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot → {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
