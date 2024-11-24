import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import wandb
import sys, os

sys.path.append('/home/sunsh16e/flow-policy-1/')
from flow_policy.pusht.dataset import PushTStateDatasetWithNextObsAsAction
from flow_policy.pusht.dp_state_notebook.network import ConditionalUnet1D
from flow_policy.pusht.sfpd import StreamingFlowPolicyDeterministic

# Constants
OBS_HORIZON = 2
ACTION_HORIZON = 8
OBS_DIM = 5
ACTION_DIM = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Train PushT Streaming Flow Policy")
    parser.add_argument("--pred_horizon", type=int, default=17, help="Prediction horizon")
    parser.add_argument("--seed", type=int, default=16, help="Random seed")
    parser.add_argument("--sin_embedding_scale", type=float, default=100, help="Sine embedding scale")
    parser.add_argument("--sigma", type=float, default=0.2, help="Sigma value")
    parser.add_argument("--num_epochs", type=int, default=800, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--save_epoch", type=int, default=100, help="Save every N epochs")
    parser.add_argument("--use_wandb", action="store_true", help="Use WandB for logging")
    parser.add_argument("--model_save_dir", type=str, default="/home/sunsh16e/flow-policy-1/models", help="Model save dir")
    return parser.parse_args()


def main():
    args = parse_args()

    # Assign arguments to local variables
    pred_horizon = args.pred_horizon
    seed = args.seed
    sin_embedding_scale = args.sin_embedding_scale
    sigma = args.sigma
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    save_epoch = args.save_epoch
    use_wandb = args.use_wandb
    model_save_dir = args.model_save_dir
    
    lr_str = str(round(-math.log10(lr)))
    hyperpapram_str = f"sfpd_sigma_{sigma}_epoch_{num_epochs}_lr_{lr_str}"
    save_path = os.path.join(model_save_dir, f"{hyperpapram_str}.pth")

    # Create network object
    velocity_net = ConditionalUnet1D(
        input_dim=ACTION_DIM,
        global_cond_dim=OBS_DIM * OBS_HORIZON,
        fc_timesteps=1,
        sin_embedding_scale=sin_embedding_scale,
    )

    # Device transfer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    velocity_net.to(device)

    policy = StreamingFlowPolicyDeterministic(
        velocity_net=velocity_net,
        action_dim=ACTION_DIM,
        pred_horizon=pred_horizon,
        sigma=sigma,
        device=device,
    )

    # Create dataset and dataloader
    dataset = PushTStateDatasetWithNextObsAsAction(
        pred_horizon=pred_horizon,
        obs_horizon=OBS_HORIZON,
        action_horizon=ACTION_HORIZON,
        transform_datum_fn=policy.TransformTrainingDatum,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )

    # Exponential Moving Average
    ema = EMAModel(parameters=velocity_net.parameters(), power=0.75)

    # Optimizer
    optimizer = torch.optim.AdamW(
        params=velocity_net.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs,
    )

    # Initialize WandB
    if use_wandb:
        wandb.init(
        project="pushT-train",
        config={
            "tag": hyperpapram_str,
            "x_t_sigma": sigma,
            "learning_rate": lr,
            "num_epochs": num_epochs,
            "weight_decay": weight_decay,
            "save_epoch": save_epoch,
            "model_save_path": save_path,
            "optimizer": "adam",
        }
    )

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Training loop
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = []
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    loss = policy.Loss(nbatch)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    ema.step(velocity_net.parameters())

                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

            avg_loss = np.mean(epoch_loss)
            tglobal.set_postfix(loss=avg_loss)
            wandb.log({"loss": avg_loss, "epoch": epoch_idx+1})


    # Save model
    ema_velocity_net = policy.velocity_net
    ema.copy_to(ema_velocity_net.parameters())
    torch.save(policy.state_dict(), save_path)
    print(f"Saved model to {save_path}.")


if __name__ == "__main__":
    main()