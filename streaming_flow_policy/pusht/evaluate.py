import argparse
import numpy as np
import torch
from pathlib import Path
# import wandb

from streaming_flow_policy.pusht.dataset import PushTStateDatasetWithNextObsAsAction
from streaming_flow_policy.pusht.dp_state_notebook.all import (
    ConditionalUnet1D, PushTEnv, Rollout,
)
from streaming_flow_policy.pusht.sfpd import StreamingFlowPolicyDeterministic
from streaming_flow_policy.pusht.sfps import StreamingFlowPolicyStochastic


assert torch.cuda.is_available(), "CUDA is not available"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Testing script for flow-policy model.")
    parser.add_argument("--model-type", type=str, required=True, help="Model used for testing: sfpd or sfps")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to EMA checkpoint file")
    parser.add_argument("--save-dir", type=str, default="./eval", help="Directory to save results")
    
    parser.add_argument("--integration-steps-per-action", type=int, default=1, help="Integration time steps per action")
    parser.add_argument("--num-tests", type=int, default=100, help="Number of tests to run")
    parser.add_argument("--name", type=str, default="newbatch", help="Name for this experiment")
    parser.add_argument("--kp", type=float, default=500, help="Position gain")
    parser.add_argument("--kv", type=float, default=20, help="Velocity gain")
    parser.add_argument("--seed", type=int, default=16, help="Random seed")
    parser.add_argument("--env-start-seed", type=int, default=500, help="Start seed for environment")
    parser.add_argument("--max-rollout-steps", type=int, default=200, help="Maximum steps in each test")
    return parser.parse_args()

def pretty_print_args(args):
    # Get the longest argument name for proper spacing
    max_arg_length = max(len(arg) for arg in vars(args))
    width = max(60, max_arg_length + 40)  # Ensure minimum width of 80 chars
    name_width = max_arg_length + 2  # Width for argument names section

    # Create box characters
    top_line = "╔" + "═" * (width - 2) + "╗"
    middle_line = "╠" + "═" * (width - 2) + "╣"
    bottom_line = "╚" + "═" * (width - 2) + "╝"
    middle = ":"
    side = "║"

    print("\n" + top_line)
    title = " Running with arguments:"
    print(f"{side}{title:<{width - 2}}{side}")
    print(middle_line)

    for arg, value in vars(args).items():
        print(f"{side}{arg:>{name_width-1}} {middle} {value:<{width - name_width - 4}}{side}")

    print(bottom_line + "\n")


def main():
    args = parse_args()
    pretty_print_args(args)

    # Derived parameters
    save_dir = Path(args.save_dir)
    score_save_path = save_dir / f'{args.name}_score.pt'
    traj_save_path = save_dir / f'{args.name}_traj.pt'
    action_save_path = save_dir / f'{args.name}_action.pt'

    # Parameters
    obs_horizon = 2
    action_horizon = 8
    obs_dim = 5
    action_dim = 2

    # Load policy weights
    if args.model_type == "sfpd":
        velocity_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim*obs_horizon,
            fc_timesteps=1,
        )
        policy = StreamingFlowPolicyDeterministic(
            velocity_net=velocity_net,
            action_dim=action_dim,
            device='cuda',
        )
        state_dict = torch.load(args.ckpt_path, map_location='cuda')
        policy.load_state_dict(state_dict)
    elif args.model_type == "sfps":
        velocity_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim*obs_horizon,
            fc_timesteps=2,
        )
        policy = StreamingFlowPolicyStochastic(
            velocity_net=velocity_net,
            action_dim=action_dim,
            device='cuda',
        )
        state_dict = torch.load(args.ckpt_path, map_location='cuda')
        policy.load_state_dict(state_dict)
    else:
        raise ValueError(f"Invalid model type {args.model_type}, only accept sfpd or sfps")

    policy.cuda()
    print('Pretrained weights loaded.')

    # Create dataset for stats
    dataset = PushTStateDatasetWithNextObsAsAction(
        pred_horizon=policy.pred_horizon.item(),
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
    )

    # Initialize lists to store results
    score_list = []
    all_action_list = []
    all_state_list = []

    # Initialize WandB
    # wandb.init(
    #     project="pushT-test",
    #     config={
    #         "name": name,
    #         "unit_int_steps": integration_steps_per_action,
    #         "num_tests": num_tests,
    #         "ema_ckpt_path": ckpt_path,
    #         "save_dir": save_dir,
    #         "k_p_scale": k_p,
    #         "k_v_scale": k_v,
    #         "seed": seed,
    #         "env_start_seed": env_start_seed,
    #         "max_steps": max_steps,
    #     }
    # )

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create environment
    env = PushTEnv(k_p=args.kp, k_v=args.kv)

    for t_ix in range(args.num_tests):
        env_seed = args.env_start_seed + t_ix
        print(f"Environment seed: {env_seed}")
        env.seed(env_seed)

        # Run rollout
        score, imgs = Rollout(
            env=env,
            policy=policy,
            stats=dataset.stats,
            max_steps=args.max_rollout_steps,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            device='cuda',
            policy_kwargs={
                'integration_steps_per_action': args.integration_steps_per_action
            }
        )

        # Get state information from the environment
        # state_all = env._get_state()

        # Log results
        # wandb.log({
        #     "t_ix": t_ix, 
        #     "score": score,
        #     "init_state_gx": state_all[0],
        #     "init_state_gy": state_all[1],
        #     "init_state_bx": state_all[2],
        #     "init_state_by": state_all[3],
        #     "init_state_theta": state_all[4],
        # })

        print(f"Score: {score}")
        score_list.append(score)
        # all_state_list.append(state_all)

    # Save results
    torch.save(score_list, score_save_path)
    print('Average score:', sum(score_list) / len(score_list))
    torch.save(all_state_list, traj_save_path)

    # wandb.finish()

if __name__ == "__main__":
    main()