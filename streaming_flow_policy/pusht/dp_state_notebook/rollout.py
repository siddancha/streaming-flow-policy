from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from torch import Tensor
import collections
from tqdm.auto import tqdm


from streaming_flow_policy.pusht.dp_state_notebook.all import (
    normalize_data, unnormalize_data, Policy, PushTEnv
)

def Rollout(
        env: PushTEnv,
        policy: Policy,
        stats: Dict,
        max_steps: int = 200,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        device: str = 'cuda',
        policy_kwargs: Optional[Dict] = None,
    ) -> Tuple[float, List[np.ndarray]]:

    # get first observation
    obs, info = env.reset()

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)

    # save visualization and rewards
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0

    policy_kwargs = policy_kwargs or {}
    policy_kwargs["num_actions"] = 1 + action_horizon

    with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
        while not done:
            B = 1
            # stack the last obs_horizon (2) number of observations
            obs_seq = np.stack(obs_deque)
            # normalize observation
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            # device transfer
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

            # infer action
            with torch.no_grad():
                # reshape observation to (B,obs_horizon*obs_dim)
                naction: Tensor = policy(nobs, **policy_kwargs)

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats['action'])

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, _, info = env.step(action[i])
                # save observations
                obs_deque.append(obs)
                # and reward/vis
                rewards.append(reward)
                imgs.append(env.render(mode='rgb_array'))

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    break

    score = max(rewards)
    return score, imgs
