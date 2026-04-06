"""
2D bimodal toy dataset for SFP experiments.

Two demonstration modes:
  Mode A (S-curve):      x = t, y =  0.8 * sin(π*(2t-1))   [t ∈ [0, 1]]
  Mode B (anti-S-curve): x = t, y = -0.8 * sin(π*(2t-1))   [t ∈ [0, 1]]

Both curves start at (0, 0) and end at (1, 0), but trace different paths,
creating a bimodal distribution ideal for testing multi-modal imitation learning.

The __getitem__ format matches PushTStateDataset:
    'obs'    (OBS_HORIZON, 2)   – position history window
    'action' (PRED_HORIZON, 2)  – future positions to predict
"""
from typing import Callable, Dict, Optional

import numpy as np
import torch

from streaming_flow_policy.pusht.dp_state_notebook.dataset import (
    create_sample_indices,
    get_data_stats,
    normalize_data,
    sample_sequence,
    unnormalize_data,
)


def generate_trajectory(mode: str, T: int = 60) -> np.ndarray:
    """Generate a clean 2D S-curve or anti-S-curve trajectory.

    Args:
        mode: 's' for S-curve, 'anti_s' for reverse S-curve.
        T: number of timesteps.

    Returns:
        np.ndarray shape (T, 2): trajectory positions (x, y).
    """
    t = np.linspace(0, 1, T, dtype=np.float32)
    x = t
    sign = 1.0 if mode == 's' else -1.0
    y = sign * 0.8 * np.sin(np.pi * (2.0 * t - 1.0)).astype(np.float32)
    return np.stack([x, y], axis=1)  # (T, 2)


class ToyDataset2D(torch.utils.data.Dataset):
    """2D bimodal toy dataset compatible with StreamingFlowPolicyStochastic.

    Parameters
    ----------
    pred_horizon:
        Number of future actions to predict (sequence window length).
    obs_horizon:
        Number of past observations to condition on (must be 2 for SFP).
    action_horizon:
        Number of executed actions per step (used for index padding only).
    n_episodes_per_mode:
        How many independent episodes to generate per mode.
    steps_per_episode:
        Trajectory length of each episode (should be > pred_horizon).
    noise_std:
        Standard deviation of Gaussian noise added to each episode to
        diversify demonstrations.
    transform_datum_fn:
        Optional function applied to each raw {'obs', 'action'} sample,
        e.g. StreamingFlowPolicyStochastic.TransformTrainingDatum.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        pred_horizon: int = 16,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        n_episodes_per_mode: int = 100,
        steps_per_episode: int = 60,
        noise_std: float = 0.01,
        transform_datum_fn: Optional[Callable] = None,
        seed: int = 42,
    ):
        assert obs_horizon == 2, "SFP currently requires obs_horizon == 2"
        assert steps_per_episode > pred_horizon, (
            "steps_per_episode must exceed pred_horizon to allow at least one window."
        )

        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.transform_datum_fn = transform_datum_fn

        rng = np.random.default_rng(seed)

        all_obs: list = []
        all_actions: list = []
        episode_ends: list = []
        cursor = 0

        for mode in ('s', 'anti_s'):
            for _ in range(n_episodes_per_mode):
                traj = generate_trajectory(mode, steps_per_episode)  # (T, 2)

                # Small per-episode noise keeps demonstrations diverse while
                # preserving the overall bimodal shape.
                traj = traj + rng.normal(0.0, noise_std, traj.shape).astype(np.float32)

                # obs[i]    = position at step i
                # action[i] = position at step i+1  (next-obs as action)
                obs_ep = traj                                           # (T, 2)
                action_ep = np.concatenate(
                    [traj[1:], traj[[-1]]], axis=0
                ).astype(np.float32)                                    # (T, 2)

                all_obs.append(obs_ep)
                all_actions.append(action_ep)
                cursor += steps_per_episode
                episode_ends.append(cursor)

        obs_arr = np.concatenate(all_obs, axis=0)       # (N_total, 2)
        action_arr = np.concatenate(all_actions, axis=0)  # (N_total, 2)
        episode_ends_arr = np.array(episode_ends, dtype=np.int64)

        train_data: Dict[str, np.ndarray] = {
            'obs': obs_arr,
            'action': action_arr,
        }

        # Normalize each key independently to [-1, 1].
        stats: Dict = {}
        normalized: Dict[str, np.ndarray] = {}
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized[key] = normalize_data(data, stats[key])

        indices = create_sample_indices(
            episode_ends=episode_ends_arr,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        self.stats = stats
        self.normalized_train_data = normalized
        self.indices = indices

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        buf_start, buf_end, samp_start, samp_end = self.indices[idx]

        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buf_start,
            buffer_end_idx=buf_end,
            sample_start_idx=samp_start,
            sample_end_idx=samp_end,
        )
        # Keep only the obs_horizon most recent observations.
        nsample['obs'] = nsample['obs'][:self.obs_horizon, :]  # (OBS_HORIZON, 2)

        if self.transform_datum_fn is not None:
            nsample = self.transform_datum_fn(nsample)

        return nsample
