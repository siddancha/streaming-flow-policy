"""
All functions in this file have been copied from the Diffusion Policy repo, in
particular, this notebook (diffusion_policy_state_pusht_demo.ipynb):
https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B
"""
from typing import Dict
import numpy as np
import torch
import gdown
import os
import zarr

def create_sample_indices(
        episode_ends: np.ndarray,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0
    ) -> np.ndarray:
    """
    Args:
        episode_ends (np.ndarray, shape=(E,), dtype=int):
            Marks one-past the last index for each episode.
        sequence_length (int): Length of the sequences to sample.
        pad_before (int): Number of timesteps to pad before the sequence.
        pad_after (int): Number of timesteps to pad after the sequence.

    Returns:
        np.ndarray, shape=(N, 4): Indices to sample sequences from the dataset.
            The second axis corresponds to (buffer_start_idx, buffer_end_idx,
            sample_start_idx, sample_end_idx).
    """
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices

def sample_sequence(
        train_data: Dict[str, np.ndarray],
        sequence_length: int,
        buffer_start_idx: int,
        buffer_end_idx: int,
        sample_start_idx: int,
        sample_end_idx: int
    ) -> Dict[str, np.ndarray]:
    """
    Args:
        train_data (Dict[str, np.ndarray]): Data to sample from. Keys are
            "action" and "obs". Values are np.ndarrays of size (N, K) where N
            is the number of timesteps in the dataset and K is the dimension of
            the data.
        sequence_length (int): Length of the sequence to sample.
        buffer_start_idx (int): Start index of the buffer.
        buffer_end_idx (int): End index of the buffer.
        sample_start_idx (int): Start index of the sample.
        sample_end_idx (int): End index of the sample.

    Returns:
        Dict[str, np.ndarray]: A dictionary with the same keys as
            `train_data` (e.g. 'action', 'obs'). Values are
            (np.ndarray, shape=(PRED_HORIZON, K)) where K is the dimension of
            the data (e.g. ACTION_DIM, OBS_DIM).
    """
    result: Dict[str, np.ndarray] = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]  # (S, K)
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)  # (PRED_HORIZON, K)

            # Repeat the first timestep
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]

            # Repeat the last timestep
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]

            # Fill in the middle
            data[sample_start_idx:sample_end_idx] = sample

        result[key] = data  # (PRED_HORIZON, K)
    return result

# normalize data
def get_data_stats(data: np.ndarray):
    """
    Args:
        data (np.ndarray, shape=(..., K)): Data to compute statistics over.
    """
    data = data.reshape(-1, data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data: np.ndarray, stats: Dict[str, np.ndarray]):
    """
    Args:
        data (np.ndarray, shape=(..., K)): Data to normalize.
        stats (Dict[str, np.ndarray]): Statistics to use for normalization.
    """
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata: np.ndarray, stats: Dict[str, np.ndarray]):
    """
    Args:
        ndata (np.ndarray, shape=(..., K)): Normalized data to un0normalize.
        stats (Dict[str, np.ndarray]): Statistics to use for un-normalization.
    """
    # unnormalize to [0, 1]
    ndata = (ndata + 1) / 2
    # unnormalize to original range
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


class PushTStateDataset(torch.utils.data.Dataset):
    """
    |o|o|                             observations:     2
    | |a|a|a|a|a|a|a|a|               actions executed: 8
    |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
    """
    def __init__(
            self,
            pred_horizon: int,
            obs_horizon: int,
            action_horizon: int
        ):

        # read from zarr dataset
        dataset_root: zarr.Group = self.GetDatasetRoot()

        # All demonstration episodes are concatenated in the first dimension N.
        train_data = self.LoadTrainData(dataset_root)

        # Marks one-past the last index for each episode.
        episode_ends = dataset_root['meta']['episode_ends'][:]  # (E,)

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics
        # maps strings (e.g. 'action', 'obs') to dicts mapping strings
        # ('min', 'max') to np.ndarrays of size (K,) where K is ACTION_DIM,
        # OBS_DIM, etc.
        stats: Dict[str, Dict[str, np.ndarray]] = dict()

        # compute normalized data to [-1, 1]
        # maps strings (e.g. 'action', 'obs') to np.ndarrays of size (K,) where
        # K is ACTION_DIM, OBS_DIM, etc.
        normalized_train_data: Dict[str, np.ndarray] = dict()

        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    @staticmethod
    def GetDatasetRoot() -> zarr.Group:
        """Load dataset from disk or Google Drive"""
        dataset_path = "pusht_cchi_v7_replay.zarr.zip"
        if not os.path.isfile(dataset_path):
            id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
            gdown.download(id=id, output=dataset_path, quiet=False)
        return zarr.open(dataset_path, 'r')

    def LoadTrainData(self, dataset_root: zarr.Group) -> Dict[str, np.ndarray]:
        """
        Args:
            dataset_root (zarr.Group): Root of the zarr dataset.

        Returns:
            Dict[str, np.ndarray]: A dictionary with keys 'action' and 'obs'.
                Values are np.ndarrays of size (N, K) where N is the number of
                timesteps in the dataset and K is the dimension of the data
                (e.g. ACTION_DIM, OBS_DIM).
        """
        # All demonstration episodes are concatenated in the first dimension N
        return {
            'action': dataset_root['data']['action'][:],  # (N, ACTION_DIM)
            'obs': dataset_root['data']['state'][:]  # (N, OBS_DIM)
        }

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx: int):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get normalized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )  # each tensor is of shape (PRED_HORIZON, K)

        # discard unused observations
        nsample['obs'] = nsample['obs'][:self.obs_horizon,:]
        return nsample
