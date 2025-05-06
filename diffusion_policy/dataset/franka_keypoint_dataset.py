import random
from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer, get_identity_normalizer_from_stat, get_identity_normalizer

class FrankaPickKeypointDataset(BaseImageDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            use_3d_keypoint=True,
            all_identity_normalizer=True,
            kp_augmentation=False,
            ):

        super().__init__()
        print('------------------- Dataset -------------------------')
        print(f'Loading FrankaImageDataset from {zarr_path}')
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['keypoints_ij', 'keypoints_xyz', 'keypoints_visibility', 'state', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_3d_keypoint = use_3d_keypoint
        self.all_identity_normalizer = all_identity_normalizer
        self.kp_augmentation = kp_augmentation

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # TODO identity okay?
        normalizer['keypoint'] = get_identity_normalizer()
        if self.all_identity_normalizer:
            normalizer['action'] = get_identity_normalizer()
            normalizer['agent_pos'] = get_identity_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'].astype(np.float32)
        action = sample['action'].astype(np.float32)
        if self.use_3d_keypoint:
            keypoint_loc = sample['keypoints_xyz'].astype(np.float32) # T, N, 3
        else:
            keypoint_loc = sample['keypoints_ij'].astype(np.float32) # T, N, 2
        keypoint_visibility = sample['keypoints_visibility'].astype(np.float32)[..., np.newaxis] # T, N

        # Data augmentation
        idx_shuffle = np.arange(keypoint_loc.shape[1])
        random.shuffle(idx_shuffle)
        keypoint_loc = keypoint_loc[:, idx_shuffle] # T, N, 2
        keypoint_visibility = keypoint_visibility[:, idx_shuffle]
        noise_keypoint_loc = np.random.rand(*keypoint_loc.shape) * 0.05 - 0.025
        keypoint_loc = keypoint_loc + noise_keypoint_loc

        if self.kp_augmentation:
            assert self.use_3d_keypoint
            augment_range = np.array([0.15, 0.15, .05]) * 2
            random_locshift_xyz = (np.random.rand(3) - 0.5) * augment_range
            keypoint_loc += random_locshift_xyz
            agent_pos[:, :3] += random_locshift_xyz
            action[:, :3] += random_locshift_xyz

        data = {
            'obs': {
                'keypoint': np.concatenate((keypoint_visibility, keypoint_loc), axis=-1), # T, 3 or T, 4
                'agent_pos': agent_pos, # T, 10
            },
            'action': action # T, 10
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    zarr_path = './data/franka_pick_kp3d_v3.zarr'
    dataset = FrankaPickKeypointDataset(zarr_path, horizon=16)

    for i in range(1000):
        # print(dataset.__getitem__(i)['action'][-1, -1],)
        step_data = dataset.__getitem__(i)
        action_data = step_action['action']
        action_data_prev = action_data[:-1]
        action_data_next = action_data[1:]
        print(action_data_next - action_data_prev)
        from IPython import embed; embed()
        print('-' * 20)

    from matplotlib import pyplot as plt
    normalizer = dataset.get_normalizer()
    nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    diff = np.diff(nactions, axis=0)
    dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
    plt.hist(dists)
    plt.show()
    plt.close()


if __name__ == '__main__':
    test()