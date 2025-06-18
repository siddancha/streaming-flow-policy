from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer, get_identity_normalizer

class FrankaPickImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            two_images=False,
            ):

        super().__init__()
        print(f'Loading FrankaImageDataset from {zarr_path}')
        if two_images:
            keylist = ['img', 'img2', 'state', 'action']
        else:
            keylist = ['img', 'state', 'action']
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=keylist)
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
        self.two_images = two_images

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
        # normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        if self.two_images: normalizer['image2'] = get_image_range_normalizer()
        # if self.normalize:
        #     stat = array_to_stats(self.replay_buffer['action'])
        #     normalizer['action'] = normalizer_from_stat(stat)
        
        normalizer['action'] = get_identity_normalizer()
        normalizer['agent_pos'] = get_identity_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'].astype(np.float32)
        image = np.moveaxis(sample['img'],-1,1)/255.
        if self.two_images: image2 = np.moveaxis(sample['img2'],-1,1)/255.

        if self.two_images:
            obs = {
                'image': image, # T, 3, 256, 256
                'image2': image2, # T, 3, 256, 256
                'agent_pos': agent_pos, # T, 10 
            }
        else:
            obs = {
                'image': image, # T, 3, 256, 256
                'agent_pos': agent_pos, # T, 10 
            }

        data = {
            'obs': obs,
            'action': sample['action'].astype(np.float32) # T, 10
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    zarr_path = './data/pushy.zarr'
    dataset = FrankaPickImageDataset(zarr_path, horizon=16)
    for data in dataset:
        print('image obs shape', data['obs']['image'].shape)

    # for i in range(1000):
    #     print(dataset.__getitem__(i)['action'][-1, -1],)

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
