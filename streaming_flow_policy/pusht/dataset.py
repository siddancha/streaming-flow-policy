from typing import Dict, Optional, Callable
import numpy as np
import zarr

from .dp_state_notebook.dataset import PushTStateDataset

class PushTStateDatasetWithNextObsAsAction (PushTStateDataset):
    """
    A dataset of where gripper observations act as actions.
    """
    def __init__(self, *args, transform_datum_fn: Optional[Callable] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform_datum_fn = transform_datum_fn

    def LoadTrainData(self, dataset_root: zarr.Group) -> Dict[str, np.ndarray]:
        """
        Overloads superclass method to set actions as the observations of the
        next timestep.

        Args:
            dataset_root (zarr.Group): Root of the zarr dataset.

        Returns:
            Dict[str, np.ndarray]: A dictionary with keys 'action' and 'obs'.
                Values are np.ndarrays of size (N, K) where N is the number of
                timesteps in the dataset and K is the dimension of the data
                (e.g. ACTION_DIM, OBS_DIM).
        """
        obs = dataset_root['data']['state'][:]  # (N, OBS_DIM)
        episode_ends = dataset_root['meta']['episode_ends'][:]  # (E,)
        assert episode_ends[-1] == obs.shape[0]  # check that last episode corresponds to last timestep

        # Iterate through the episode ends to slice and shift the tensor
        actions = []
        start_idx = 0
        for end_idx in episode_ends:
            obs_ep = obs[start_idx:end_idx]  # (T, OBS_DIM)
            obs_gripper_ep = obs_ep[:, :2]  # (T, 2)
            actions_ep = np.concatenate((
                obs_gripper_ep[1:],  # (T-1, 2)
                obs_gripper_ep[[-1]],  # (1, 2)
            ))  # (T, 2)
            actions.append(actions_ep)
            start_idx = end_idx

        # Combine the shifted episodes back into a single tensor if needed
        actions = np.concatenate(actions, axis=0)

        return {
            'action': actions,  # (N, ACTION_DIM)
            'obs': obs  # (N, OBS_DIM)
        }

    def __getitem__(self, *args, **kwargs):
        datum = super().__getitem__(*args, **kwargs)
        if self.transform_datum_fn is not None:
            datum = self.transform_datum_fn(datum)
        return datum
