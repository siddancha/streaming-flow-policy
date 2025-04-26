import cv2
import zarr
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def center_crop(rgb_im):
    im_h, im_w = rgb_im.shape[:2]
    min_hw = min(im_h, im_w)
    cropped_image = rgb_im[(im_h - min_hw) // 2: (im_h - min_hw) // 2 + min_hw, (im_w - min_hw) // 2: (im_w - min_hw) // 2 + min_hw]
    return cropped_image


def get_state(record):
    rotation = record['ee_pose'][:3, :2].T.reshape(-1)
    translation = record['ee_pose'][:3, 3]
    return np.concatenate((translation, rotation, record['gripper_state'].reshape(-1)), axis=0)


def preprocess_rgb(rgb_im, crop_size=256):
    return cv2.resize(rgb_im, (crop_size, crop_size))


def main():
    data_folder = '/data3/xiaolinf/franka_pick/'
    zarr_save_path = 'data/franka_pick.zarr'
    pkl_files = glob.glob(data_folder + 'pick_apple**.pkl')
    all_rgb, all_state, all_action, episode_end = [], [], [], []
    for pkl_file in tqdm(pkl_files):
        print(f'Processing {pkl_file}')
        raw_data = np.load(pkl_file, allow_pickle=True)
        onetraj_rgb, onetraj_state = [], []

        for step in raw_data['trajectory']:
            onetraj_rgb.append(preprocess_rgb(center_crop(step['mount2']['rgb'])))

        for step in raw_data['trajectory']:
            onetraj_state.append(get_state(step['robot']))

        # TODO
        onetraj_action = onetraj_state[1:]
        onetraj_action.append(np.zeros(10))

        all_rgb.extend(onetraj_rgb)
        all_state.extend(onetraj_state)
        all_action.extend(onetraj_action)
        episode_end.append(len(all_rgb)) # TODO: double check

    store = zarr.DirectoryStore(zarr_save_path)
    root = zarr.group(store=store, overwrite=True)
    # Create the 'data' group
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')
    data_group.create_dataset('action', data=np.asarray(all_action).astype(np.float32))
    data_group.create_dataset('img', data=np.asarray(all_rgb))
    data_group.create_dataset('state', data=np.asarray(all_state).astype(np.float32))
    meta_group.create_dataset('episode_ends', data=np.asarray(episode_end, dtype=np.int64))

    # for im in all_rgb[::100]:
    #     plt.imshow(im)
    #     plt.show()
    #     plt.close()

    print(f'save to {zarr_save_path}')


if __name__ == '__main__':
    main()
