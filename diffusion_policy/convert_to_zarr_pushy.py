import os
import cv2
import zarr
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# from streaming_flow_policy.franka.keypoint_tracker import Tracker
# from beepp.perception import initialize_perception_interface, RGBDObservation

def center_crop(rgb_im):
    im_h, im_w = rgb_im.shape[:2]
    min_hw = min(im_h, im_w)
    cropped_image = rgb_im[(im_h - min_hw) // 2: (im_h - min_hw) // 2 + min_hw, (im_w - min_hw) // 2: (im_w - min_hw) // 2 + min_hw]
    return cropped_image


def get_state(record):
    ## remove gripper bc always closed in this case
    rotation = record['ee_pose'][:3, :2].T.reshape(-1)
    translation = record['ee_pose'][:3, 3]
    return np.concatenate((translation, rotation, record['gripper_state'].reshape(-1)), axis=0)


def preprocess_rgb(rgb_im, crop_size=256):
    return cv2.resize(center_crop(rgb_im), (crop_size, crop_size))


def convert_scale(points_yx, original_hw, crop_size):
    y, x = points_yx[:, 0], points_yx[:, 1]
    im_h, im_w = original_hw
    min_hw = min(im_h, im_w)

    offset_y = (im_h - min_hw) // 2
    offset_x = (im_w - min_hw) // 2

    x_cropped = x - offset_x
    y_cropped = y - offset_y

    scale = crop_size / min_hw
    x_resized = x_cropped * scale
    y_resized = y_cropped * scale
    return np.stack([y_resized, x_resized], axis=-1).astype(np.int32)


def get_validstep_ids(traj_qpos, valid_delta=8e-3) -> list:
    """
    Truncate steps that are too close to each other.
    Args:
        traj_qpos:
        valid_delta:
    Returns:
        Array of valid entry ids
    """
    truncated_ids = [0]
    for k in range(traj_qpos.shape[0]):
        delta_dist = np.linalg.norm(traj_qpos[k] - traj_qpos[truncated_ids[-1]])
        if delta_dist < valid_delta:
            continue
        truncated_ids.append(k)
    return truncated_ids


def main():
    # For push Y:
    ## chanded the camera mount 2 -> mount 1
    # changed the folder, save_path
    two_images = True #use two cameras or not
    data_folders = ['data_unprocessed/pushy_v2']
    zarr_save_path = '/home/sfp/streaming-flow-policy/data/pushy_v3.zarr'
    crop_size = 256
    pkl_files = []
    for data_folder in data_folders:
        pkl_files += glob.glob(data_folder + '/**.pkl')


    all_rgb, all_rgb2, all_state, all_action, episode_end = [], [], [], [], []



    for pkl_filename in tqdm(pkl_files):
        print(f'Processing {pkl_filename}')
        raw_data = np.load(pkl_filename, allow_pickle=True)
        print(f'Data loaded. {len(raw_data)} steps')
        onetraj_rgb, onetraj_rgb2, onetraj_state_v1 = [], [], []

        valid_trajectory = raw_data
        print(f'{len(valid_trajectory)} valid steps after truncating')

        for step in valid_trajectory:
            onetraj_rgb.append(preprocess_rgb(step['mount1']['rgb_im'], crop_size))
            if two_images:
                onetraj_rgb2.append(preprocess_rgb(step['robot1_hand']['rgb_im'], crop_size))
                # concat_img = np.concatenate([preprocess_rgb(step['mount1']['rgb_im'], crop_size), 
                #                         preprocess_rgb(step['robot1_hand']['rgb_im'], crop_size)], axis=-1)
                # onetraj_rgb.append(concat_img)
            
            onetraj_state_v1.append(get_state(step))


        onetrajs_state = [onetraj_state_v1]

        print(f'{len(onetrajs_state)} variants')
        for onetraj_state in onetrajs_state:
            # TODO: use next state for now. can be OSC cmd action.
            onetraj_action = onetraj_state[1:]
            onetraj_action.append(onetraj_action[-1])

            all_rgb.extend(onetraj_rgb)
            if two_images: all_rgb2.extend(onetraj_rgb2)
            all_state.extend(onetraj_state)
            all_action.extend(onetraj_action)
            episode_end.append(len(all_rgb))

    store = zarr.DirectoryStore(zarr_save_path)
    root = zarr.group(store=store, overwrite=True)
    # Create the 'data' group
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')
    data_group.create_dataset('action', data=np.asarray(all_action).astype(np.float32))
    data_group.create_dataset('img', data=np.asarray(all_rgb))
    if two_images: data_group.create_dataset('img2', data=np.asarray(all_rgb2))
    data_group.create_dataset('state', data=np.asarray(all_state).astype(np.float32))
    meta_group.create_dataset('episode_ends', data=np.asarray(episode_end, dtype=np.int64))

    print(len(all_rgb))
    # for im in all_rgb[::100]:
    #     plt.imshow(im)
    #     plt.show()
    #     plt.close() 

    print(f'save to {zarr_save_path}')


if __name__ == '__main__':
    main()
