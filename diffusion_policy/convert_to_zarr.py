import os
import cv2
import zarr
import glob
import numpy as np
from tqdm import tqdm
from streaming_flow_policy.franka.keypoint_tracker import Tracker
from beepp.perception import initialize_perception_interface, RGBDObservation

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


def main():
    data_folder = '/data3/xiaolinf/franka_pick/'
    zarr_save_path = 'data/franka_pick_kp.zarr'
    target_keypoints_num = 10
    crop_size = 256
    pkl_files = glob.glob(data_folder + 'pick_apple**.pkl')
    all_rgb, all_state, all_action, episode_end = [], [], [], []
    all_keypoints_xy, all_keypoints_visibility = [], []

    perception_interface = initialize_perception_interface()
    keypoint_tracker = Tracker()

    for pkl_filename in tqdm(pkl_files):
        print(f'Processing {pkl_filename}')
        raw_data = np.load(pkl_filename, allow_pickle=True)
        print(f'Data loaded. {len(raw_data["trajectory"])} steps')
        onetraj_rgb, onetraj_state = [], []

        for step in raw_data['trajectory']:
            onetraj_rgb.append(preprocess_rgb(step['mount2']['rgb'], crop_size))

        step0 = raw_data['trajectory'][0]

        init_im = onetraj_rgb[0]

        rgbd_observation = RGBDObservation(
            step0['mount2']['rgb'], step0['mount2']['depth'],
            raw_data['camera_configs']['mount2']['intrinsics'], raw_data['camera_configs']['mount2']['extrinsics']
        )
        target_object_mask = perception_interface.get_object_mask(rgbd_observation, 'apple')
        target_object_mask = cv2.erode(target_object_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
        valid_points_yx = np.argwhere(target_object_mask)
        selected_keypoints_idx = np.random.choice(np.arange(valid_points_yx.shape[0]), size=target_keypoints_num, replace=False)
        selected_keypoints_yx_ori_size = valid_points_yx[selected_keypoints_idx]
        selected_keypoints = convert_scale(selected_keypoints_yx_ori_size, step['mount2']['rgb'].shape[:2], crop_size)[..., ::-1].copy()
        keypoint_tracker.initialize(init_im, selected_keypoints)
        tracked_keypoints_xy, tracked_keypoints_visibility = keypoint_tracker.track_offline(
            onetraj_rgb[1:], save_name=f'tracked_{os.path.basename(pkl_filename)[:-4]}'
        )
        onetraj_keypoints_xy = np.concatenate((selected_keypoints[np.newaxis, ...], tracked_keypoints_xy)) / crop_size # normalize to 0-1
        onetraj_keypoints_visibility = np.concatenate(
            (np.ones((1, selected_keypoints.shape[0]), dtype=bool), tracked_keypoints_visibility)
        )

        for step in raw_data['trajectory']:
            onetraj_state.append(get_state(step['robot']))

        # TODO
        onetraj_action = onetraj_state[1:]
        onetraj_action.append(np.zeros(10))

        all_rgb.extend(onetraj_rgb)
        all_keypoints_xy.append(onetraj_keypoints_xy)
        all_keypoints_visibility.append(onetraj_keypoints_visibility)
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
    data_group.create_dataset('keypoints_xy', data=np.concatenate(all_keypoints_xy))
    data_group.create_dataset('keypoints_visibility', data=np.concatenate(all_keypoints_visibility))
    data_group.create_dataset('state', data=np.asarray(all_state).astype(np.float32))
    meta_group.create_dataset('episode_ends', data=np.asarray(episode_end, dtype=np.int64))

    # for im in all_rgb[::100]:
    #     plt.imshow(im)
    #     plt.show()
    #     plt.close()

    print(f'save to {zarr_save_path}')


if __name__ == '__main__':
    main()
