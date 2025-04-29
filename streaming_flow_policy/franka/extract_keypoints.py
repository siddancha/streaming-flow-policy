#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : extract_keypoints.py
# Author : Xiaolin Fang
# Email  : fxlfang@gmail.com
# Date   : 04/16/2025
#
# Distributed under terms of the MIT license.

"""

"""
import glob
import numpy as np
from keypoint_utils import load_data
from keypoint_tracker import Tracker
from streaming_flow_policy.franka.keypoint_utils import MaskPointPicker, MaskPickerConfig


def test_keypoint_selection():
    mask_picker = MaskPointPicker(MaskPickerConfig())
    keypoint_tracker = Tracker()
    data = load_data('trajectory_20250407-171604_replay.pkl')
    im = data[0]['mount2']['rgb_im']

    # # Run auto keypoint selection
    # seeding_points = mask_picker.get_seeding_point(im)
    # selected_mask = mask_picker.select_mask_from_point_query(seeding_points)
    # show_images(im, selected_mask)

    # Manual keypoint selection
    selected_keypoints = mask_picker.select_keypoints(im)
    print(selected_keypoints)
    keypoint_tracker.initialize(im, np.array(selected_keypoints))
    keypoint_tracker.track_offline([im for _ in range(10)], save_name='tracked.mp4')


def main(data_folder):
    pkl_filenames = glob.glob(data_folder + '/**_replay.pkl')
    print(pkl_filenames)

    mask_picker = MaskPointPicker(MaskPickerConfig())
    keypoint_tracker = Tracker()

    keypoint_records = []

    for pkl_filename_full in pkl_filenames:
        data = load_data(pkl_filename_full)
        pkl_filename = pkl_filename_full.split('/')[-1]
        print(f"Processing {pkl_filename}")
        image_sequence = [step_data['mount2']['rgb_im'] for step_data in data]
        init_im = image_sequence[0]
        selected_keypoints = np.array(mask_picker.select_keypoints(init_im))
        print(selected_keypoints)
        keypoint_tracker.initialize(init_im, selected_keypoints)
        tracked_keypoints_xy, tracked_keypoints_visibility = keypoint_tracker.track_offline(image_sequence[1:], save_name=f'tracked_{pkl_filename[:-4]}.mp4')
        keypoint_records.append({
            'record_name': pkl_filename,
            'keypoints_xy': np.concatenate((selected_keypoints[np.newaxis, ...],tracked_keypoints_xy)),
            'keypoints_visibility': np.concatenate((np.ones((1, selected_keypoints.shape[0]), dtype=bool), tracked_keypoints_visibility)),
        })

    np.savez('keypoint_records.npz', keypoint_records)


if __name__ == '__main__':
    main('../../../hat/data/real/drawer_250407/')
    # test_keypoint_selection()


