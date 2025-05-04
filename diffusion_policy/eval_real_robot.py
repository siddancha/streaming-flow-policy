#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : eval_robot.py
# Author : Xiaolin Fang
# Email  : fxlfang@gmail.com
# Date   : 04/27/2025
#
# This file is part of Project streaming-flow-policy.
# Distributed under terms of the MIT license.

import os
import cv2
import time
import argparse
import torch
import hydra
import dill
import pickle
import numpy as np
import os.path as osp
from beepp.perception import initialize_perception_interface, RGBDObservation
from diffusion_policy.convert_to_zarr import get_state, preprocess_rgb, center_crop
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.hw_utils.robot_client_interface import initialize_robot_interface
from diffusion_policy.hw_utils.rotation_utils import compute_rotation_matrix_from_ortho6d


CONTROL_FREQUENCY = 10 # Default 20 Hz
VERBOSE_TIME_PERF = True #False
gripper_open_width = 0.079

def get_obs_dict(obs_im_buffer, action_buffer, args, keypoint_tracker=None):
    if VERBOSE_TIME_PERF:
        global prev_obs_time
        cur_obs_time = time.perf_counter()
        if prev_obs_time is not None:
            print(f'>>>> Update traj time elapse {cur_obs_time - prev_obs_time}')
        prev_obs_time = cur_obs_time
    robot_states_10d = np.stack([get_state(state).astype(np.float32) for state in action_buffer[-2:]])
    if args.kp2d or args.kp3d:
        st_tracking = time.perf_counter()
        # TODO: Keypoint tracker memory update & Async Inference
        last_obs = preprocess_rgb(center_crop(obs_im_buffer[-1].rgb_im))
        tracked_keypoints_ij, tracked_keypoints_visibility = keypoint_tracker.track_online(last_obs)
        # TODO: double check the following code and 2d kp input
        tracked_keypoints_ij = tracked_keypoints_ij[-2:]
        tracked_keypoints_visibility = tracked_keypoints_visibility[-2:]
        tracked_keypoints_ij /= args.crop_size

        show_image_with_keypoints(last_obs, tracked_keypoints_ij[-1] * args.crop_size, tracked_keypoints_visibility[-1])

        if args.kp3d:
            scene_pcds_worldframe = [preprocess_rgb(obs.pcd_worldframe, crop_size=args.crop_size) for obs in obs_im_buffer[-2:]]
            tracked_keypoints_ij_scaled = (tracked_keypoints_ij * args.crop_size).astype(np.int64)
            tracked_keypoints_ij_scaled[tracked_keypoints_ij_scaled == args.crop_size] = args.crop_size - 1
            tracked_keypoints_loc = np.stack(
                [pcd[ij[:, 1], ij[:, 0]] for pcd, ij in zip(scene_pcds_worldframe, tracked_keypoints_ij_scaled)]
            )
        else:
            tracked_keypoints_loc = tracked_keypoints_ij

        assert tracked_keypoints_loc.shape[0] == 2
        assert tracked_keypoints_visibility.shape[0] == 2
        ##########################################

        obs_dict_np = {
            'keypoint': np.concatenate((
                tracked_keypoints_visibility.astype(np.float32)[..., np.newaxis],
                tracked_keypoints_loc.astype(np.float32)
            ), axis=-1),
            'agent_pos': robot_states_10d,  # T, 10
        }
        if VERBOSE_TIME_PERF:
            print(f'>>>> keypoint tracking time elapse {time.perf_counter() - st_tracking}')
    else:
        images = np.array([preprocess_rgb(center_crop(obs.rgb_im)) for obs in obs_im_buffer[-2:]])
        image = np.moveaxis(images, -1, 1) / 255
        obs_dict_np = {
            'image': image,
            'agent_pos': robot_states_10d,  # T, 10
        }
        show_image_with_keypoints(images[-1], [], [])
    return obs_dict_np


def get_mat4_from_9d(ee_pose_9d):
    # Convert 9D pose to 4x4 matrix
    translation = ee_pose_9d[:3]
    rotation = ee_pose_9d[3:]
    mat4 = np.eye(4)
    mat4[:3, :3] = compute_rotation_matrix_from_ortho6d(torch.from_numpy(rotation).unsqueeze(0)).numpy()[0]
    mat4[:3, 3] = translation
    return mat4


def show_image_with_keypoints(im, keypoints_ij, keypoints_visibility):
    vis_im = im[..., ::-1].copy()
    if len(keypoints_ij) > 0:
        for i, (kp_i, kp_j) in enumerate(keypoints_ij.astype(np.int64)):
            if keypoints_visibility[i]:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.circle(vis_im, (kp_i, kp_j), 3, color, -1)
    cv2.imshow('mount2', vis_im)
    cv2.waitKey(1)


def main(args):
    # load checkpoint
    payload = torch.load(open(args.ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # hacks for method-specific setup.
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 16 # DDIM inference iterations
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    camera_names = ['mount2']
    robot_interface = initialize_robot_interface(args.robot_ip, local_rs=args.local_rs)
    if args.kp2d or args.kp3d:
        perception_interface = initialize_perception_interface()
        keypoint_tracker = Tracker()
    else:
        keypoint_tracker = None


    robot_interface.go_to_home(gripper_open=True)
    extrinsics_allcameras = robot_interface.get_fixed_camera_extrinsic()
    extrinsics = extrinsics_allcameras[camera_names[0]]

    cv2.namedWindow(camera_names[0], cv2.WINDOW_NORMAL)
    rgb_im, dep_im, intrinsics = robot_interface.capture_image()
    robot_state = robot_interface.get_current_joint_states()
    rgbd_observation = RGBDObservation(rgb_im, dep_im, intrinsics, extrinsics)
    obs_im_buffer: list[RGBDObservation] = [rgbd_observation, rgbd_observation]
    action_buffer = [robot_state, robot_state]

    if args.kp2d or args.kp3d:
        # extrinsics = np.array([[ 0.0024531 ,  0.55724814, -0.8303424 ,  1.11168072],
        #    [ 0.99965826,  0.02024397,  0.01653918, -0.38456383],
        #    [ 0.02602586, -0.83009921, -0.55700804,  0.3821937 ],
        #    [ 0.        ,  0.        ,  0.        ,  1.        ]])
        target_object_mask = perception_interface.get_object_mask(obs_im_buffer[-1], 'apple')
        target_object_mask = cv2.erode(target_object_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
        valid_points_yx = np.argwhere(target_object_mask)
        selected_keypoints_idx = np.random.choice(np.arange(valid_points_yx.shape[0]), size=args.keypoint_num, replace=False)
        selected_keypoints_yx_ori_size = valid_points_yx[selected_keypoints_idx]
        from diffusion_policy.convert_to_zarr import convert_scale
        selected_keypoints = convert_scale(selected_keypoints_yx_ori_size, rgb_im.shape[:2], args.crop_size)[..., ::-1].copy()
        keypoint_tracker.initialize(preprocess_rgb(rgb_im, args.crop_size), selected_keypoints)


    # dry run test
    with torch.no_grad():
        policy.reset()
        obs_dict_np = get_obs_dict(obs_im_buffer, action_buffer, args, keypoint_tracker=keypoint_tracker)
        obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
        result = policy.predict_action(obs_dict)
        action = result['action'][0].detach().to('cpu').numpy()
        print(f'{action.shape=}')
        assert action.shape[-1] == 10
        del result

    while True:
        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk_10d = None
        action_10d_prev = None
        obs_time_prev = None

        # video = []
        prev_action =  None
        for t_step in range(600):
            start_time = time.time()
            try:
                # Get the current observation
                if VERBOSE_TIME_PERF:
                    cur_obs_time = time.perf_counter()
                    if obs_time_prev is not None:
                        print(f'>>>> Update obs time elapse {cur_obs_time - obs_time_prev}')
                    obs_time_prev = cur_obs_time

                obs_rgb, obs_dep, obs_intrinsics = robot_interface.capture_image()
                obs_im_buffer.append(RGBDObservation(obs_rgb, obs_dep, obs_intrinsics, extrinsics))
                action_buffer.append(robot_interface.get_current_joint_states())
                # Time elapse: 0.05 per if local_rs. 0.2 per if remote rs

                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # run inference
                    with torch.no_grad():
                        s = time.time()
                        obs_dict_np = get_obs_dict(obs_im_buffer, action_buffer, args, keypoint_tracker=keypoint_tracker)

                        obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                        # result = policy.predict_action(obs_dict)
                        if hasattr(policy, "use_action_traj") and policy.use_action_traj: # if policy has attribute use_action_traj and it is True
                            result = policy.predict_action(obs_dict, prev_action=prev_action)
                        else:
                            result = policy.predict_action(obs_dict)
                        # this action starts from the first obs step
                        if hasattr(policy, "use_action_traj") and policy.use_action_traj: 
                            prev_action = result['prev_action']
                        action_chunk = result['action'][0].detach().to('cpu').numpy()
                        print('Inference latency:', time.time() - s)
                        # print(action_chunk)

                    if action_10d_prev is None:
                        action_10d_prev = action_chunk[0]
                    # convert policy action to env actions
                    pred_action_chunk_10d = action_chunk

                # Select current action to execute from chunk
                action_10d = pred_action_chunk_10d[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                osc_pose_target = get_mat4_from_9d(action_10d[: 9])
                if action_10d[-1].item() < gripper_open_width and action_10d[-1].item() < action_10d_prev[-1].item():
                    closing_gripper = True
                elif action_10d[-1].item() > gripper_open_width:# and action_10d[-1].item() > action_10d_prev[-1].item():
                    closing_gripper = False
                else:
                    closing_gripper = None

                robot_interface.execute_cartesian_impedance_path(
                    [osc_pose_target],
                    gripper_isclose=closing_gripper,
                )

                action_10d_prev = action_10d

                # Sleep to match data collection control frequency. Jittering if not.
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / CONTROL_FREQUENCY:
                    if VERBOSE_TIME_PERF:
                        print(f'>>>>>>>> WAITING till elapsed go from {elapsed_time} to {1 / CONTROL_FREQUENCY}')
                    time.sleep(1 / CONTROL_FREQUENCY - elapsed_time)

                # cv2.imshow(camera_names[0], obs_im_buffer[-1].rgb_im[..., ::-1])
                # cv2.waitKey(1)

            except KeyboardInterrupt:
                break

    cv2.destroyWindow(camera_names[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--robot_ip', type=str, default=f'tcp://{os.environ["FR3_CONTROL_ADDR"]}:5560')
    parser.add_argument('--open_loop_horizon', type=int, default=8, help='Open loop horizon for action prediction')
    parser.add_argument('--kp2d', action='store_true', help='use keypoint 2d conditioned policy')
    parser.add_argument('--kp3d', action='store_true', help='use keypoint 3d conditioned policy')
    parser.add_argument('--keypoint_num', type=int, default=10, help='Number of keypoints to use in kp policy')
    parser.add_argument('--crop_size', type=int, default=256, help='Resize target size')
    parser.add_argument('--local_rs', action='store_true', help='RealSense connected to this machine directly')
    args = parser.parse_args()
    if args.kp2d or args.kp3d:
        from streaming_flow_policy.franka.keypoint_tracker import Tracker
    if VERBOSE_TIME_PERF:
        global prev_obs_time
        prev_obs_time = None
    main(args)
