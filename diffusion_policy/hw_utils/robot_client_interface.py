#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : robot_client_interface.py
# Author : Xiaolin Fang
# Email  : fxlfang@gmail.com
# Date   : 04/17/2025
#
# This file is part of Project Beep-picknplace.
# Distributed under terms of the MIT license.

"""
Minimal client interface for Franka robot in real world.
"""
import time
import zmq
import zlib
import pickle
import numpy as np
from typing import Optional, Union

# Deoxys EE to gripper_camera
EE2CAM = np.array([
    [0.01019998, -0.99989995, 0.01290367, 0.03649885],
    [0.9999, 0.0103, 0.0057, -0.034889],
    [-0.00580004, 0.01280367, 0.99989995, -0.04260014],
    [0.0, 0.0, 0.0, 1.0],
])


class FrankaController:
    def __init__(self):
        pass
    
    def get_current_joint_states(self):
        '''
        Get the current joint states of the robot.
        Returns:
            dict: A dictionary containing joint positions and end-effector pose.
        '''
        raise NotImplementedError('This method should be overridden by subclasses.')

    def get_current_joint_confs(self):
        return self.get_current_joint_states()['qpos']

    def get_gripper_camera_extrinsics(self):
        robot_state_cur = self.get_current_joint_states()
        extrinsic = robot_state_cur['ee_pose'].dot(EE2CAM)
        return extrinsic

    def capture_image(self):
        '''
        Capture an image from the robot's camera.
        Returns:
            tuple: A tuple containing RGB image, depth image, and camera intrinsics.
        '''
        raise NotImplementedError('This method should be overridden by subclasses.')

    def open_gripper(self):
        '''
        Open the robot's gripper.
        '''
        raise NotImplementedError('This method should be overridden by subclasses.')

    def close_gripper(self):
        '''
        Close the robot's gripper.
        '''
        raise NotImplementedError('This method should be overridden by subclasses.')

    def execute_cartesian_impedance_path(self, poses, gripper_isclose: Optional[Union[np.ndarray, bool]] = None, speed_factor=3):
        """
        Execute a Cartesian impedance path.
        Args:
            poses:          list of end-effector poses in world frame.
            gripper_isclose: bool or np.ndarray, whether to close the gripper.
            speed_factor:   int, speed factor for the execution.
        """
        raise NotImplementedError('This method should be overridden by subclasses.')

    def execute_joint_impedance_path(self, poses, gripper_isclose: Optional[Union[np.ndarray, bool]] = None, speed_factor=3):
        """
        Execute a joint impedance path.
        Args:
            poses:          list of joint configurations.
            gripper_isclose: bool or np.ndarray, whether to close the gripper.
            speed_factor:   int, speed factor for the execution.
        """
        raise NotImplementedError('This method should be overridden by subclasses.')

    def go_to_home(self, gripper_open=False):
        """
        Move the robot to the home position.
        Args:
            gripper_open: bool, whether to open the gripper.
        """
        raise NotImplementedError('This method should be overridden by subclasses.')

    def get_gripper_state(self):
        return self.get_current_joint_states()['gripper_state']


class FrankaPybulletController(FrankaController):
    def __init__(self, config):
        import pybullet as pb
        self.pb = pb
        pb.connect(pb.GUI)
        self.robot = pb.loadURDF(config.iksolver_config.urdf_path, useFixedBase=True)
        self.movable_joint_ids = [i for i in range(pb.getNumJoints(self.robot)) if pb.getJointInfo(self.robot, i)[2] == pb.JOINT_REVOLUTE]
        self.gripper_joint_ids = [10, 11]
        self.home_joint_conf = [-1.450444, -0.019580, -0.224367, -1.684073, 0.835103, 1.431834, -1.137849]

    def capture_image(self):
        with open('./example/captures.pkl', 'rb') as f:
            message = pickle.load(f)
        self.execute_joint_impedance_path([message['qpos']])
        return message['rgb'], message['depth'], message['intrinsics']

    def execute_joint_impedance_path(self, poses):
        for pose in poses:
            for joint_i, joint_val in enumerate(pose):
                self.pb.resetJointState(self.robot, self.movable_joint_ids[joint_i], targetValue=joint_val, targetVelocity=0)
            time.sleep(0.05)

    def execute_cartesian_impedance_path(self, poses, gripper_isclose: Optional[Union[np.ndarray, bool]] = None, speed_factor=3):
        # TODO flying gripper or diff-ik
        pass

    def get_current_joint_states(self):
        joint_states = {}
        joint_states['qpos'] = [self.pb.getJointState(self.robot, joint_i)[0] for joint_i in self.movable_joint_ids]
        ee_link_state = self.pb.getLinkState(self.robot, 10)
        from beepp.utils.rotation_utils import quaternion_to_matrix
        ee_pose = np.eye(4)
        quat_x, quat_y, quat_z, quat_w = ee_link_state[1]
        ee_pose[:3, :3] = quaternion_to_matrix(np.array([quat_w, quat_x, quat_y, quat_z]))
        ee_pose[:3, 3] = ee_link_state[0]
        joint_states['ee_pose'] = ee_pose
        joint_states['gripper_state'] = self.pb.getJointState(self.robot, self.gripper_joint_ids[0])[0]
        return joint_states

    def open_gripper(self):
        for gripper_joint_id in self.gripper_joint_ids:
            self.pb.resetJointState(self.robot, gripper_joint_id, targetValue=.04, targetVelocity=0)
            time.sleep(0.01)

    def close_gripper(self):
        for gripper_joint_id in self.gripper_joint_ids:
            self.pb.resetJointState(self.robot, gripper_joint_id, targetValue=0., targetVelocity=0)
            time.sleep(0.01)

    def go_to_home(self, gripper_open=False):
        for joint_i, joint_val in enumerate(self.home_joint_conf):
            self.pb.resetJointState(self.robot, self.movable_joint_ids[joint_i], targetValue=joint_val, targetVelocity=0)
        if gripper_open:
            self.open_gripper()
        else:
            self.close_gripper()
        time.sleep(1.)

    def dump_captured_list(self, filename):
        pass


class FrankaRealworldController(FrankaController):
    def __init__(self, robot_ip):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(robot_ip)
        self.socket = socket
        self.camera_intrinsics = None
        self.image_dim = None
        self.capture_rs = None

    def capture_image(self):
        self.socket.send(zlib.compress(pickle.dumps({'message_name': 'capture_realsense'})))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message['rgb'], message['depth'], message['intrinsics'] # dep in m (not mm, no need to /1000)

    def get_fixed_camera_extrinsic(self):
        self.socket.send(zlib.compress(pickle.dumps({'message_name': 'get_fixed_camera_extrinsic'})))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def execute_cartesian_impedance_path(
        self, poses, gripper_isclose: Optional[Union[np.ndarray, bool]] = None, speed_factor=1,
        is_capturing: bool = False, capture_step: int = 1
    ):
        """
        End-effector poses in world frame.
        """
        self.socket.send(zlib.compress(pickle.dumps({
            'message_name': 'execute_posesmat4_osc',
            'ee_poses': poses,
            'gripper_isclose': gripper_isclose,
            'speed_factor': speed_factor,
            'is_capturing': is_capturing,
            'capture_step': capture_step,
        })))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def execute_joint_impedance_path(
        self, poses, gripper_isclose: Optional[Union[np.ndarray, bool]] = None, speed_factor=3,
        is_capturing: bool = False, capture_step: int = 1
    ):
        self.socket.send(zlib.compress(pickle.dumps({
            'message_name': 'execute_joint_impedance_path',
            'joint_confs': poses,
            'gripper_isclose': gripper_isclose.astype(bool) if isinstance(gripper_isclose, np.ndarray) else gripper_isclose,
            'speed_factor': speed_factor,
            'is_capturing': is_capturing,
            'capture_step': capture_step,
        })))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def open_gripper(self):
        self.socket.send(zlib.compress(pickle.dumps({'message_name': 'open_gripper'})))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def close_gripper(self):
        self.socket.send(zlib.compress(pickle.dumps({'message_name': 'close_gripper'})))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def get_current_joint_states(self):
        self.socket.send(zlib.compress(pickle.dumps({'message_name': 'get_joint_states'})))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def go_to_home(self, gripper_open=False):
        self.socket.send(zlib.compress(pickle.dumps({'message_name': 'go_to_home', 'gripper_open': gripper_open})))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message['success']

    def free_motion(self, gripper_open=False, timeout=3.0):
        self.socket.send(zlib.compress(pickle.dumps({
            'message_name': 'free_motion_control',
            'gripper_open': gripper_open,
            'timeout': timeout,
        })))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message['success']

    def reset_joint_to(self, qpos, gripper_open=False):
        self.socket.send(zlib.compress(pickle.dumps({
            'message_name': 'reset_joint_to',
            'gripper_open': gripper_open,
            'qpos': qpos,
        })))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message['success']

    def dump_captured_list(self, filename):
        self.socket.send(zlib.compress(pickle.dumps({
            'message_name': 'dump_captured_list',
            'filename': filename,
        })))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message['success']


def initialize_robot_interface(robot_ip) -> FrankaController:
    robot_interface = FrankaRealworldController(robot_ip)
    return robot_interface
