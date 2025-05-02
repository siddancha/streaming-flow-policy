#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : keypoint_tracker.py
# Author : Xiaolin Fang
# Email  : fxlfang@gmail.com
# Date   : 01/12/2025
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.
from cotracker.utils.visualizer import Visualizer
import numpy as np
import torch

class Tracker:
    def __init__(self, device='cuda'):
        self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)
        self.device = torch.device(device)
        self.history = None

    def initialize(self, image: np.ndarray, keypoints: np.ndarray):
        """
        Initialize the tracker with the first frame and the keypoints

        Args:
            image: the first frame 0-255 H W C
            keypoints: the keypoints on image (ij). N x 2
        """
        video = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float().to(self.device)  # B T C H W 0-255
        keypoints = torch.cat((torch.ones((1, keypoints.shape[0], 1)), torch.tensor(keypoints).unsqueeze(0)), dim=2).float().to(self.device)
        self.history = [image]
        self.model(video_chunk=video, is_first_step=True, queries=keypoints)

    @torch.no_grad()
    def track_offline(self, image_sequence: list[np.ndarray], save_name=None):
        """
        Track the keypoints in the video

        Args:
            image_sequence: the video to track. H W C 0-255
        """
        video = torch.tensor(np.array(image_sequence)).permute(0, 3, 1, 2).unsqueeze(0).float().to(self.device)
        # Process the video
        for ind in range(0, video.shape[1] - self.model.step, self.model.step):
            pred_tracks, pred_visibility = self.model(video_chunk=video[:, ind: ind + self.model.step * 2])  # B T N 2,  B T N
        if save_name is not None:
            vis = Visualizer(save_dir="./saved_videos", pad_value=0, linewidth=3)
            vis.visualize(video, pred_tracks, pred_visibility, query_frame=0, filename=save_name)
        tracked_keypoints_xy = pred_tracks[0].cpu().numpy() # T N 2
        tracked_keypoints_visibility = pred_visibility[0].cpu().numpy().astype(bool) # T N
        return tracked_keypoints_xy, tracked_keypoints_visibility

    @torch.no_grad()
    def track_online(self, image: np.ndarray):
        """
        Track the keypoints in the video

        Args:
            image: the next image to track. H W C 0-255
        """
        self.history.append(image)
        if len(self.history) < self.model.step * 2:
            video = (torch.tensor(np.array(self.history[:-1])).permute(0, 3, 1, 2).unsqueeze(1).unsqueeze(0)
                     .repeat(1, 1, (self.model.step * 2) // len(self.history), 1, 1, 1)
                     .reshape(1, (self.model.step * 2) // len(self.history) * len(self.history[:-1]), 3, *image.shape[:2]))
            video = torch.cat(
                (video, torch.tensor(np.array(self.history[-1:])).permute(0, 3, 1, 2).unsqueeze(1).repeat(1, self.model.step * 2 - video.shape[1], 1,1,1)),
                dim=1).float().to(self.device)  # B T C H W 0-255
        else:
            video = torch.tensor(np.array(self.history[-self.model.step*2:])).permute(0, 3, 1, 2).unsqueeze(0).float().to(self.device)  # B T C H W 0-255
        pred_tracks, pred_visibility = self.model(video)
        tracked_keypoints_xy = pred_tracks[0].cpu().numpy() # T N 2
        tracked_keypoints_visibility = pred_visibility[0].cpu().numpy().astype(bool) # T N
        # self.history.pop(0)
        return tracked_keypoints_xy, tracked_keypoints_visibility
