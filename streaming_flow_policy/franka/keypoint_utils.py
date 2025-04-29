#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : keypoint_utils.py
# Author : Xiaolin Fang
# Email  : fxlfang@gmail.com
# Date   : 04/16/2025
#
# Distributed under terms of the MIT license.

"""

"""
import os
import cv2
import tap
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

def load_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def overlay_mask_simple(rgb_im, mask: np.ndarray, colors=None, mask_alpha=.5):
    if rgb_im.max() > 2:
        rgb_im = rgb_im.astype(np.float32) / 255.
    if colors is None:
        colors = np.array([1, 0, 0])
    return (rgb_im * (1 - mask_alpha) + mask[..., np.newaxis] * colors * mask_alpha).copy()


def show_images(im, selected_mask):
    fig, axes = plt.subplots(1, 2, figsize=(20, 20))
    axes[0].imshow(im)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(overlay_mask_simple(im, selected_mask))
    axes[1].set_title('Overlayed Image')
    axes[1].axis('off')
    plt.show()
    plt.close()


class MaskPointPicker:
    """
    MaskPicker is used to pick the mask to sample keypoints or pick the keypoints directly.
    """
    def __init__(self, config):
        # load SAM model
        self.config = config
        self.rgb_im = None
        self.pcd = None
        device = config.device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", self.config.sam2_ckpt_path).to(device=device)
        self.predictor = SAM2ImagePredictor(sam)
        # 1024 x 1024 is the input size for SAM pretrained model
        self.device = torch.device(device)

        self.seeding_point = []
        self.image = None

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Coordinates: ({x}, {y})")
            self.seeding_point.append((x, y))
            image_show = self.image.copy()
            cv2.circle(image_show, (x, y), 3, (0, 0, 255), -1)
            for prev_x, prev_y in self.seeding_point:
                cv2.circle(image_show, (prev_x, prev_y), 3, (0, 255, 0), -1)
            cv2.imshow('image', image_show[..., ::-1])

    def select_mask_from_point_query(self, seeding_point, input_label=None) -> list[np.ndarray]:
        """
        Query SAM with point prompts.
        """
        if input_label is None:
            input_label = np.ones(len(seeding_point))
        with torch.no_grad():
            self.predictor.set_image(self.image)
            masks, scores, logits = self.predictor.predict(
                point_coords=seeding_point,
                point_labels=input_label,
                multimask_output=True,
            )
        mask_list = []
        for (mask, score) in zip(masks, scores):
            mask_list.append((mask, score))
        mask_list.sort(key=lambda x: x[1], reverse=True)
        fig, axes = plt.subplots(1, len(mask_list), figsize=(20, 20))
        for i, (mask, score) in enumerate(mask_list):
            axes[i].imshow(overlay_mask_simple(self.image, mask))
            axes[i].set_title(f'Mask {i+1:02d}. Score: {score:.3f}')
            axes[i].axis('off')
        plt.show()
        plt.close()
        selected_mask_id = input('Select mask number to keep: ')
        selected_mask = mask_list[int(selected_mask_id)-1]
        return selected_mask[0]

    def select_keypoints(self, rgb_image):
        self.seeding_point = []
        self.image = rgb_image
        cv2.imshow('image', rgb_image[..., ::-1])
        cv2.setMouseCallback('image', self.click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self.seeding_point


class MaskPickerConfig(tap.Tap):
    """
    MaskPickerConfig is used to configure the MaskPicker.
    """
    # path to the checkpoint of SAM model
    sam2_ckpt_path: str = os.path.join(os.path.dirname(__file__), './sam2.1_hiera_large.pt')
    min_area_percentage: float = .0001

    device: str = 'cuda'
