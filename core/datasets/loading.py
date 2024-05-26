# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import cv2
import numpy as np

from core.datasets.io import imfrombytes
from .build import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        
    def get(self,filepath):
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.
        """
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.get(filename)
        img = imfrombytes(img_bytes, flag=self.color_type)
        
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    # def __repr__(self):
    #     repr_str = (f'{self.__class__.__name__}('
    #                 f'to_float32={self.to_float32}, '
    #                 f"color_type='{self.color_type}', "
    #                 f'file_client_args={self.file_client_args})')
    #     return repr_str


@PIPELINES.register_module()
class LoadImageFromFileE(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.

    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type

    def get(self, filepath):
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.
        """
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def extract_color(self, img_file):
        # 读取图像并将其转换为HSV格式
        img = cv2.imread(img_file)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 定义要筛选的颜色范围
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])

        lower_yellow = np.array([10, 100, 100])
        upper_yellow = np.array([40, 255, 255])

        lower_red_1 = np.array([0, 100, 100])
        upper_red_1 = np.array([10, 255, 255])

        lower_red_2 = np.array([150, 100, 100])
        upper_red_2 = np.array([179, 255, 255])

        # 创建三个掩码
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask_red_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

        # 对掩码进行形态学操作以去除噪点
        kernel = np.ones((5, 5), np.uint8)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_red_1 = cv2.morphologyEx(mask_red_1, cv2.MORPH_OPEN, kernel)
        mask_red_2 = cv2.morphologyEx(mask_red_2, cv2.MORPH_OPEN, kernel)

        # 在原始图像上应用掩码，以仅显示有颜色的像素区域
        result_blue = cv2.bitwise_and(img, img, mask=mask_blue)
        result_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)
        result_red_1 = cv2.bitwise_and(img, img, mask=mask_red_1)
        result_red_2 = cv2.bitwise_and(img, img, mask=mask_red_2)

        # 将所有结果合并在一张图像上
        result = result_blue + result_yellow + result_red_1 + result_red_2
        return result

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        maskfile = filename.replace('-1', '-2')

        img_bytes = self.get(filename)
        img = imfrombytes(img_bytes, flag=self.color_type)
        # imgmask_bytes = self.get(maskfile)
        # imgmask = imfrombytes(imgmask_bytes, flag=self.color_type)
        imgmask = self.extract_color(maskfile)

        if self.to_float32:
            img = img.astype(np.float32)
            imgmask = imgmask.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_mask'] = imgmask
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results