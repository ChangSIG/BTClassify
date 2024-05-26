from random import shuffle
from PIL import Image
import copy
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import torch

from core.datasets.compose import Compose


class Mydataset(Dataset):
    def __init__(self, gt_labels, cfg):
        self.gt_labels = gt_labels
        self.cfg = cfg
        self.pipeline = Compose(self.cfg)
        self.data_infos = self.load_annotations()

    def __len__(self):

        return len(self.gt_labels)

    # def __getitem__(self, index):
    #     image_path = self.gt_labels[index].split(' ')[0].split()[0]
    #     image = Image.open(image_path)
    #     cfg = copy.deepcopy(self.cfg)
    #     image = self.preprocess(image,cfg)
    #     gt = int(self.gt_labels[index].split(' ')[1])

    #     return image, gt, image_path

    # def preprocess(self, image,cfg):
    #     if not (len(np.shape(image)) == 3 and np.shape(image)[2] == 3):
    #         image = image.convert('RGB')
    #     funcs = []

    #     for func in cfg:
    #         funcs.append(eval('transforms.'+func.pop('type'))(**func))
    #     image = transforms.Compose(funcs)(image)
    #     return image

    def __getitem__(self, index):
        results = self.pipeline(copy.deepcopy(self.data_infos[index]))
        return results['img'], int(results['gt_label']), results['filename']

    def load_annotations(self):
        """Load image paths and gt_labels."""
        if len(self.gt_labels) == 0:
            raise TypeError('ann_file is None')
        samples = [x.strip().rsplit(' ', 1) for x in self.gt_labels]

        data_infos = []
        for filename, gt_label in samples:
            info = {'img_prefix': None}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos


class Mydataset_e(Dataset):
    def __init__(self, gt_labels, cfg):
        self.gt_labels = gt_labels
        self.cfg = cfg
        self.pipeline = Compose(self.cfg)
        self.data_infos = self.load_annotations()

    def __len__(self):
        return len(self.gt_labels)

    def __getitem__(self, index):
        results = self.pipeline(copy.deepcopy(self.data_infos[index]))

        # img_tensor = results['img']
        # img_mask = results['img_mask']
        # # 将img tensor转换为NumPy数组
        # img_np = img_tensor.numpy()
        # img_np_mask = img_mask.numpy()
        # # 调整图像数组的形状（如果需要）
        # img_np = np.transpose(img_np, (1, 2, 0))
        # img_np_mask = np.transpose(img_np_mask, (1, 2, 0))
        # # 显示图像
        # cv2.imshow('Image', img_np)
        # cv2.imshow('ImageMask', img_np_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if 'img_mask' in results:
            return results['img'], results['img_mask'], int(results['gt_label']), results['filename']

        return results['img'], int(results['gt_label']), results['filename']

    def load_annotations(self):
        """Load image paths and gt_labels."""
        if len(self.gt_labels) == 0:
            raise TypeError('ann_file is None')
        samples = [x.strip().rsplit(' ', 1) for x in self.gt_labels]

        data_infos = []
        for filename, gt_label in samples:
            info = {'img_prefix': None}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos

    def extract_color(self, img_file):
        # 读取图像并将其转换为HSV格式
        img = cv2.imread(img_file['filename'].replace('-1', '-2'))
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


def collate(batches):
    results = tuple(zip(*batches))
    if len(results) == 3:
        images, gts, image_path = results
        images = torch.stack(images, dim=0)
        gts = torch.as_tensor(gts)
        return images, gts, image_path
    else:
        images, images_mask, gts, image_path = results
        images = torch.stack(images, dim=0)
        images_mask = torch.stack(images_mask, dim=0)
        gts = torch.as_tensor(gts)
        return images, images_mask, gts, image_path

