from base import BaseDataSet, BaseDataLoader
from utils import palette
from glob import glob
import numpy as np
import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ignore_label = 255
ID_TO_TRAINID = {0: 0,
                 1: 1}

class CityScapesDataset(BaseDataSet):
    def __init__(self, mode='fine', **kwargs):
        self.num_classes = 2
        self.mode = mode
        self.palette = palette.CityScpates_palette
        self.id_to_trainId = ID_TO_TRAINID
        super(CityScapesDataset, self).__init__(**kwargs)

    def _set_files(self):
        assert (self.mode == 'fine' and self.split in ['train', 'val']) or \
        (self.mode == 'coarse' and self.split in ['train', 'train_extra', 'val'])

        SUFIX = '_gtFine_color.png'
        if self.mode == 'coarse':
            img_dir_name = 'leftImg8bit_trainextra' if self.split == 'train_extra' else 'leftImg8bit_trainvaltest'
            label_path = os.path.join(self.root, 'gtCoarse', 'gtCoarse', self.split)
        else:
            img_dir_name = 'leftImg8bit_trainvaltest'
            label_path = os.path.join(self.root, 'gtFine_trainvaltest', 'gtFine', self.split)
        image_path = os.path.join(self.root, img_dir_name, 'leftImg8bit', self.split)
        assert os.listdir(image_path) == os.listdir(label_path)

        image_paths, label_paths = [], []
        for city in os.listdir(image_path):
            image_paths.extend(sorted(glob(os.path.join(image_path, city, '*.jpg'))))
            label_paths.extend(sorted(glob(os.path.join(label_path, city, f'*{SUFIX}'))))
        self.files = list(zip(image_paths, label_paths))

    def _load_data(self, index):
        image_path, label_path = self.files[index]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        label = label.astype(np.uint8)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(label, 1, 1, cv2.THRESH_BINARY_INV)
        label = 1 - binary_img
        #label = label.astype(np.int32)
        for k, v in self.id_to_trainId.items():
            label[label == k] = v
        label = label.astype(np.int32)
        return image, label, image_id



class CityScapes(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, mode='fine', val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        self.MEAN = [0.28689529, 0.32513294, 0.28389176]
        self.STD = [0.17613647, 0.18099176, 0.17772235]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = CityScapesDataset(mode=mode, **kwargs)
        super(CityScapes, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)


