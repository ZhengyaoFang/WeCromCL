import os
import os.path as op
import math
import sys
import cv2
import random
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import json
from .augmentations import PSSAugmentation
import imageio
import re

def label_filter(cfg, text):
    alphabet = cfg.data.alphabet
    pattern = f"[{re.escape(alphabet)}]"
    new_text = ''.join(re.findall(pattern, text))
    return new_text

class ImagePaths(Dataset):
    def __init__(self, cfg, image_rootdir, annotation_file, training=True):
        self.cfg = cfg
        self.is_training = training
        self.img_dir = image_rootdir
        self.augmentaion = PSSAugmentation(cfg, training)
        self.load_data(image_rootdir, annotation_file)
    
    def load_data(self, image_rootdir, annotation_file):
        self.img_paths = []
        self.labels = []
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        for img_name in data:
            if len(self.img_paths) == 100:
                break
            new_texts = []
            for text in data[img_name]:
                new_text = label_filter(self.cfg, text)
                if len(new_text) != 0:
                    new_texts.append(new_text)
            if len(new_texts) != 0: 
                self.img_paths.append(os.path.join(image_rootdir, img_name))
                self.labels.append(new_texts)
        self._length = len(self.img_paths)
        print("Loading data from {} containing {} images".format(annotation_file, self._length))
    
    def preprocess_image(self, img_path):
        img = imageio.imread(img_path, pilmode='RGB')
        ori_height, ori_width, _ = img.shape
        img = self.augmentaion(img)
        return img, (ori_width, ori_height)
    
    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = dict()
        example['image_path'] = os.path.join(self.img_dir, self.img_paths[index])
        example['image'], example['image_size'] = self.preprocess_image(example['image_path'])
        example['recs'] = self.labels[index]
        return example

class BaseDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data = None
        self.data_propotion = None
        self._length = None
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        if isinstance(self.data_proportion, list):
            p = np.array(self.data_proportion) / sum(self.data_proportion)
            data_index = np.random.choice(np.arange(len(self.data)), p = p.ravel())
            data = self.data[data_index]
            example = data[index % len(data)]
        else:
            example = self.data[index]
        return example

class TrainDataset(BaseDataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        if isinstance(cfg.data.train.annotation_files, str):
            self.data = ImagePaths(cfg, cfg.data.train.img_rootdir ,cfg.data.train.annotation_files, training=True)
        else:
            if cfg.data.train.data_proportion == None:
                data = {}
                length = 0
                for file in cfg.data.train.annotation_files:
                    data_ = ImagePaths(cfg, file, training=True)
                    length += len(data_)
                    for k in data_:
                        data[k] += data_[k]
                    
                self.data = data
                self._length = length
            else:
                training_annotation_file = list(cfg.data.train.annotation_files)
                image_rootdirs = list(cfg.data.train.img_rootdir)
                self.data_proportion = list(cfg.data.train.data_proportion)
                self.data = []
                length = 0
                for img_dir,file in zip(image_rootdirs, training_annotation_file):
                    data_ = ImagePaths(cfg, img_dir, file, training=True)
                    length += len(data_)
                    self.data.append(data_)
                self._length = length

class TestDataset(BaseDataset):
    pass

class TextCollate(object):
    def __init__(self, cfg, dataset):
        self.cfg = cfg

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        self.dataset = dataset
        self.load_vocabulary(cfg.vocabulary_file)
    
    def load_vocabulary(self, vocabulary_file):
        with open(vocabulary_file, 'r') as f:
            lines = f.readlines()
        self.vocabulary = [label_filter(self.cfg, line.strip()) for line in lines]
    
    def __call__(self, batch):
        all_exist_texts = []
        confirmed_examples = []

        while len(confirmed_examples) < len(batch):
            choice = random.randint(0, self.dataset.__len__() - 1)
            example = self.dataset.__getitem__(choice)
            compatible_texts = set(example['recs'])-set(all_exist_texts)
            if len(compatible_texts) != 0:
                example['selected_text'] = random.choice(list(compatible_texts))
                example['false_texts'] = random.sample(set(random.choices(self.vocabulary, k=500))-set(example['recs']), self.cfg.false_label_num)
                confirmed_examples.append(example)
                all_exist_texts += example['recs']

        imgs = [self.transformer(example['image'].astype(np.uint8)) for example in confirmed_examples]
        
        max_size = self._max_by_axis([list(img.shape) for img in imgs])
        batch_shape = [len(imgs)] + [max_size[0], math.ceil(max_size[1] * 1.0 / 32) * 32,
                                     math.ceil(max_size[2] * 1.0 / 32) * 32]
        dtype = imgs[0].dtype
        device = imgs[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        b, c, h, w = tensor.shape
        for img, pad_img in zip(imgs, tensor):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        return tensor, confirmed_examples
    
    def _max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfgs = OmegaConf.load("configs/Base.yaml")
    