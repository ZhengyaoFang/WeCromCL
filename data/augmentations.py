import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from shapely.geometry import box, Polygon
import math
from math import fabs, sin, cos, radians
from shapely import affinity
import logging
from imgaug import augmenters as iaa

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
    
class RandomResize(object):
    def __init__(self, min_size, max_size):
        self.min_sizes = min_size
        self.max_size = max_size
        logging.info('Random resize min sizes is {}, max_size is {}'.format(self.min_sizes, self.max_size))
        
    def __call__(self, image):
        min_size = random.choice(self.min_sizes)

        size = self.get_size_with_aspect_ratio(image.shape[:2], min_size, self.max_size)
        rescaled_image = cv2.resize(image, size)


        return rescaled_image

    def get_size_with_aspect_ratio(self, image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        oh = math.ceil(oh / 32) * 32
        ow = math.ceil(ow / 32) * 32
        return (oh, ow)
    
class Resize(object):
    def __init__(self, min_side, max_side):
        self.min_side = math.ceil(min_side / 32) * 32
        self.max_side = math.ceil(max_side / 32) * 32
    
    def __call__(self, image):
        ori_h, ori_w = image.shape[:2]
        if self.min_side > 0:
            if ori_h < ori_w:
                test_h = self.min_side
                test_w = int(ori_w * self.min_side / ori_h)
                test_w = math.ceil(test_w / 32) * 32
            else:
                test_w = self.min_side
                test_h = int(ori_h * self.min_side / ori_w)
                test_h = math.ceil(test_h / 32) * 32
        elif self.max_side > 0:
            if ori_h < ori_w:
                test_w = self.max_side
                test_h = int(ori_h * self.max_side / ori_w)
                test_h = math.ceil(test_h / 32) * 32
            else:
                test_h = self.max_side
                test_w = int(ori_w * self.max_side / ori_h)
                test_w = math.ceil(test_w / 32) * 32                
        else:
            raise ValueError("min_side or max_side must be positive")

        img_resized = cv2.resize(image, (test_w, test_h))
        return img_resized

        
class RandomRotate(object):
    def __init__(self, prob, max_theta=10):
        self.prob = prob
        self.max_theta = max_theta
        logging.info('Random rotate max_theta={}'.format(max_theta))

    def __call__(self, image):
        if random.random() < self.prob:
            degree = random.uniform(-1 * self.max_theta, self.max_theta)
            #degree = 10
            height, width, _ = image.shape
            heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
            widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
            matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

            matRotation[0, 2] += (widthNew - width) / 2
            matRotation[1, 2] += (heightNew - height) / 2
            imgRotation = cv2.warpAffine(image, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))
            return imgRotation
        else:
            return image

class RandomBlur(object):
    def __init__(self, prob=0.1):
        self.prob = prob
        self.blur_methods = ['GaussianBlur', 'AverageBlur', 'MedianBlur', 'BilateralBlur', 'MotionBlur']

    def __call__(self, image):
        ## aug1 0-3.0  3-9 3-9 3-9 3-9
        if random.random() < self.prob:
            blur = random.choice(self.blur_methods)
            if blur == 'GaussianBlur':
                seq = iaa.Sequential([iaa.GaussianBlur(sigma=(0, 3.0))])
            elif blur == 'AverageBlur':
                seq = iaa.Sequential([iaa.AverageBlur(k=(3, 9))])
            elif blur == 'MedianBlur':
                seq = iaa.Sequential([iaa.MedianBlur(k=(3, 9))])
            elif blur == 'BilateralBlur':
                seq = iaa.Sequential([iaa.BilateralBlur((3, 9), sigma_color=250, sigma_space=250)])
            else:
                seq = iaa.Sequential([iaa.MotionBlur(k=(3, 9), angle=0, direction=0.0)])

            images = np.expand_dims(image, axis=0)
            images_aug = seq.augment_images(images)
            image = images_aug[0]
        return image

class ToPIL(object):
    def __init__(self):
        self.converter = transforms.ToPILImage()

    def __call__(self, image):
        return self.converter(image.astype(np.uint8))

class RandomColorJitter(object):
    def __init__(self, prob):
        self.prob = prob
        self.func = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)

    def __call__(self, image):
        if random.random() < self.prob:
            image = self.func(image)
            return image
        return image

class ToNP(object):
    def __init__(self):
        pass

    def __call__(self, image):
        return np.array(image)

class Normalize(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    def __call__(self, image):
        return self.normalize(image)

class PSSAugmentation(object):
    def __init__(self, cfg, trainging):
        self.cfg = cfg
        self.training = trainging
        if self.training:
            self.augment = Compose([
                RandomRotate(1, 10),
                RandomResize(min_size=list(cfg.data.train.augmentation.ResizeMinsizes), max_size=cfg.data.train.augmentation.ResizeMaxsize),
                RandomBlur(0.1),
                ## cv2 to PIL
                ToPIL(),
                ## apply ColorJitter
                RandomColorJitter(0.5),
                ToNP(),
            ])
        else:
            self.augment = Compose([
                Resize(cfg.data.test.input_min_side, cfg.data.test.input_max_side),
                ToPIL(), 
                ToNP(), 
                transforms.ToTensor(),
                Normalize()
            ])
        

    def __call__(self, img):
        return self.augment(img)