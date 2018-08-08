# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from skimage import transform
import matplotlib.pyplot as plt

class TorchImageProcessor:
    """Simple data processors"""

    def __init__(self, image_size, #is_color, mean, scale,
                 crop_size=0, pad=28, color='RGB',#'BGR',
                 use_cutout=False,
                 use_mirroring=False,
                 use_random_crop=False,
                 use_center_crop=False,
                 use_random_gray=False):
        """Everything that we need to init"""
        self.image_size = image_size
        pass

    def process(self, image_path):
        """
        Returns processed data.
        """
        try:
            image = cv2.imread(image_path)
        except:
            image = image_path

        if image is None:
            print(image_path)

        # TODO: реализуйте процедуры аугментации изображений используя OpenCV и TorchVision
        # на выходе функции ожидается массив numpy с нормированными значениям пикселей
        print(image.shape)
        # Resize and scaling
        image = self.resize(image, output_size=self.image_size)
        return image
    
    def resize(self, image, output_size):
        # from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        h, w = image.shape[:2]
        if isinstance(output_size, int):
            if h > w:
                new_h, new_w = output_size * h / w, output_size
            else:
                new_h, new_w = output_size, output_size * w / h
        else:
            new_h, new_w = output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return img
    
    def show(self, image):
        plt.figure()
        plt.imshow(image)
        plt.show()
    