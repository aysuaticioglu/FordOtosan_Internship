import numpy as np
import cv2
import json
import os
import torch
import tqdm 
from matplotlib import pyplot as plt
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage import transform
import random 
from torchvision import transforms as T
from PIL import Image

IMAGE_DIR = "data/images" 
MASK_DIR = "data/masks"


image_path = [os.path.join(IMAGE_DIR, name) for name in os.listdir(IMAGE_DIR) if name.endswith('.png')]
mask_path = [os.path.join(MASK_DIR, name) for name in os.listdir(MASK_DIR) if name.endswith('.png')]

valid_size = 0.3
test_size = 0.1
indices = np.random.permutation(len(image_path))
test_ind = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)
train_input_path_list = image_path[valid_ind:]
train_label_path_list = mask_path[valid_ind:]

for image, mask in tqdm.tqdm(zip(train_input_path_list, train_label_path_list)):
    if os.path.isfile(image) and os.path.isfile(mask):
        img = Image.open(image)
        msk = cv2.imread(mask)
        color_aug = T.ColorJitter(brightness=0.4, contrast=0.4, hue=0.06)
        img_aug = color_aug(img)
        new_image_path = image[:-4] + "-1.png"
        new_mask_path = mask[:-4] + "-1.png"
        img_aug = np.array(img_aug)
        cv2.imwrite(new_image_path, img_aug)
        cv2.imwrite(new_mask_path, msk)

