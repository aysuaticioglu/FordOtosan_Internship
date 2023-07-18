import os
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from constant import *


# Create a list which contains every file name in masks folder
mask_list = os.listdir(MASK_DIR)
# Remove hidden files if any
for f in mask_list:
    if f.startswith('.'):
        mask_list.remove(f)

# For every mask image
for mask_name in tqdm.tqdm(mask_list):
    # Name without extension
    mask_name_without_ex = mask_name.split('.')[0]

    # Access required folders
    mask_path      = os.path.join(MASK_DIR, mask_name)
    image_path     = os.path.join(IMAGE_DIR, mask_name_without_ex+'.jpg')
    image_out_path = os.path.join(IMAGE_OUT_DIR, mask_name)

    # Read mask and corresponding original image

    #########################################
    # CODE
    #########################################

    # Change the color of the pixels on the original image that corresponds
    # to the mask part and create new image

    #########################################
    # CODE
    #########################################

    # Write output image into IMAGE_OUT_DIR folder


    #########################################
    # CODE
    #########################################

    # Visualize created image if VISUALIZE option is chosen
    if VISUALIZE:

        #########################################
        # CODE
        #########################################
