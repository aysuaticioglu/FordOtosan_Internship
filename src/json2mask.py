import json
import os
import numpy as np
import cv2
import tqdm
from constant import JSON_DIR, MASK_DIR



# Create a list which contains every file name in "jsons" folder
json_list = os.listdir(JSON_DIR)

""" tqdm Example Start"""

iterator_example = range(1000000)

for i in tqdm.tqdm(iterator_example):
    pass

""" rqdm Example End"""


# For every json file
for json_name in tqdm.tqdm(json_list):

    # Access and open json file as dictionary
    json_path = os.path.join(JSON_DIR, json_name)
    json_file = open(json_path, 'r')

    # Load json data
    json_dict = json.load(json_file)

    # Create an empty mask whose size is the same as the original image's size

    #########################################
    # CODE
    #########################################

    # For every objects
    for obj in json_dict["objects"]:
        # Check the objects ‘classTitle’ is ‘Freespace’ or not.
        if obj['classTitle']=='Freespace':

            #########################################
            # CODE
            #########################################

    # Write mask image into MASK_DIR folder

        #########################################
        # CODE
        #########################################