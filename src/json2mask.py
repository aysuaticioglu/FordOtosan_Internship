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
""" tqdm Example End"""

# For every json file
for json_name in tqdm.tqdm(json_list):
    json_path = os.path.join(JSON_DIR, json_name)
    try:
        with open(json_path, 'r', encoding='latin-1') as json_file:
            json_data = json_file.read()
            json_dict = json.loads(json_data)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError occurred in file: {json_name}")
        print("Error:", e)
        continue

    # Create an empty mask whose size is the same as the original image's size
    mask = np.zeros((json_dict["size"]["height"], json_dict["size"]["width"]), dtype=np.uint8)
    mask_path = os.path.join(MASK_DIR, json_name[:-9] + ".png")

    # For every objects
    for obj in json_dict["objects"]:
        # Check the objects ‘classTitle’ is ‘Freespace’ or not.
        if obj['classTitle'] == 'Freespace':
            mask = cv2.fillPoly(mask, np.array([obj['points']['exterior']]), color=1)

    # Write mask image into MASK_DIR folder
    cv2.imwrite(mask_path, mask.astype(np.uint8))
