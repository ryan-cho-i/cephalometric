import os 
import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
import glob
import cv2 

from pre_utils import ListOfTarget
from pre_utils import createFolder
from pre_utils import print_progress
from pre_utils import CreateExel
from pre_extract import extractor

# Argument  

parser = argparse.ArgumentParser(description="Draw the line on the image")
parser.add_argument('--base', '--b', help = 'the base address of the data')
args = parser.parse_args()

Basic = args.base
Basic = os.path.abspath(Basic)

# Mask

def draw_line (Input_Address, image_address, json_address, Output_Address) : 

    # Step 1. Read the image   

    img = cv2.imread(image_address)
    a,b,c = img.shape
    name1 = image_address
    sub1 = Input_Address
    sub2 ="/"
    name2 = name1.replace(sub1, '')
    fname = name2.replace(sub2, '')

    # Step 2. Extract the coordinate from JSON File 

    target_list = ['Cranial Fossa', 'Symphysis',  'Nasal Bone', 'Maxilla', 'Pterygomaxillary Fissure', 'Orbit', 'Mandible']
    i = 0 
    while i in range(len(target_list)) : 

        d = extractor (json_address, target_list [i])
        pts = np.array(d, np.int32)

        # Step 3. Fill the Mask
        i += 1 
        img = cv2.polylines(img, [pts], isClosed = True, color = (255, 0, 0), thickness = 2)

    os.chdir(Output_Address) 
    cv2.imwrite(fname, img) 

def convert2line (Basic, folder): 

    # The Address

    Input_Address = os.path.join(Basic, folder)
    Input_Address = os.path.abspath(Input_Address)
    print('Input Address is ' + Input_Address)

    Output = "Draw_Image_" + folder
    Output_Address = os.path.join(Basic, Output)
    Output_Address = os.path.abspath(Output_Address)
    print('Output Address is ' + Output_Address)
    createFolder(Output_Address)

    # Arrange the file

    image_address = []
    json_address = []

    list1 = Input_Address + "/*.BMP"
    file_list1 = glob.glob(list1)

    list2 = Input_Address + "/*.bmp"
    file_list2 = glob.glob(list2)

    image_address = file_list1 + file_list2   

    for n in range(len(image_address)):
        json_address.append(image_address[n] + ".json") 

    # Convert

    i=0
    print_progress(i,len(image_address))

    for j in range(len(image_address)):

        # Create the Mask

        draw_line (Input_Address, image_address[j], json_address[j], Output_Address) 

        i += 1
        print_progress(i,len(image_address))

# Convert

folder_name = ['tests', 'train', 'valid']

for folder in folder_name:      
    
    print ("Draw the line on the image")

    convert2line (Basic = Basic, folder=folder)
