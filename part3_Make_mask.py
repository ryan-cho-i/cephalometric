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

parser = argparse.ArgumentParser(description="Make a mask using image")
parser.add_argument('--base', '--b', help = 'the base address of the data')
parser.add_argument('--target_list', '--t', help = 'the target of model')
args = parser.parse_args()

Basic = args.base
Basic = os.path.abspath(Basic)
t_list=args.target_list

# Designate the list 

target_list = ListOfTarget (t_list)

# Mask

def make_mask (target_list, Input_Address, image_address, json_address, Output_Address) : 

    # Step 1. Read the image   

    img = cv2.imread(image_address)
    a,b,c = img.shape
    name1 = image_address
    sub1 = Input_Address
    sub2 ="/"
    name2 = name1.replace(sub1, '')
    fname = name2.replace(sub2, '')

    img = np.zeros((a,b))

    # Step 2. Extract the coordinate from JSON File 

    i = 0 
    while i in range(len(target_list)) : 

        d = extractor (json_address, target_list [i])
        pts = np.array(d, np.int32)

        # Step 3. Fill the Mask
        i += 1 
        img = cv2.fillPoly(img, [pts], i) 

    os.chdir(Output_Address) 
    cv2.imwrite(fname, img) 

def convert2mask (Basic, folder, t_list, target_list): 

    # The Address

    Input_Address = os.path.join(Basic, folder)
    Input_Address = os.path.abspath(Input_Address)
    print('Input Address is ' + Input_Address)

    Output = t_list + "M_" + folder
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

        make_mask(target_list, Input_Address, image_address[j], json_address[j], Output_Address) 

        i += 1
        print_progress(i,len(image_address))

    # Remove the json file 

    if t_list == "g3":
        for name in sorted(glob.glob(Input_Address + '/*.json')):
            os.remove(name)

# Convert

folder_name = ['tests', 'train', 'valid']

for folder in folder_name:      

    print ("Converting the file")
    convert2mask (Basic = Basic, folder = folder, t_list=t_list, target_list = target_list)
