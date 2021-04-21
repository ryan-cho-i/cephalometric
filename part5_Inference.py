from pre_utils import * 
from pre_datasets import * 
from pre_monitor import *
from pre_model import build_model
from pre_upload import upload_file

import segmentation_models_pytorch as smp
import openpyxl 
import time
import argparse
import os 
import psutil
from minio import Minio
import torch
from torch.utils.data import DataLoader
import numpy as np

# Calculate the initial

start = time.time()

first_mem = psutil.virtual_memory()
initial_cpu = first_mem.used / (2 ** 20)

record_file_first = "first.txt"
os.system("gpustat > " + record_file_first)
initial_gpu = gpu_max(record_file_first)

# Argument Parsing 

parser = argparse.ArgumentParser(description="Infer the result")
parser.add_argument('--base', '--b', help = 'the base address of the data')
parser.add_argument('--architecture', '--a', help = 'the Architecture of model')
parser.add_argument('--encoder', '--e', help = 'the Encoder of model')
args = parser.parse_args()

Basic = args.base
Basic = os.path.abspath(Basic)
archi = args.architecture
encoder = args.encoder

record = []
record.append(archi)
record.append(encoder)

# Directory

x_test_dir = os.path.join(Basic, 'tests')
x_test_dir = os.path.abspath(x_test_dir)

name1 = 'g1' + "M_" + "tests"
y_test_dir1 = os.path.join(Basic, name1)
y_test_dir1 = os.path.abspath(y_test_dir1)

name2 = 'g2' + "M_" + "tests"
y_test_dir2 = os.path.join(Basic, name2)
y_test_dir2 = os.path.abspath(y_test_dir2)

name3 = 'g3' + "M_" + "tests"
y_test_dir3 = os.path.join(Basic, name3)
y_test_dir3 = os.path.abspath(y_test_dir3)

name1 = 'g1' + archi + encoder + 'best_model.pth'
model_dir1 = os.path.join(Basic, name1)
model_dir1 = os.path.abspath(model_dir1)

name2 = 'g2' + archi + encoder + 'best_model.pth'
model_dir2 = os.path.join(Basic, name2)
model_dir2 = os.path.abspath(model_dir2)

name3 = 'g3' + archi + encoder + 'best_model.pth'
model_dir3 = os.path.join(Basic, name3)
model_dir3 = os.path.abspath(model_dir3)

# Inference Function 

def inference (best_model1, best_model2, best_model3, test_dataset1, test_dataset2, test_dataset3, DEVICE) : 

    print("Inferencing the Result using GPU and CPU")

    prob=[[], [], [], [], [], [], []]

    m = len(test_dataset1)
    for n in range(m):

        print_progress(n,m)

        # Group 1. 

        image, gt_mask = test_dataset1[n]
        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model1.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        for i in range(len(['Cranial Fossa', 'Symphysis',  'Nasal Bone'])) :
            if i == 0 :
                prob[0].append(my_IOU_score (gt_mask[i], pr_mask[i]))
            elif i == 1 :
                prob[1].append(my_IOU_score (gt_mask[i], pr_mask[i]))
            elif i == 2 :
                prob[2].append(my_IOU_score (gt_mask[i], pr_mask[i]))

        # Group 2. 

        image, gt_mask = test_dataset2[n]
        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model2.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        for i in range(len(['Maxilla', 'Pterygomaxillary Fissure', 'Orbit'])) :
            if i == 0 :
                prob[3].append(my_IOU_score (gt_mask[i], pr_mask[i]))
            elif i == 1 :
                prob[4].append(my_IOU_score (gt_mask[i], pr_mask[i]))
            elif i == 2 :
                prob[5].append(my_IOU_score (gt_mask[i], pr_mask[i]))

        # Group 3. 

        image, gt_mask = test_dataset3[n]
        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model3.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        prob[6].append(my_IOU_score (gt_mask, pr_mask))

        if n == 0 :
            first = time.time()

    done = time.time()

    return first, done, m, prob

########## Practice ##########

# Basic Setting 

ACTIVATION = 'sigmoid' 
DEVICE = 'cuda'
ENCODER_WEIGHTS = 'imagenet'

loss = smp.utils.losses.DiceLoss()

metrics = [smp.utils.metrics.IoU(threshold=0.5),]

# Model Setting 

model1, preprocessing_fn1 = build_model (Architecture = archi, encoder = encoder, weights = ENCODER_WEIGHTS, CLASSES = ['Cranial Fossa', 'Symphysis',  'Nasal Bone'], activation = ACTIVATION)

model2, preprocessing_fn2 = build_model (Architecture = archi, encoder = encoder, weights = ENCODER_WEIGHTS, CLASSES = ['Maxilla', 'Pterygomaxillary Fissure', 'Orbit'], activation = ACTIVATION)

model3, preprocessing_fn3 = build_model (Architecture = archi, encoder = encoder, weights = ENCODER_WEIGHTS, CLASSES = ['Mandible'], activation = ACTIVATION)

# load best saved checkpoint

best_model1 = torch.load(model_dir1, map_location=torch.device(DEVICE))

best_model2 = torch.load(model_dir2, map_location=torch.device(DEVICE))

best_model3 = torch.load(model_dir3, map_location=torch.device(DEVICE))

# create test dataset

test_dataset1 = Dataset1(
    x_test_dir, 
    y_test_dir1, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn1),
    classes=['Cranial Fossa', 'Symphysis',  'Nasal Bone'],
)

test_dataset2 = Dataset2(
    x_test_dir, 
    y_test_dir2, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn2),
    classes=['Maxilla', 'Pterygomaxillary Fissure', 'Orbit'],
)

test_dataset3 = Dataset3(
    x_test_dir, 
    y_test_dir3, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn3),
    classes=['Mandible'],
)

# Inference 

name = archi + encoder + "gpu.txt"

cpu, first, done, m, prob = Monitor (inference, best_model1, best_model2, best_model3, test_dataset1, test_dataset2, test_dataset3, DEVICE, name)

# Result 

Latency = first - start
record.append(Latency)

Throughput = (done - start) / m 
record.append(Throughput)

CPU = int(max(cpu)) - int(initial_cpu)
record.append(CPU)

GPU = int(gpu_max(name)) - int(initial_gpu)
record.append(GPU)

prob = np.array(prob)
record.append(np.mean(prob))

for i in range(len(prob)) :
    record.append(np.mean(prob[i]))
    record.append(np.var(prob[i]))
    record.append(np.max(prob[i]))
    record.append(np.min(prob[i]))

print ()
print_inf (archi, encoder, Latency, Throughput, CPU, GPU, prob) 

# Store the data 

name_excel = "result.xlsx"
excel = os.path.join(Basic, name_excel)
load_wb = openpyxl.load_workbook(excel)
load_ws = load_wb['result1(GPU & CPU)']
load_ws.append(record)
load_wb.save(excel)
