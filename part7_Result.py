from pre_utils import * 
from pre_datasets import *
from pre_model import build_model
from pre_upload import upload_folder

import segmentation_models_pytorch as smp
import openpyxl 
import time
import argparse
import os 
import psutil
import cv2

# Argument Parsing 

parser = argparse.ArgumentParser(description="Infer and Upload the data based on the best model")
parser.add_argument('--base', '--b', help = 'the base address of the data')
parser.add_argument('--architecture', '--a', help = 'the Architecture of model')
parser.add_argument('--encoder', '--e', help = 'the Encoder of model')
parser.add_argument('--ground_image', '--in', help = 'the ground truth images')

parser.add_argument('--address', '--m', help = 'the address of MinIO')
parser.add_argument('--id', '--i', help = 'ID')
parser.add_argument('--password', '--p', help = 'Password')
args = parser.parse_args()

Basic = args.base
Basic = os.path.abspath(Basic)
archi = args.architecture
encoder = args.encoder
ground_image = args.ground_image

AD=args.address
ID=args.id
PS=args.password

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

json_dir = os.path.join(Basic, 'json_file')
json_dir = os.path.abspath(json_dir)
createFolder(json_dir)

images_dir = os.path.join(Basic, 'images_file')
images_dir = os.path.abspath(images_dir)
createFolder(images_dir)




# extract the coordinates 

def extract_coordinates (pr_mask):

    im = pr_mask.astype("uint8")
    coordi = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)        
    contours = coordi[0][0].tolist()

    return contours 

# Store the result of inference in forms of json 

def inference_json (x_test_dir, best_model1, best_model2, best_model3, test_dataset1, test_dataset2, test_dataset3, json_dir, DEVICE) : 

    print("Inferencing the Result and store the data into json")

    # Name

    name = os.listdir(x_test_dir)

    # inference

    m = len(name)
    for n in range(m):

        print_progress(n,m)

        coordinates = dict ()
        filename = name[n] + '.json'
        file_path = os.path.join(json_dir, filename)

        # Group 1. 

        image, gt_mask = test_dataset1[n]
        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model1.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        for i in range(len(['Cranial Fossa', 'Symphysis',  'Nasal Bone'])) :
            if i == 0 :
                coordinates ["Cranial Fossa"] = extract_coordinates (pr_mask[i])
            elif i == 1 :
                coordinates ["Symphysis"] = extract_coordinates (pr_mask[i])
            elif i == 2 :
                coordinates ["Nasal Bone"] = extract_coordinates (pr_mask[i])

        # Group 2. 

        image, gt_mask = test_dataset2[n]
        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model2.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        for i in range(len(['Maxilla', 'Pterygomaxillary Fissure', 'Orbit'])) :
            if i == 0 :
                coordinates ["Maxilla"] = extract_coordinates(pr_mask[i])
            elif i == 1 :
                coordinates ["Pterygomaxillary Fissure"] = extract_coordinates(pr_mask[i])
            elif i == 2 :
                coordinates ["Orbit"] = extract_coordinates(pr_mask[i])

        # Group 3. 

        image, gt_mask = test_dataset3[n]
        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model3.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        coordinates ["Mandible"] = extract_coordinates(pr_mask)
        
        # Store the coordinate into json 

        with open(file_path, 'w') as make_json:
            json.dump(coordinates, make_json)

# Draw the polyline on the image

def draw_polyline_predicted (image_dir, json_dir, Output_Address) : 

    print("Draw the line on the image")

    # Name

    name = os.listdir(image_dir)

    json_name = []
    for n in range(len(name)):
        json_name.append(name[n] + ".json") 

    # inference

    m = len(name)
    for n in range(m):

        print_progress(n,m)

        image_path = os.path.join (image_dir, name[n])
        json_path = os.path.join (json_dir, json_name[n])

        # bring the coordinate data 

        with open (json_path, 'r') as file :
            json_data = json.load(file)

            target_list = ['Cranial Fossa', 'Symphysis',  'Nasal Bone', 'Maxilla', 'Pterygomaxillary Fissure', 'Orbit', 'Mandible']
            target=[[], [], [], [], [], [], []]
            pts=[[], [], [], [], [], [], []]

            for i in range (len(target_list)) : 
                target [i] = json_data [target_list[i]]
                pts[i] = np.array(target[i])

        # bring the image

        path = image_path
        image = cv2.imread(path)
        image = cv2.resize(image, dsize=(480, 480), interpolation=cv2.INTER_AREA)

        color = [(47,30,221), (53,176,235), (203,162,6), (168, 182, 33), (188,214,244), (31,23,127), (63,141,247)]

        # draw 

        for j in range(len(target_list)) : 
            image = cv2.polylines(image, [pts[j]], isClosed = True, color = color[j], thickness = 1)

        # write the comment on the images     

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'Ground_Truth', (40,50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        os.chdir(Output_Address) 
        cv2.imwrite(name[n], image) 

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

best_model1 = torch.load(model_dir1)

best_model2 = torch.load(model_dir2)

best_model3 = torch.load(model_dir3)

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

# After inferencing, store data into json

inference_json (x_test_dir, best_model1, best_model2, best_model3, test_dataset1, test_dataset2, test_dataset3, json_dir, DEVICE) 
print()

# Draw the predicted area 

ground_dir = os.path.join (Basic, ground_image)
draw_polyline_predicted (ground_dir, json_dir, images_dir) 
print()

# Upload the data 

upload_folder (AD, ID, PS, images_dir, "soo", "ceph_result_images") 
upload_folder (AD, ID, PS, json_dir, "soo", "ceph_result_json") 



