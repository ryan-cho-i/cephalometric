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

    contours =[] 
    for i in range (len (coordi)) :
        contours.append(coordi[i])

    final_contours =[]
    for j in range (len(contours)):
        for k in range(len(contours[j])) :
            final_contours.append(contours[j][k].tolist())

    final_contours = [coordi for coordi in final_contours if len(coordi) >= 10]    

    return final_contours

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
                coordinates ["Cranial Fossa"] = extract_coordinates(pr_mask[i])
            elif i == 1 :
                coordinates ["Symphysis"] = extract_coordinates(pr_mask[i])
            elif i == 2 :
                coordinates ["Nasal Bone"] = extract_coordinates(pr_mask[i])

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

# Connect two regions

def find_point (list_ex_1, list_ex_2) :
    
    list_num_1=[]
    for i in range(len(list_ex_1)):
        list_num_1.append(list_ex_1[i][0][0])

    list_num_2=[]
    for i in range(len(list_ex_2)):
        list_num_2.append(list_ex_2[i][0][0])

    list_num_np_1 = np.array(list_num_1)
    list_num_np_2 = np.array(list_num_2)

    if abs(list_num_np_1.max()-list_num_np_2.min()) > abs(list_num_np_1.min()-list_num_np_2.max()) :
        standard1 = list_num_np_1.min()
        standard2 = list_num_np_2.max()
    else : 
        standard1 = list_num_np_1.max()
        standard2 = list_num_np_2.min()

    index1 = list_num_1.index(standard1)
    index2 = list_num_2.index(standard2)

    start_point = list_ex_1[index1][0]
    end_point = list_ex_2[index2][0]

    return tuple(start_point), tuple(end_point)
    
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
            target=[]
            pts=[]
            for i in range(len(target_list)) : 
                target.append(json_data[target_list[i]])
            
            if len(target[0]) == 2 : 
                pts.append(np.array(target[0][0], np.int32))
                pts.append(np.array(target[0][1], np.int32))       
                for i in range(1, len(target)) :
                    pts.append(np.array(target[i][0], np.int32))  
                    
            else :
                for i in range(len(target)) :
                    pts.append(np.array(target[i][0], np.int32))

        # bring the image

        path = image_path
        image = cv2.imread(path)
        image = cv2.resize(image, dsize=(480, 480), interpolation=cv2.INTER_AREA)

        # draw 

        for j in range(len(pts)) : 
            image = cv2.polylines(image, [pts[j]], isClosed = True, color = (0,255,0), thickness = 1)

        if len(pts) == 8 :
            start_point, end_point = find_point (pts[0], pts[1])
            image = cv2.line(image, start_point, end_point, (0,255,0), 1)

        # write the comment on the images     

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'Ground_Truth', (40,50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, 'Prediction', (40,90), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

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

print("Json file Upload")
upload_folder (AD, ID, PS, json_dir, "soo", "ceph_result_json") 

print("Image file Upload")
upload_folder (AD, ID, PS, images_dir, "soo", "ceph_result_images") 


