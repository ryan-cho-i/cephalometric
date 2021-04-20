# Project Explaination

### 0. Description of this project 

1) This project is to do Semantic Segmentation for Cephalometric Radiograph.

### 1. Part 1. Download the data 

1) Download the data from the database. 

2) Data are constituded by image and json file.

  > Image file is Cephalometric Radiograph.

  > Json file contains the coordinates about various bones.

### 2. Part 2. Draw the line on the images. 

1) Extract the meaningful coordinates from Json file. 

  > Therefore, there are 7 classes to draw.   

2) Draw the polylines of 7 classes on the image.

### 3. Part 3. (Labeling the Data) Make a mask of image.

1) Make masks of the image.

  > Cautions : It is needed to divide 7 classes into 3 groups without overlapping. 
  
  > Hence, I divided into 3 groups.
  >> (1) Group 1. 'Cranial Fossa', 'Symphysis',  'Nasal Bone'
    (2) Group 2. 'Maxilla', 'Pterygomaxillary Fissure', 'Orbit'
    (3) Group 3. 'Mandible'

### 4. Part 4. Training the models.

1) There are 3 Groups to be predicted. This is why it is required to design 3 models. 
  
  > For example, if you want to design the model using Unet as architecture and resnet18 as encoder, you give the model the name 'g1_Unet_Resnet18'. 
    Because there are two models except this. (g2_Unet_Resnet18, g3_Unet_Resnet18) 

2) When the dataset is coded, it is essential to take care of 'CLASSES'. 

### 5. Part 5. "Inferencing the Result using GPU and CPU"

1) This code is intended to calculate the Latency, Throughput, the usage of CPU and GPU, and accuracy of the Models. 

### 6. Part 6. "Inferencing the Result using only CPU"

1) Cautions : When the CUDA is converted into the CPU at DEVICE, it is necessary to add 'map_location=torch.device(DEVICE)' to torch.load 
  
  ex) best_model1 = torch.load(model_dir1, map_location=torch.device(DEVICE))

### 7. Part 7. Result

1) This is coded to predict the output in forms of Json file, Images. 
(Blue Color means ground truth and Other colors mean the predictions)

![ES_Vatech_PaX-i_61141](https://user-images.githubusercontent.com/78337318/115352138-f7b5a900-a1f1-11eb-948b-a85d447a8b39.png)

# Conclusion



