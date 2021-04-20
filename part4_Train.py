import os 
import torch
import segmentation_models_pytorch as smp
import argparse
from torch.utils.data import DataLoader

from pre_utils import ListOfTarget
from pre_utils import ListOfDatasets
from pre_utils import ListOfClass
from pre_model import build_model
from pre_datasets import get_preprocessing
from pre_upload import upload_file

# Argument Parsing 

parser = argparse.ArgumentParser(description="Train the Model")
parser.add_argument('--base', '--b', help = 'the base address of the data')
parser.add_argument('--architecture', '--a', help = 'the Architecture of model')
parser.add_argument('--encoder', '--e', help = 'the Encoder of model')
parser.add_argument('--batch_size', '--bat', type=int, help = 'the batch_size of training')
parser.add_argument('--epochs', '--ep', type=int, help = 'the epochs of training')
parser.add_argument('--target_list', '--t', help = 'the target of model')
parser.add_argument('--address', '--m', help = 'the address of MinIO')
parser.add_argument('--id', '--i', help = 'ID')
parser.add_argument('--password', '--p', help = 'Password')

args = parser.parse_args()

Basic = args.base
Basic = os.path.abspath(Basic)
archi = args.architecture
encoder = args.encoder
batch_size = args.batch_size
epochs = args.epochs
t_list=args.target_list
AD=args.address
ID=args.id
PS=args.password

# Designate the list 

target_list = ListOfTarget (t_list)
Dataset = ListOfDatasets (t_list) 
CLASSES = ListOfClass (t_list)

# Directory

x_train_dir = os.path.join(Basic, 'train')
x_train_dir = os.path.abspath(x_train_dir)

name = t_list + "M_" + "train"
y_train_dir = os.path.join(Basic, name)
y_train_dir = os.path.abspath(y_train_dir)

x_valid_dir = os.path.join(Basic, 'valid')
x_valid_dir = os.path.abspath(x_valid_dir)

name = t_list + "M_" + "valid"
y_valid_dir = os.path.join(Basic, name)
y_valid_dir = os.path.abspath(y_valid_dir)

x_test_dir = os.path.join(Basic, 'tests')
x_test_dir = os.path.abspath(x_test_dir)

name = t_list + "M_" + "tests"
y_test_dir = os.path.join(Basic, name)
y_test_dir = os.path.abspath(y_test_dir)

name = t_list + archi + encoder + 'best_model.pth'
model_dir = os.path.join(Basic, name)
model_dir = os.path.abspath(model_dir)

# Train Function 

def train_model (x_train_dir, y_train_dir, x_valid_dir, y_valid_dir, model_dir, model, preprocessing_fn, CLASSES, DEVICE, loss, metrics, optimizer, batch_size, epochs) : 

    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    # train model for 40 epochs

    max_score = 0

    for i in range(0, epochs):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, model_dir)
            print('Model saved!')
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

# Basic Setting 

ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid' 
DEVICE = 'cuda'

loss = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5),]

# Model Setting 

model, preprocessing_fn = build_model (Architecture = archi, encoder = encoder, weights = ENCODER_WEIGHTS, CLASSES = CLASSES, activation = ACTIVATION)

optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001),])

# Train 

print ("Training the Model")
train_model (x_train_dir, y_train_dir, x_valid_dir, y_valid_dir, model_dir, model, preprocessing_fn, CLASSES, DEVICE, loss, metrics, optimizer, batch_size, epochs)
print ("Train is finished")

# Upload the data

upload_file (AD, ID, PS, model_dir, "soo", "ceph_trainig_result_parameter") 
