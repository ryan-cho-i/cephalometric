from minio import Minio
import os 
import argparse

from pre_utils import print_progress
from pre_utils import CreateExel

# Argument  

parser = argparse.ArgumentParser(description="Download the Data")
parser.add_argument('--base', '--b', help = 'the base address of the data')
parser.add_argument('--address', '--m', help = 'the address of MinIO')
parser.add_argument('--id', '--i', help = 'ID')
parser.add_argument('--password', '--p', help = 'Password')
parser.add_argument('--busket_name', '--bus', help = 'the name of the busket to download')
args = parser.parse_args()

Basic = args.base
Basic = os.path.abspath(Basic)
AD=args.address
ID=args.id
PS=args.password
busket_name=args.busket_name

# Download the Data 

def download (busket_name, client, Basic, folder) :

    Input_Address = os.path.join(Basic, folder)
    Input_Address = os.path.abspath(Input_Address)
    print('Input Address is ' + Input_Address)

    objects = client.list_objects(busket_name, prefix=folder, recursive=True)
    ob=0
    m=0
    for ob in objects :
        m +=1 

    objects = client.list_objects(busket_name, prefix=folder, recursive=True)
    k=0
    print_progress(k,m)
    for obj in objects:
        client.fget_object(busket_name, obj.object_name, Basic + "/" + obj.object_name)        
        k+=1
        print_progress(k,m)

# Connect the MinIO server

client = Minio(AD, access_key=ID, secret_key=PS, secure=True)

folder_name = ['tests', 'train', 'valid']

# download

for folder in folder_name:      
    
    print ("Downloading the file")
    download (busket_name = busket_name, client = client, Basic = Basic, folder = folder)    

# Create the Exel file to record
    
name_excel = "result"
CreateExel (Basic = Basic, name = name_excel, sheet1= 'result1(GPU & CPU)', sheet2='result2(Only CPU)')    