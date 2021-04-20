from minio import Minio
import os 

# Upload the file 

def upload_file (AD, ID, PS, file_path, basket_name, folder_in_MinIO) :

    print("Upload the file to Minio")
    
    client = Minio(AD, access_key=ID, secret_key=PS, secure=True)

    file_name = file_path.split('/')
    minIO_address = os.path.join(folder_in_MinIO, file_name[-1])
    client.fput_object(basket_name, minIO_address, file_path)

# Upload the folder 

def upload_folder (AD, ID, PS, folder_path, basket_name, folder_in_MinIO) :

    print("Upload the folder to Minio")
    
    client = Minio(AD, access_key=ID, secret_key=PS, secure=True)

    file_name = os.listdir (folder_path)

    for i in range (len(file_name)) : 

        file_path = os.path.join(folder_path, file_name[i])
        minIO_address = os.path.join(folder_in_MinIO, file_name[i])
        client.fput_object(basket_name, minIO_address, file_path)


