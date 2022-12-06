from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from google.cloud import storage
from tensorflow.image import resize_with_pad
from tensorflow import cast

target_path=os.environ.get("SPAC_TARGET_PATH")
target_df=pd.read_excel(target_path)


def get_png_image(dataset="SPAC",data_status="raw",image_name="00001.jpg"):

    """This function return the image as a tensor of shape (H,L,3), of float between 0 and 1"""

    folder_path=os.environ.get("LOCAL_PROJECT_PATH")+ "local_data/" + f"data_{data_status.lower()}/{dataset.upper()}"

    if folder_path[-1]=="/":
        image_path=folder_path+image_name
    else:
        image_path=folder_path+"/"+image_name

    img=load_img(image_path)
    img_array=img_to_array(img)
    img_array=img_array/255

    return img_array

def get_X_SPAC(dataset="SPAC",data_status="raw",first_index=None,last_index=None):
    """This function return a list X of np array store on loca_data"""

    folder_path=os.environ.get("LOCAL_PROJECT_PATH")+ "local_data/" + f"data_{data_status.lower()}/{dataset.upper()}"

    X_list=[]

    accepted_img_type=["jpg","png","bmp"]

    img_list=os.listdir(folder_path)
    img_list=[img for img in img_list if img[-3:] in accepted_img_type]

    if first_index==None:
        first_index=0

    if last_index==None:
        last_index=len(img_list)

    if first_index<0 or first_index>len(img_list):
        print(f"Wrong first index, {first_index} is given but we needed to be between 0 and {len(img_list)}")
        return None
    if last_index<first_index or last_index>len(img_list):
        print(f"Wrong last index, {last_index} is given but we needed to be between {first_index} and {len(img_list)}")
        return None

    for img in img_list[first_index:last_index]:
        X=get_png_image(dataset=dataset,data_status=data_status,image_name=img)
        X_list.append(X)

    return X_list,img_list

def resize_X(X_list,hight=512,width=900):
    """This function return a list a resized tensor-image given a list of tensor-image"""
    #X_resize=[resize_with_pad(x,512,900) for x in X_list]
    X_resize=[]
    for X in X_list:
        shape_1=int(X.shape[1]*(512/X.shape[0]))
        img_resize=resize_with_pad(X,512,shape_1)
        X_resize.append(img_resize)


    return X_resize

def download_data(dataset="SPAC",status="RAW",first_index=1,last_index=10):
    project_name="le-wagon-image-selection"
    bucket_name="image_selection"
    spac_bucket_path=f"data_{status.lower()}/{dataset.upper()}/"
    file_storage=os.environ.get("LOCAL_PROJECT_PATH")+ "local_data"

    storage_client = storage.Client(project_name)
    bucket = storage_client.get_bucket(bucket_name)
    blobs=list(bucket.list_blobs(prefix=spac_bucket_path))

    for blob in blobs[first_index:last_index]:
        destination_file_name=file_storage+"/"+blob.name
        f=open(destination_file_name,"wb")
        blob.download_to_file(f)

    return None

def upload_data(dataset="SPAC",data_status="processed"):
    project_name="le-wagon-image-selection"
    bucket_name="image_selection"
    PROCESSED_LOCAL_PATH=os.environ.get("LOCAL_PROJECT_PATH")+ "local_data/" + f"data_{data_status.lower()}/{dataset.upper()}/"
    PROCCESSED_SPAC_BUCKET_PATH=f"data_{data_status.lower()}/{dataset.upper()}/"
    storage_client = storage.Client(project_name)
    bucket = storage_client.get_bucket(bucket_name)
    upload_img_list=os.listdir(PROCESSED_LOCAL_PATH)
    upload_img_list=[img for img in upload_img_list if img[-3:]=="npy"]

    for img in upload_img_list:
        blob=bucket.blob(PROCCESSED_SPAC_BUCKET_PATH+img)
        blob.upload_from_filename(PROCESSED_LOCAL_PATH+img)

    return None

def store_data_local(tensor,name,dataset="SPAC",data_status="processed"):
    array=np.array(tensor)
    filepath=os.environ.get("LOCAL_PROJECT_PATH")+ "local_data/" + f"data_{data_status.lower()}/{dataset.upper()}/"+name

    np.save(filepath,array)
    return None

def clean_raw_data(dataset="SPAC",data_status="raw"):

    folder_path=os.environ.get("LOCAL_PROJECT_PATH")+ "local_data/" + f"data_{data_status.lower()}/{dataset.upper()}/"

    img_list=os.listdir(folder_path)

    for img in img_list:
        filepath=folder_path+img
        os.remove(filepath)

    return None

def clean_processed_data(dataset="SPAC",data_status="processed"):

    folder_path=os.environ.get("LOCAL_PROJECT_PATH")+ "local_data/" + f"data_{data_status.lower()}/{dataset.upper()}/"

    img_list=os.listdir(folder_path)

    for img in img_list:
        filepath=folder_path+img
        os.remove(filepath)

    return None

def downsize_tensor_list(X_list):
    return [cast(x*255,"uint8") for x in X_list]

def preprocess_pipeline(dataset="SPAC",first_index=1,last_index=15):
    print("start downloading")
    print(f"downloading from index {first_index} to {last_index}")
    download_data(dataset=dataset,status="RAW",first_index=first_index,last_index=last_index)
    print("start charging as np array")
    X_list,name_list=get_X_SPAC(dataset=dataset,data_status="raw")

    print("start resizing")
    X_list=resize_X(X_list,hight=512,width=900)

    print("start downsizing")
    X_list=downsize_tensor_list(X_list=X_list)

    print("start writing on DD")

    for i,X in enumerate(X_list):
        print(i)
        store_data_local(X,name_list[i],dataset=dataset,data_status="processed")

    print("start uploading")
    upload_data(dataset=dataset,data_status="processed")
    print("cleaning")
    clean_raw_data(dataset=dataset,data_status="raw")
    clean_processed_data(dataset=dataset,data_status="processed")


if __name__=="__main__":

    for d in range (0,42):
        first=10*d+1
        last=10*(d+1)
        preprocess_pipeline(dataset="DATA_TEST",first_index=first,last_index=last+1)

    #preprocess_pipeline(dataset="TYPOLOGIE_CLUSTER",first_index=51,last_index=59)

    #download_data(dataset="TYPOLOGIE_CLUSTER",status="processed",first_index=1,last_index=10)
