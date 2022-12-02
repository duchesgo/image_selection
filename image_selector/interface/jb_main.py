import warnings
warnings.filterwarnings("ignore")

import os
from google.cloud import storage
import numpy as np

from image_selector.models.model_3_distortion_score.main_model_3_distortion_score import pred_model_3_distortion_score
from image_selector.models.model_face_detection.main_model_face_detection import face_detecting

def get_data_chunk(dataset="SPAC",data_status="processed",start=1,finish=10):
    """return chunk of data on format X_list,name_list, from image n°start to n°finish
       the X in Xlist are nparray_image, normalized, img is a list of name"""

    project_name="le-wagon-image-selection"
    bucket_name="image_selection"
    spac_bucket_path=f"data_{data_status}/{dataset}/"

    storage_client = storage.Client(project_name)
    bucket = storage_client.get_bucket(bucket_name)
    blobs=list(bucket.list_blobs(prefix=spac_bucket_path))

    X_list=[]
    name_list=[]


    for blob in blobs[start:finish]:
        local_byte=blob.download_as_bytes(raw_download=False)
        trunc_byte=local_byte[128:]
        img_array=np.frombuffer(trunc_byte,dtype="uint8")
        img_array=img_array.reshape(512,-1,3)

        X_list.append(img_array)

        blob_string=str(blob)
        img_name=blob_string[44:53]
        name_list.append(img_name)


    return X_list,name_list

def scoring_pipeline(dataset="SPAC",data_status="processed",start=1,finish=10):

    print(f"download data from index {start} to {finish}")
    X_list,name_list=get_data_chunk(dataset=dataset,data_status=data_status,start=start,finish=finish)

    dic_list=[]

    if len(X_list)!=len(name_list):
        print ("Error : different lenght in image list and name list")

    for index in range(len(X_list)):
        print(f"scoring image {name_list[index]}")
        master_dic={}
        master_dic["initial_image"]=X_list[index]
        master_dic["image_name"]=name_list[index]

        master_dic=pred_model_3_distortion_score(master_dic)

        dic_list.append(master_dic)

    print(dic_list[0]["image_name"])
    print(dic_list[0]["MOS"])

    print(dic_list[2]["image_name"])
    print(dic_list[2]["MOS"])

    print(dic_list[4]["image_name"])
    print(dic_list[4]["MOS"])

if __name__=="__main__":
    scoring_pipeline()
