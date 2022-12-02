import warnings
warnings.filterwarnings("ignore")

import os
from google.cloud import storage
import numpy as np
import cv2

from image_selector.models.model_3_distortion_score.main_model_3_distortion_score import pred_model_3_distortion_score
from image_selector.models.model_face_detection.main_model_face_detection_dev import face_detecting

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
        img_name=blob_string
        name_list.append(img_name)


    return X_list,name_list

def scoring_pipeline(dataset="SPAC",data_status="processed",start=1,finish=8):

    print(f"download data from index {start} to {finish}")
    X_list,name_list=get_data_chunk(dataset=dataset,data_status=data_status,start=start,finish=finish)

    dic_list=[]

    if len(X_list)!=len(name_list):
        print ("Error : different lenght in image list and name list")

    for index in range(len(X_list)):
        print(f"scoring image {name_list[index]}")
        master_dic={}
        master_dic["image_array"]=X_list[index]
        master_dic["image_name"]=name_list[index]

        scaleFactor_tp=1.1   # compense pour les visages plus ou moins proches de l'objectif
        minNeighbors_tp=5    # Nb de voisins pour reconnaître un objet ; pas clair
        minSize_tp=(30, 30)  # Taille de chaque fenêtre précédente ; pas clair
        cascade_path_tp = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"

        master_dic=face_detecting(master_dic=master_dic, cascade_path=cascade_path_tp, scaleFactor=scaleFactor_tp, minNeighbors=minNeighbors_tp, minSize=minSize_tp, visualize=False)

        master_dic=pred_model_3_distortion_score(master_dic)

        dic_list.append(master_dic)

    for i in range (len(dic_list)):
        print(dic_list[i]["image_name"])
        print(dic_list[i]["MOS"])
        print(dic_list[i]['nb_faces'])


if __name__=="__main__":

    scaleFactor_tp=1.1   # compense pour les visages plus ou moins proches de l'objectif
    minNeighbors_tp=5    # Nb de voisins pour reconnaître un objet ; pas clair
    minSize_tp=(30, 30)  # Taille de chaque fenêtre précédente ; pas clair
    surface_visage_min_in_image = 0.004  # Surface minimale d'un visage pour être recevable
    cascade_path_tp = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    scoring_pipeline(dataset="TYPOLOGIE_CLUSTER",data_status="processed",start=1,finish=10)
    #scoring_pipeline(dataset="SPAC",data_status="processed",start=1,finish=2)
