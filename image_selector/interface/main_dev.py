
import warnings
warnings.filterwarnings("ignore")

import math

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from google.cloud import storage
import numpy as np
import cv2
import re
import pickle
from image_selector.models.model_3_distortion_score.main_model_3_distortion_score import pred_model_3_distortion_score
from image_selector.models.model_2_face_detection.main_model_face_detection import face_detecting
from image_selector.models.model_4_face_quality.main_model_4 import pred_model_4
#from image_selector.models.model_1_clustering.getdata_model_1_clustering import img_to_ndarray
from image_selector.models.model_1_clustering.utils_model_1_clustering import preprocessed_X_clustering, extract_features, pca, clustering, dict_clustering
# for the VGG model
from keras.applications.vgg16 import VGG16
from keras.models import Model

def get_name_blob(image_name):
    regex='/[^/]*,'
    cleaned_name=re.search(regex,image_name)
    cleaned_name=cleaned_name.group(0)[1:-5]
    return cleaned_name

def store_one_dict(dataset="GROUP",data_status="categorized_dict",image_dict={"image_name":"test","image_array":0}):
    storing_file_path=SCORED_LOCAL_PATH=os.environ.get("LOCAL_PROJECT_PATH")+ "local_data/" + f"data_{data_status.lower()}/{dataset.upper()}/"

    dict_path=storing_file_path+image_dict["image_name"]+".pickle"

    storing_file=open(dict_path,"wb")
    pickle.dump(image_dict,storing_file)
    storing_file.close()

    return None

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

        blob_string=get_name_blob(str(blob))
        img_name=blob_string
        name_list.append(img_name)


    return X_list,name_list

def upload_pickle(dataset="GROUP",data_status="categorized_dict"):
    project_name="le-wagon-image-selection"
    bucket_name="image_selection"
    SCORED_LOCAL_PATH=os.environ.get("LOCAL_PROJECT_PATH")+ "local_data/" + f"data_{data_status.lower()}/{dataset.upper()}/"
    SCORED_SPAC_BUCKET_PATH=f"data_{data_status.lower()}/{dataset.upper()}/"
    storage_client = storage.Client(project_name)
    bucket = storage_client.get_bucket(bucket_name)
    upload_img_list=os.listdir(SCORED_LOCAL_PATH)
    upload_img_list=[img for img in upload_img_list if img[-7:]==".pickle"]

    for img in upload_img_list:
        blob=bucket.blob(SCORED_SPAC_BUCKET_PATH+img)
        blob.upload_from_filename(SCORED_LOCAL_PATH+img)

def clean_pickle(dataset="GROUP",data_status="categorized_dict"):

    folder_path=os.environ.get("LOCAL_PROJECT_PATH")+ "local_data/" + f"data_{data_status.lower()}/{dataset.upper()}/"

    img_list=os.listdir(folder_path)

    for img in img_list:
        filepath=folder_path+img
        os.remove(filepath)

    return None

def download_pickle(dataset="GROUP",status="categorized_dict",first_index=1,last_index=30):
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

def open_dict_pickle(dataset="GROUP",data_status="categorized_dict",image_name="test"):
    storing_file_path=SCORED_LOCAL_PATH=os.environ.get("LOCAL_PROJECT_PATH")+ "local_data/" + f"data_{data_status.lower()}/{dataset.upper()}/"

    dict_path=storing_file_path+image_name

    dict_image=pickle.load( open( dict_path,"rb") )

    return dict_image

def get_pickle(dataset="GROUP",data_status="categorized_dict",first_index=1,last_index=10):
    """This function return a list of dic_image from the hardrive"""

    folder_path=os.environ.get("LOCAL_PROJECT_PATH")+ "local_data/" + f"data_{data_status.lower()}/{dataset.upper()}"

    img_dic_list=[]

    accepted_type=["pickle"]
    name_list=os.listdir(folder_path)
    name_list=[name for name in name_list if name[-6:] in accepted_type]


    if first_index==None:
        first_index=0

    if last_index==None:
        last_index=len(name_list)

    for img in name_list[first_index:last_index]:
        X=open_dict_pickle(dataset=dataset,data_status=data_status,image_name=img)
        img_dic_list.append(X)

    return img_dic_list,name_list[first_index:last_index]

def exctract_features_chunk(dict_chunk):

    X=[dico["image_array"] for dico in dict_chunk]
    list_name=[dico["image_name"] for dico in dict_chunk]

    # Preprocessed X

    X_preprocessed = preprocessed_X_clustering(X)
    # Model VGG16 : load, remove the output layer and extract features
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = extract_features(X_preprocessed, model)

    return features

def save_features_on_disc(features,name_list,data_status="extracted_features",dataset="GROUP"):
    storing_file_path=os.environ.get("LOCAL_PROJECT_PATH")+ "local_data/" + f"data_{data_status.lower()}/{dataset.upper()}/"


    for no_img_dict in range(len(features)):
        features_path=storing_file_path+name_list[no_img_dict]
        storing_file=open(features_path,"wb")
        pickle.dump(features[no_img_dict],storing_file)
        storing_file.close()

    return None

def save_cluster_dic(dict_cluster,data_status="extracted_features",dataset="GROUP"):
    storing_file_path=os.environ.get("LOCAL_PROJECT_PATH")+ "local_data/" + f"data_{data_status.lower()}/CLUSTER_DICT/"
    cluster_path=storing_file_path+dataset
    storing_file=open(cluster_path,"wb")
    pickle.dump(dict_cluster,storing_file)
    storing_file.close()

def clustering_pipeline(dataset="GROUP",start=1,finish=50,chunk_size=10):

    #Download the image_dict on the hard-drive

    nb_chunk=math.ceil(finish/chunk_size)

    print("download chunk")

    for chunk in range (nb_chunk):
        first_index=(chunk*chunk_size)+1
        last_index=((chunk+1)*chunk_size)+1

        download_pickle(dataset=dataset,status="categorized_dict",first_index=first_index,last_index=last_index)

    #Extracting features and storing in hard drive

    #Accessing the number of donwloaded image in order to correct the number of chunk

    donwload_data_folder=os.environ.get("LOCAL_PROJECT_PATH")+ "local_data/" + f"data_categorized_dict/{dataset.upper()}"
    name_download_list=os.listdir(donwload_data_folder)

    nb_chunk=math.ceil(len(name_download_list)/chunk_size)


    print("extracting features")

    for chunk in range (nb_chunk):
        first_index=(chunk*chunk_size)
        last_index=((chunk+1)*chunk_size)

        img_dic_list,name_list=get_pickle(dataset=dataset,data_status="categorized_dict",first_index=first_index,last_index=last_index)

        features=exctract_features_chunk(img_dic_list)

        print(f"saving features from {first_index} to {last_index} ")

        save_features_on_disc(features,name_list,data_status="extracted_features",dataset=dataset)

    #cleaning image_dict store previously

    print("cleaning local image")

    clean_pickle(dataset=dataset,data_status="categorized_dict")

    #Fitting the PCA

    print("reducing dimension, PCA")

    dataset_features,list_names=get_pickle(dataset=dataset,data_status="extracted_features",first_index=1,last_index=None)

    pca_features = pca(dataset_features)

    #Calculating the right number of features, and transforming the features

    print("Clustering")

    nb_clusters, dict_clusters = clustering(pca_features)

    #Caluclating the clustering, returning a name/cluster

    dict_clusters_final = dict_clustering(dict_clusters, list_names)

    print(f"found {nb_clusters} clusters")

    print(dict_clusters_final)

    #Cleaning extracted features

    print("cleaning features")

    clean_pickle(dataset=dataset,data_status="extracted_features")

    #storing the cluster dictionnary

    print("saving cluster dict")

    save_cluster_dic(dict_cluster=dict_clusters_final,data_status="extracted_features",dataset=dataset)

def model_clustering(X_path):
    # Load X images and transform images into ndarray
    #X, list_names = img_to_ndarray(X_path)


    # Preprocessed X
    X_preprocessed = preprocessed_X_clustering(X)
    # Model VGG16 : load, remove the output layer and extract features
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = extract_features(X_preprocessed, model)


    # Reduce dimension features (pca)
    pca_features = pca(features)
    # Find cluster
    nb_clusters, dict_clusters = clustering(pca_features)
    # Dictionnaire à jour
    dict_clusters_final = dict_clustering(dict_clusters, list_names)
    return nb_clusters, dict_clusters_final

def scoring_pipeline(dataset="SPAC",data_status="processed",start=1,finish=20,chunk_size=10):

    nb_chunk=math.ceil(finish/chunk_size)

    for chunk in range(nb_chunk):

        first_index=(chunk*chunk_size)+1
        last_index=((chunk+1)*chunk_size)+1

        scoring_chunk(dataset=dataset,data_status=data_status,start=first_index,finish=last_index)
    pass

def scoring_chunk(dataset="SPAC",data_status="processed",start=1,finish=8):

    scaleFactor_tp=1.1   # compense pour les visages plus ou moins proches de l'objectif
    minNeighbors_tp=3    # Nb de voisins pour reconnaître un objet ; pas clair
    minSize_tp=(23, 23)  # Taille de chaque fenêtre précédente ; pas clair
    surface_visage_min_in_image = 0.002  # Surface minimale d'un visage pour être recevable
    cascade_path_tp = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"

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

        master_dic=pred_model_3_distortion_score(master_dic)

        master_dic=face_detecting(image_dict=master_dic, cascade_path=cascade_path_tp, scaleFactor=scaleFactor_tp, minNeighbors=minNeighbors_tp, minSize=minSize_tp, visualize=False,min_face_surface_in_image=surface_visage_min_in_image)

        master_dic=pred_model_4(master_dic)

        dic_list.append(master_dic)

    print("creating pickle")

    for dic in dic_list:
        if dic['nb_faces']==0:
            store_one_dict(dataset="LANDSCAPE",data_status="categorized_dict",image_dict=dic)
        if dic['nb_faces']==1:
            store_one_dict(dataset="PORTRAIT",data_status="categorized_dict",image_dict=dic)
        else :
            store_one_dict(dataset="GROUP",data_status="categorized_dict",image_dict=dic)

    print("uploading pickle")

    upload_pickle(dataset="GROUP",data_status="categorized_dict")
    upload_pickle(dataset="PORTRAIT",data_status="categorized_dict")
    upload_pickle(dataset="LANDSCAPE",data_status="categorized_dict")

    print("cleaning pickle")

    clean_pickle(dataset="GROUP",data_status="categorized_dict")
    clean_pickle(dataset="PORTRAIT",data_status="categorized_dict")
    clean_pickle(dataset="LANDSCAPE",data_status="categorized_dict")



    for i in range (len(dic_list)):
        print(dic_list[i]["image_name"])
        print(dic_list[i]["MOS"])
        print(dic_list[i]['nb_faces'])


if __name__=="__main__":

    clustering_pipeline(dataset="PORTRAIT",start=1,finish=30,chunk_size=10)
