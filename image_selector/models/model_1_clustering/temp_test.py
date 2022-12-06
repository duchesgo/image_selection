import numpy as np
import pandas as pd
import os
# for the VGG model
from keras.applications.vgg16 import VGG16
from keras.models import Model
# for dimension reduction
from sklearn.decomposition import PCA

from image_selector.models.model_1_clustering.getdata_model_1_clustering import img_to_ndarray
from image_selector.models.model_1_clustering.utils_model_1_clustering import preprocessed_X_clustering, extract_features

### DIMENSIONALITY REDUCTION - PCA

def pca_calcul_components(features):
    """The fonction returns the minimal n_component needed for the pca, based on 80%"""
    pca = PCA()
    features_pca = pca.fit_transform(features)
    cumulated_variance = np.cumsum(pca.explained_variance_ratio_)
    minimal_pc_count = len(cumulated_variance[cumulated_variance <= 0.8]) + 1

    return minimal_pc_count

def pca(features):
    """The fonction returns """
    n_compo = pca_calcul_components(features)
    pca_cluster = PCA(n_components=n_compo, random_state=22)
    pca_features = pca_cluster.fit_transform(features)
    return pca_features


### CLUSTERING

def clustering(pca_features):
    '''This fonction return a dictionnary an idex-img -key- and if they are part of a cluster (number) or not (0) -value-'''

    # get the number of images in the dataset
    nb_images = len(pca_features)
    # initiate a dictionnary to 0: key = name of the images, value = 0
    dict_clusters = dict()
    for img in range(nb_images):
        dict_clusters[img] = 0
    dict_clusters

    # identify the clusters
    nb_clusters = 0
    cluster_found = False
    for h in range(nb_images):
        if dict_clusters[h] != 0:
            continue
        if cluster_found:
            nb_clusters += 1
            cluster_found = False
        for v in range(nb_images):
            if h==v :
                continue
            if dict_clusters[v] != 0:
                continue
            if np.linalg.norm(pca_features[h]-pca_features[v]) < 10:
                cluster_found = True
                dict_clusters[h] = nb_clusters + 1
                dict_clusters[v] = nb_clusters + 1
    return nb_clusters+1, dict_clusters

def dict_clustering(dict_clusters, list_names):
    '''This fonction return a dictionnary with the name of the images -key- and if they are part of a cluster (number) or not (0) -value-'''
    list_values = list()
    for value in dict_clusters.values():
        list_values.append(value)
    dict_clustering = dict(zip(list_names, list_values))
    return dict_clustering


def model_clustering(X_path):
    # Load X images and transform images into ndarray
    X, list_names = img_to_ndarray(X_path)
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


if __name__=='__main__':
    print('*** Sart test: main_model1_clustering ***')
    X_path = "/home/celinethomas/code/duchesgo/image_selection/draft/data/selection5/autres/autres1/"
    nb_clusters, result_clusters = model_clustering(X_path)
    print(f'Clusters number: {nb_clusters}')
    print(f'Clusters dict final: {result_clusters}')
    df_clusters = pd.DataFrame(list(result_clusters.items()),
                   columns=['photo_id', 'clusters']).reset_index()
    df_clusters.to_csv('/home/celinethomas/code/duchesgo/image_selection/draft/data/selection5/autres/autres1_10.csv')
    print('*** End test: main_model1_clustering ***')
