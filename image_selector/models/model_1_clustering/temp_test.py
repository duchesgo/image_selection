import numpy as np
import pandas as pd
import os
# for the VGG model
from keras.applications.vgg16 import VGG16
from keras.models import Model
# for dimension reduction
from sklearn.decomposition import PCA
# for loading images and processing them
from image_selector.models.model_1_clustering.getdata_model_1_clustering import img_to_ndarray
from image_selector.models.model_1_clustering.utils_model_1_clustering import preprocessed_X_clustering, extract_features



### DIMENSIONALITY REDUCTION - PCA

def pca_calcul_components(features):
    pca = PCA()
    features_pca = pca.fit_transform(features)
    cumulated_variance = np.cumsum(pca.explained_variance_ratio_)
    minimal_pc_count = len(cumulated_variance[cumulated_variance <= 0.8]) + 1
    return minimal_pc_count

def pca(features):
    n_compo = pca_calcul_components(features)
    pca_cluster = PCA(n_components=n_compo, random_state=22)
    pca_features = pca_cluster.fit_transform(features)
    return pca_features



### CLUSTERING

def clustering(pca_features):
    nb_images = len(pca_features)
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
    list_values = list()
    for value in dict_clusters.values():
        list_values.append(value)
    dict_clustering = dict(zip(list_names, list_values))
    return dict_clustering



# MODEL 1 - clustering main function

def model_clustering(X_path):
    X, list_names = img_to_ndarray(X_path)
    X_preprocessed = preprocessed_X_clustering(X)
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = extract_features(X_preprocessed, model)
    pca_features = pca(features)
    nb_clusters, dict_clusters = clustering(pca_features)
    dict_clusters_final = dict_clustering(dict_clusters, list_names)
    return nb_clusters, dict_clusters_final



# if __name__=='__main__':
#     print('*** Sart test: main_model1_clustering ***')
#     X_path = "/home/celinethomas/code/duchesgo/image_selection/draft/data/selection5/autres/autres1/"
#     nb_clusters, result_clusters = model_clustering(X_path)
#     print(f'Clusters number: {nb_clusters}')
#     print(f'Clusters dict final: {result_clusters}')
#     df_clusters = pd.DataFrame(list(result_clusters.items()),
#                    columns=['photo_id', 'clusters']).reset_index()
#     df_clusters.to_csv('/home/celinethomas/code/duchesgo/image_selection/draft/data/selection5/autres/autres1_10.csv')
#     print('*** End test: main_model1_clustering ***')
