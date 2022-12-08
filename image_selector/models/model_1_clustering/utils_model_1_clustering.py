import numpy as np
# for preproccess X
from image_selector.models.model_1_clustering.preprocessor_model_1_clustering import resize_X_clustering, input_X_clustering
# for the VGG model
# from keras.applications.vgg16 import VGG16
from keras.models import Model
# for dimension reduction
from sklearn.decomposition import PCA



### PREPROCESSED IMAGES

def preprocessed_X_clustering(X_list):
    """
    The function returns a list a preprocessed ndarray-images given a list of ndarray-images.
    """
    X_resize = resize_X_clustering(X_list)
    X_preprocessed = input_X_clustering(X_resize)
    return X_preprocessed



### EXTRACT FEATURES USING VGG16 MODEL

def extract_features(X_preprocessed, model):
    """
    The fonction returns a ndarray of features, using the model VGG16 as a feature extractor.
    """
    features = model.predict(np.array(X_preprocessed), use_multiprocessing=True)
    return features



### DIMENSIONALITY REDUCTION - PCA

def pca_calcul_components(features):
    """
    The fonction returns the minimal n_component needed for the pca, based on a cumulated_variance of 80%.
    """
    pca = PCA()
    features_pca = pca.fit_transform(features)
    cumulated_variance = np.cumsum(pca.explained_variance_ratio_)
    minimal_pc_count = len(cumulated_variance[cumulated_variance <= 0.8]) + 1
    return minimal_pc_count

def pca(features):
    """
    The fonction returns a ndarray of features reduced.
    """
    n_compo = pca_calcul_components(features)
    pca_cluster = PCA(n_components=n_compo, random_state=22)
    pca_features = pca_cluster.fit_transform(features)
    return pca_features



### CLUSTERING

def clustering(pca_features,distance=20):
    """
    This fonction returns the number of clusters and a dictionnary:
    key: index of the image, value: if the image is part of a cluster (cluster number) or not (0).
    """
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
            if np.linalg.norm(pca_features[h]-pca_features[v]) < distance:
                cluster_found = True
                dict_clusters[h] = nb_clusters + 1
                dict_clusters[v] = nb_clusters + 1

    return nb_clusters+1, dict_clusters


def dict_clustering(dict_clusters, list_names):
    """
    This fonction returns a dictionnary, given a dictionnary.
    It replace the index of an image by the name of this image:
    key: name of the image, value: if the image is part of a cluster (cluster number) or not (0)
    """
    list_values = list()
    for value in dict_clusters.values():
        list_values.append(value)
    dict_clustering = dict(zip(list_names, list_values))
    return dict_clustering




# -----SUPPRIMER -----
# if __name__=='__main__':
#     print('*** Sart test: model1_clustering ***')
#     # LOAD
#     X_path = "/home/celinethomas/code/duchesgo/image_selection/draft/dataset_typologie/selection/"
#     X, list_names = img_to_ndarray(X_path)
#     print(f'List: {len(list_names)}')
#     print(f'Get data, X shape: {X[3].shape}')
#     # PREPROCESSED
#     X_preprocessed = preprocessed_X_clustering(X)
#     print(f'Preprocessed, X shape : {X_preprocessed[3].shape}')
#     # MODEL
#     features = extract_features(X_preprocessed, model)
#     print(f'VGG model, features shape (56,4096):  {features.shape}, type features: {type(features)}')
#     # PCA
#     pca_features = pca(features)
#     print(f'PCA, features shape (56, 26) : {pca_features.shape}')
#     # CLUSTERING
#     nb_clusters, dict_clusters = clustering(pca_features)
#     dict_clusters_final = dict_clustering(dict_clusters, list_names)
#     print(f'Clusters number: {nb_clusters}')
#     print(f'Clusters dict final: {dict_clusters_final}')
#     print('*** End test: model1_clustering ***')
