import pandas as pd
import numpy as np
# for the VGG model
from keras.applications.vgg16 import VGG16
from keras.models import Model
# for dimension reduction
from sklearn.decomposition import PCA
# for clustering
from sklearn.cluster import KMeans


### EXTRACT FEATURES USING VGG16 MODEL

# load model
model = VGG16()
# remove the output layer
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

def extract_features(X_processed, model):
    '''this fonction return a ndarray (X, features) using this model VGG as a feature extractor only'''
    # preprocessing
    X_resized = resize_X_clustering(X_processed)
    X_preprocessed = preproc_X_clustering(X_resized)
    # get the featur vector
    features = model.predict(np.array(X_preprocessed),use_multiprocessing=True)
    return features


### DIMENSIONALITY REDUCTION (PCA)

def pca_calcul_component(X_features):
    pca = PCA()
    features_pca = pca.fit_transform(X_features)
    cumulated_variance = np.cumsum(pca.explained_variance_ratio_)
    minimal_pc_count = len(cumulated_variance[cumulated_variance <= 0.8]) + 1
    return minimal_pc_count

def pca_features(X_features):
    n_compo = pca_calcul_component(X_features)
    pca_cluster = PCA(n_components=n_compo, random_state=22)
    features_cluster = pca_cluster.fit_transform(X_features)
    return features_cluster


### CLUSTERING

def clustering():
    '''Function returning '''
    pass
