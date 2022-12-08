from getdata_model_1_clustering import img_to_ndarray
from image_selector.models.model_1_clustering.utils_model_1_clustering import preprocessed_X_clustering, extract_features, pca, clustering, dict_clustering
# for the VGG model
from keras.applications.vgg16 import VGG16
from keras.models import Model



def model_clustering(X_path):
    """
    The function returns the number of clusters and a dictionnary:
    key: name of the image, value: if the image is part of a cluster (cluster number) or not (0).
    """
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
    # Update dictionnary
    dict_clusters_final = dict_clustering(dict_clusters, list_names)

    return nb_clusters, dict_clusters_final




# if __name__=='__main__':
#     print('*** Sart test: main_model1_clustering ***')
#     X_path = "/home/celinethomas/code/duchesgo/image_selection/draft/data/selection/"
#     nb_clusters, result_clusters = model_clustering(X_path)
#     print(f'Clusters number: {nb_clusters}')
#     print(f'Clusters dict final: {result_clusters}')
#     print('*** End test: main_model1_clustering ***')
