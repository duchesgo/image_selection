import numpy as np
from tensorflow import image
from keras.applications.vgg16 import preprocess_input

# -----SUPPRIMER -----
from getdata_model_1_clustering import img_to_ndarray, bmp_to_ndarray, jpg_to_ndarray


def resize_X_clustering(X_list):
    """This function return a list a resized nparray-image given a list of nparray-image"""
    X_resize = [np.array(image.resize_with_pad(x, 224, 224)).astype('uint8') for x in X_list]
    return X_resize

def input_X_clustering(X_list):
    """This function return a list a nparray-image given a list of tensor-image, using preprocess_input method"""
    X_preproc = [preprocess_input(x) for x in X_list]
    return X_preproc


# -----SUPPRIMER -----
# if __name__=='__main__':
#     print('*** Sart test: preprocessed_model1_clustering ***')
#     X_path = "/home/celinethomas/code/duchesgo/image_selection/draft/dataset_typologie/selection/"
#     X, list_names = img_to_ndarray(X_path)
#     print(f'List: {len(list_names)}')
#     print(f'Get data, X shape: {X[3].shape}')
#     X_resized = resize_X_clustering(X)
#     print(f'Resized, X shape : {X_resized[3].shape}')
#     X_preprocessed = input_X_clustering(X_resized)
#     print(f'Preprocessed, X shape : {X_preprocessed[3].shape}')
#     print('*** End test: preprocessed_model1_clustering ***')
