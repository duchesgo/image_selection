import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.image import resize_with_pad
from keras.applications.vgg16 import preprocess_input

def resize_X_clustering(X_list):
    """This function return a list a resized nparray-image given a list of nparray-image"""
    X_resize = [np.array(resize_with_pad(x, 224, 224)).astype('uint8') for x in X_list]
    return X_resize

def preproc_X_clustering(X_list):
    """This function return a list a tensor-image given a list of tensor-image, using preprocess_input method"""
    X_preproc = [preprocess_input(x) for x in X_list]
    return X_preproc
