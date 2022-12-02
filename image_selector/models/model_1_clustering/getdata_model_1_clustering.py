import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# for loading the images
import os
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import convert_to_tensor


# GET DATA AND TRANSFORM IMAGES INTO NDARRAY

def img_to_ndarray(path):
    '''return a list of ndarray images from the path of a folder containing the images'''
    # get the image names in a list from a folder
    contenu = os.listdir(path)
    contenu = [x for x in contenu]
    contenu.sort()
    print(contenu)
    # for boucle to transform image into nparray
    list_img = []
    for img in contenu :
        if img.endswith('.jpg'):
            list_img.append(jpg_to_ndarray(path, img))
        elif img.endswith('.bmp'):
            list_img.append(bmp_to_ndarray(path, img))
    return list_img

def bmp_to_ndarray(path, img):
    ''' return an image in ndarray from a .bmp'''
    img_bmp = Image.open(path+img)
    img_array = np.array(img_bmp)
    return img_array

def jpg_to_ndarray(path, img):
    ''' return an image in ndarray from a .jpg'''
    img_jpg = Image.open(path+img)
    img_jpg = ImageOps.exif_transpose(img_jpg)
    img_array = np.array(img_jpg)
    return img_array

if __name__=='__main__':
    print('Début test face_detecting')
    X_path = "/home/celinethomas/code/duchesgo/image_selection/draft/dataset_typologie/typologie.csv"
    X = img_to_ndarray(X_path)
    print(X[3].shape)
    print('Fin test face_detecting')
