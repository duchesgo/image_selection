import numpy as np
# for loading the images
import os
from PIL import Image, ImageOps


# GET DATA AND TRANSFORM IMAGES INTO NDARRAY

def img_to_ndarray(path):
    '''return a list of ndarray images from the path of a folder containing the images'''
    # get the image names in a list from a folder
    contenu = os.listdir(path)
    contenu = [x for x in contenu]
    contenu.sort()
    # for boucle to transform image into nparray
    list_img, list_names = [], contenu
    for img in contenu :
        if img.endswith('.jpg'):
            list_img.append(jpg_to_ndarray(path, img))
        elif img.endswith('.bmp'):
            list_img.append(bmp_to_ndarray(path, img))
    return list_img, list_names

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


# -----SUPPRIMER -----
# if __name__=='__main__':
#     print('*** Sart test: getdata_model1_clustering ***')
#     X_path = "/home/celinethomas/code/duchesgo/image_selection/draft/dataset_typologie/selection/"
#     X, list_names = img_to_ndarray(X_path)
#     print(f'Image shape: {X[3].shape}')
#     print(f'List img : {list_names}')
#     print('*** End test: getdata_model1_clustering ***')
