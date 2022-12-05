from tensorflow.io import decode_jpeg
from tensorflow.keras.utils import load_img
from tensorflow.image import rgb_to_grayscale, resize_with_pad
from PIL import Image
import numpy as np

def convert_greyscale(img_rgb):
    r,g,b=img_rgb[:,:,0],img_rgb[:,:,1],img_rgb[:,:,2]
    img_grey=0.2989*r+0.5870*g+0.1140*b

    img_grey=np.expand_dims(img_grey,axis=2)
    return img_grey


def preprocess_model_4(img_array) -> np.array:
    """
    Preprocess an image as np.array for model 4
    Outputs a processed image as np.array
    """
    #img_jpg = Image.open(image_path)

    #img_array = np.array(img_array)
    #img_tensor = convert_to_tensor(img_array)


    #Gray scaling image
    #img_grey = rgb_to_grayscale(img_array)
    img_grey = convert_greyscale(img_array)

    #Resizing image
    img_resize = resize_with_pad(img_grey, 120,128)

    #Normalization
    image_preprocessed = img_resize / np.max(img_resize)

    return image_preprocessed
