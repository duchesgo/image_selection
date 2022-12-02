from tensorflow.io import decode_jpeg
from tensorflow.keras.utils import load_img
from tensorflow.image import rgb_to_grayscale, resize_with_pad
from PIL import Image
import numpy as np




def preprocess_model_4(img_array) -> np.array:
    """
    Preprocess an image as np.array for model 4
    Outputs a processed image as np.array
    """
    #img_jpg = Image.open(image_path)
    #img_array = np.array(img_jpg)
    #img_tensor = convert_to_tensor(img_array)

    #Gray scaling image
    img_grey = rgb_to_grayscale(img_array)

    #Resizing image
    img_resize = resize_with_pad(img_grey, 120,128)

    #Normalization
    image_preprocessed = img_resize / np.max(img_resize)

    return image_preprocessed
