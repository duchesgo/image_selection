import numpy as np
from image_selector.models.model_3_distortion_score.utils_model_3 import return_score_torch

def pred_model_3_distortion_score(input_dictionnary):
    img=input_dictionnary["initial_image"]
    MOS=round(return_score_torch(img),2)
    input_dictionnary["MOS"]=MOS
    return input_dictionnary


if __name__=="__main__":

    pass
