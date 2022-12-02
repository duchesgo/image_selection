import numpy as np
from tensorflow import convert_to_tensor
import os
#from tensorflow.io import write_file, encode_


def pgm_reader(filename):
    """Takes file name / path as string
    "an2i/an2i_left_angry_open.pgm"
    Returns tuple :
    ( tensor of image , width , height ) """

    f = open(filename, 'r')

    #Read header info
    count = 0

    while count < 3:
        line = f.readline()
        if line[0] == '#': # Ignore comments
            continue
        count = count + 1
        if count == 1: # Magic num info
            magicNum = line.strip()
            if magicNum != 'P2' and magicNum != 'P5':
                f.close()
                print ('Not a valid PGM file')
                exit()

        elif count == 2: # Width and Height
            [width, height] = (line.strip()).split()
            width = int(width)
            height = int(height)

        elif count == 3: # Max gray level
            maxVal = int(line.strip())

    # Read pixels information
    img = []
    buf = f.read()
    elem = buf.split()
    if len(elem) != width*height:
        print("Error in number of pixels")
        return None
    for i in range(height):
        tmplist=[]
        for j in range (width):
            tmplist.append(float(elem[i * width + j ]))
        img.append(tmplist)
    img_tensor = convert_to_tensor(np.array(img))
    return (img_tensor, width, height)

def get_target_CMU (file_path):
    """Takes file name / path as string
    eg : "an2i/an2i_left_angry_open.pgm"

    Return a dictionnary of features, using the name of image convention
    <userid> <pose> <expression> <eyes> <scale>.pgm"""

    filename = os.path.split(file_path)[-1].rstrip(".pgm")



    if "/" in filename :
        tmplist = filename.split("/")[1].split("_")
    else :
        tmplist=filename.split("_")

    dico_features = {"filename" : filename + ".pgm",
                    "userid" : tmplist[0],
                    "pose" : tmplist[1],
                    "expression" : tmplist[2],
                    "eyes" : tmplist[3]}

    return dico_features

def get_data_CMU (file_path):
    res = []
    if file_path.endswith("2.pgm") or file_path.endswith("4.pgm"):
        return None
    return (pgm_reader(file_path)[0],get_target_CMU(file_path))


def get_data_folder_CMU(folder):
    """absolute path of folder if possible, else, data has to be at root"""
    X = []
    y = []
    for filename in os.listdir(folder):    #iterate over files in folder
        file_path = os.path.join(folder, filename)

        # checking if it is a file
        data_of_image = get_data_CMU(file_path)
        if os.path.isfile(file_path) and data_of_image is not None:
            X.append(data_of_image[0])
            y.append(data_of_image[1])

    return X,y

#def get_data_all_persons(root_directory):
#    X = []
#    y = []
#    for file in os.listdir(root_directory) :
#        X_i,y_i = get_data_folder_CMU(file)
#        X.append(X_i)
#        y.append(y_i)
#    return X,y


#if __name__ == "__main__":
#    X,y = get_data_folder_CMU("CMU_data")
#    print("X ========>", len(X))
#    print (X[0])
#    print("y ========>", len(y))
#    print (y[0])
