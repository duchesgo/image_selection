##############################################################
# Fichier d'anayse des visages dans une image
# input : une image
# output : nb de visages, photo avec les visages, les visages et leur surface
##############################################################

import pandas as pd
import cv2
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
#  from image_selection.temp_GD.OpenCV.params import xxxsss


# PARAMETRES
# xxxsss Variables à optimiser via GridSearch
# xxxsss migrer en variables locales ou d'environnement ?
scaleFactor_tp=1.1   # compense pour les visages plus ou moins proches de l'objectif
minNeighbors_tp=5    # Nb de voisins pour reconnaître un objet ; pas clair
minSize_tp=(30, 30)  # Taille de chaque fenêtre précédente ; pas clair
surface_visage_min_in_image = 0.004  # Surface minimale d'un visage pour être recevable

# FICHIERS IMAGES
# xxxsss comment pointer sur  l'image et le path
image_path_tp = 'images_test/abba.png'


# FICHIERS CASCADE
cascade_path_tp = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"


def face_cropping(image_path, x, y, w, h):
    # Fonction qui renvoie un visage au sein d'une image à partir des coordonnées du visage
    # input = fichier de format jpg, png ; coordonnées
    # output = fichier image (pas un tensor pour l'instant) ; surface occupée par ce visage dans l'image
    cropped_face_dict = {}
    image_array = img_to_array(image_path)
    cropped_face_dict['cropped_face'] = array_to_img(image_array[y:y+h,x:x+w,:])
    cropped_face_dict['relative_face_surface'] = round((w*h) / (image_array.shape[0] * image_array.shape[1]), 4)
    return cropped_face_dict


def face_detecting(image_path, cascade_path, scaleFactor, minNeighbors, minSize, visualize=False):
    # Fonction de détection de visages dans une image
    # Input : une image (type jpg, png...)
    # Output : dictionnaire contenant
    #     nb de visages ;
    #     image avec les cadres
    #     pour chaque visage, cropped_visage ; surface relative du visage

    detected_faces={}

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascade_path)

    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize,
        flags = cv2.CASCADE_SCALE_IMAGE      # xxxsss est-ce un hyper-paramètre à gérer ?
    )

    detected_faces['base_image'] = image_path
    detected_faces['nb_faces'] = 0
    detected_faces['cropped_faces']=[]

    # Draw a rectangle around the faces and
    image_with_faces = cv2.imread(image_path)
    for (x, y, w, h) in faces:
        cropped_face_dict = face_cropping(image, x, y, w, h)

        # filtre qui élimine les visages trop petits
        if cropped_face_dict['relative_face_surface'] > surface_visage_min_in_image:
            image_with_faces = cv2.rectangle(image_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
            detected_faces['nb_faces'] += 1
            detected_faces['cropped_faces'].append(cropped_face_dict)

    detected_faces['image_with_faces'] = image_with_faces

    # xxxsss Fonction temporaire à garder le temps d'optimiser le paramétrage supra et que le nb de visages détectés soit correct
    # xsx mkdir -p ~/code/duchesgo/image-selection/temp_GD/OpenCV/{scaleFactor}_{minNeighbors}_{minSize} && cd $_
    if visualize:
        list_faces_cropped =[]
        list_faces_cropped.append(image)
        list_faces_cropped.append(image_with_faces)

        for i in range(len(detected_faces['cropped_faces'])):
            list_faces_cropped.append(detected_faces['cropped_faces'][i]['cropped_face'])

        fig, axes = plt.subplots(nrows=4,ncols=4,figsize=(15,10))

        for index, (image, ax) in enumerate(zip(list_faces_cropped, axes.flat)):
            ax.imshow(image)
            if index>=2:
                ax.set_title(f"surface relative {detected_faces['cropped_faces'][index-2]['relative_face_surface']:.2%}")

        fig.suptitle("Found {0} faces!".format(detected_faces['nb_faces']))
        fig.show()
        print(f"{scaleFactor}_{minNeighbors}_{minSize}_{image_path}.jpg")
        # fig.savefig(f"{scaleFactor}_{minNeighbors}_{minSize}_{image_path}.jpg")

    return detected_faces


if __name__=='__main__':
    print('Début test face_detecting')
    # xxxsss attention à la gestion des variables ci-dessous pour l'appel de face_detecting
    face_detecting(image_path_tp, cascade_path_tp, scaleFactor_tp, minNeighbors_tp, minSize_tp)
    print('Fin test face_detecting')
