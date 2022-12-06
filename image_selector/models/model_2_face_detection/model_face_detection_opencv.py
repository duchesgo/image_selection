##############################################################
# Fichier d'anayse des visages dans une image via modèle OpenCV
# input : une image
# output : nb de visages, photo avec les visages, les visages et leur surface
##############################################################

import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img



# PARAMETRES
# xxxsss Variables à optimiser via GridSearch
# xxxsss migrer en variables locales ou d'environnement ?
scaleFactor_tp=1.1   # compense pour les visages plus ou moins proches de l'objectif
minNeighbors_tp=3    # Nb de voisins pour reconnaître un objet ; pas clair
minSize_tp=(23, 23)  # Taille de chaque fenêtre précédente ; pas clair
min_face_surface_in_image_tp = 0.002  # Surface minimale d'un visage pour être recevable

# FICHIERS IMAGES
# xxxsss comment pointer sur  l'image et le path
# image_name_tp = 'abba.png'
image_name_tp = '2021 06 19 - 202339 - Etretat.jpg'
image_path_tp = '../../../draft/images_test'

image_dict_tp = {'image_name': image_name_tp,
                 'image_array': cv2.imread(os.path.join(image_path_tp, image_name_tp))}


# FICHIERS CASCADE
cascade_path_tp = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"


def face_cropping(image_array, x, y, w, h):
    # Fonction qui renvoie un visage au sein d'une image à partir des coordonnées du visage
    # input = image en format np.array ; coordonnées
    # output = visage en np.array ; surface occupée par ce visage dans l'image
    cropped_face_dict = {}
    cropped_face_dict['cropped_face'] = image_array[y:y+h,x:x+w,:]
    cropped_face_dict['relative_face_surface'] = round((w*h) / (image_array.shape[0] * image_array.shape[1]), 4)
    return cropped_face_dict


def visualize_cropped_faces(image_dict, scaleFactor, minNeighbors, minSize, min_face_surface_in_image):
    # Fonction qui prend :
    # input : l'output de face_detecting
    # output : imprime une feuille de synthèse avec
    #     nb de visages ;
    #     image avec les cadres
    #     pour chaque visage, cropped_visage ; surface relative du visage

    list_faces_cropped =[]
    list_faces_cropped.append(image_dict['image_array'])
    list_faces_cropped.append(image_dict['image_with_faces'])

    for i in range(len(image_dict['cropped_faces'])):
        list_faces_cropped.append(image_dict['cropped_faces'][i]['cropped_face'])

    fig, axes = plt.subplots(nrows=4,ncols=4,figsize=(15,10))

    for index, (image, ax) in enumerate(zip(list_faces_cropped, axes.flat)):
        ax.imshow(image)
        if index>=2:
            ax.set_title(f"surface relative {image_dict['cropped_faces'][index-2]['relative_face_surface']:.2%}")

    fig.suptitle(f"Found {image_dict['nb_faces']} faces! \
        with {scaleFactor}_{minNeighbors}_{minSize}_{min_face_surface_in_image}")
    # fig.show()
    fig.savefig(f"{scaleFactor}_{minNeighbors}_{minSize}_{min_face_surface_in_image}_{image_dict['image_name']}.jpg")
    # xsx créer un mkdir et sauver les photos à cet endroit
    plt.close()

    return None


def face_detecting(image_dict, cascade_path, scaleFactor, minNeighbors, minSize,\
    min_face_surface_in_image, visualize=False):
    # Fonction de détection de visages dans une image
    # Input : une image (type jpg, png...)
    # Output : dictionnaire contenant
    #     nb de visages ;
    #     image avec les cadres
    #     pour chaque visage, cropped_visage ; surface relative du visage

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascade_path)

    # Read the image
    gray = cv2.cvtColor(image_dict['image_array'], cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize,
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    image_dict['nb_faces'] = 0
    image_dict['cropped_faces']=[]

    # Draw a rectangle around the faces and
    image_with_faces = image_dict['image_array']
    for (x, y, w, h) in faces:
        cropped_face_dict = face_cropping(image_dict['image_array'], x, y, w, h)

        # filtre qui élimine les visages trop petits
        if cropped_face_dict['relative_face_surface'] > min_face_surface_in_image:
            image_with_faces = cv2.rectangle(image_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
            image_dict['nb_faces'] += 1
            image_dict['cropped_faces'].append(cropped_face_dict)

    image_dict['image_with_faces'] = image_with_faces

    if visualize:
        path_draft = os.getcwd()
        path_mkdir = f"{scaleFactor_tp}_{minNeighbors_tp}_{minSize_tp}_{min_face_surface_in_image_tp}"
        path_full = os.path.join(path_draft, path_mkdir)
        if not os.path.exists(path_full):
            os.mkdir(path_full)
        visualize_cropped_faces(image_dict, scaleFactor, minNeighbors, minSize, min_face_surface_in_image)

    return image_dict


if __name__=='__main__':
    print('Début test face_detecting')
    # xxxsss attention à la gestion des variables ci-dessous pour l'appel de face_detecting
    face_detecting(image_dict_tp, cascade_path_tp, scaleFactor_tp, \
        minNeighbors_tp, minSize_tp, min_face_surface_in_image_tp, visualize=True)
    print('Fin test face_detecting')
