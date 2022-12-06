##############################################################
# Fichier d'anayse des visages dans une image via modèle OpenCV
# input : une image
# output : nb de visages, photo avec les visages, les visages et leur surface
##############################################################

# IMPORTS
import pandas as pd
import os
import matplotlib.pyplot as plt

from time import gmtime, strftime

import cv2
from deepface import DeepFace
from retinaface import RetinaFace


# FICHIERS IMAGES - Variables temporaires pour tester la fct dans if name = main
image_name_tp = 'abba.png'
# image_name_tp = '2021 06 19 - 202339 - Etretat.jpg'

image_path_tp = '../../../draft/images_test'
image_dict_tp = {'image_name': image_name_tp,
                 'image_array': cv2.imread(os.path.join(image_path_tp, image_name_tp))}


# def face_cropping(image_array, x, y, w, h):
#     # Fonction qui renvoie un visage au sein d'une image à partir des coordonnées du visage
#     # input = image en format np.array ; coordonnées
#     # output = visage en np.array ; surface occupée par ce visage dans l'image
#     cropped_face_dict = {}
#     cropped_face_dict['cropped_face'] = image_array[y:y+h,x:x+w,:]
#     cropped_face_dict['relative_face_surface'] = round((w*h) / (image_array.shape[0] * image_array.shape[1]), 4)
#     return cropped_face_dict


def visualize_cropped_faces(image_dict):
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

    fig.suptitle(f"Found {image_dict['nb_faces']} faces!")
    # fig.show()

    path_draft = os.getcwd()
    path_mkdir = f"DeepFace"
    path_full = os.path.join(path_draft, path_mkdir)

    fig.savefig(os.path.join(path_full, f"DeepFace_{image_dict['image_name']}.jpg"))

    plt.close()

    return None




# metrics pour les fonctions .find() et .verify()
# distance_metric = metrics[1]
metrics = ["cosine", "euclidean", "euclidean_l2"]    # Il semble qu'il faille préférer metrics[2]

# Modèle pour les fonctions .find() et .verify()
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]

# Paramètre qui définit le fond posé derrière la personne découpée ; par défaut, 'opencv' ;
# il semble que 'retinaface' soit très performant, soit backends[4]
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']


def face_detecting_DeepFace(image_dict, visualize=False):
    # Fonction de détection de visages dans une image via DeepFace
    # xxxsss voir si on réintroduit min_face_surface_in_image ;
    # Input : une image (type jpg, png...)
    # Output : dictionnaire contenant
    #     nb de visages ;
    #     image avec les cadres xxxsss
    #     pour chaque visage, cropped_visage ; surface relative du visage

    # ----- Travail sur la partie picturale des visages
    faces_pict_RetinaFace = []
    faces_pict_RetinaFace = RetinaFace.extract_faces(image_dict['image_array'], align = False, allow_upscaling=False)

    image_dict['nb_faces'] = len(faces_pict_RetinaFace)
    image_dict['cropped_faces']=[]

    for face_pict in faces_pict_RetinaFace:
        cropped_face_dict = {}
        cropped_face_dict['cropped_face'] = face_pict

        plt.imshow(face_pict)
        plt.savefig("array")
        emotions = {}
        emotions = DeepFace.analyze(img_path = "array.png", \
            detector_backend = backends[4], actions = ["emotion"], prog_bar=False, enforce_detection=False)
        cropped_face_dict['emotion'] = emotions['dominant_emotion']
        plt.close()
        # os.remove("array.png")

#       cropped_face_dict['relative_face_surface'] = 0  # Valeur forcée à 0 pour l'instant, faute de calcul

        cropped_face_dict['relative_face_surface'] = \
            round((face_pict.shape[0] * face_pict.shape[1]) \
                / (image_dict['image_array'].shape[0] * image_dict['image_array'].shape[1]), 4)

        image_dict['cropped_faces'].append(cropped_face_dict)

    # ----- Travail sur partie informative des visages ; création d'une image avec les visages repérés
    image_with_faces = image_dict['image_array'].copy()

    faces_info_RetinaFace = []
    faces_info_RetinaFace = RetinaFace.detect_faces(image_dict['image_array'])

    # la fonction RetinaFace rend un tuple si l'image ne contient pas de visage
    # et un dictionnaire si l'image contient des visages
    if isinstance(faces_info_RetinaFace, dict) :
        for i in range(len(faces_info_RetinaFace)):

            face_info = faces_info_RetinaFace[f'face_{i+1}']
            image_with_faces = cv2.rectangle(image_with_faces, \
                (face_info["facial_area"][2], face_info["facial_area"][3]), \
                (face_info["facial_area"][0], face_info["facial_area"][1]), \
                (0, 255, 0), 2)

    image_dict['image_with_faces'] = image_with_faces

    if visualize:
        path_draft = os.getcwd()
        path_mkdir = "DeepFace"
        path_full = os.path.join(path_draft, path_mkdir)
        if not os.path.exists(path_full):
            os.mkdir(path_full)
        visualize_cropped_faces(image_dict)


    return image_dict


if __name__=='__main__':
    print('Début test face_detecting')
    # xxxsss attention à la gestion des variables ci-dessous pour l'appel de face_detecting
    image_dict_test_name = face_detecting_DeepFace(image_dict_tp, visualize=False)
    print(image_dict_test_name['nb_faces'])
    print('Fin test face_detecting')
