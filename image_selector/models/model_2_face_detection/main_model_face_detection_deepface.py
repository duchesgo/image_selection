##############################################################
# Fichier d'anayse des visages dans une image via modèle OpenCV
# input : une image
# output : nb de visages, photo avec les visages, les visages et leur surface
##############################################################

# IMPORTS
import pandas as pd
import os
import matplotlib.pyplot as plt

import cv2
from deepface import DeepFace
from retinaface import RetinaFace

from image_selector.models.model_2_face_detection.utils_model_2 import visualize_cropped_faces


# FICHIERS IMAGES - Variables temporaires pour tester la fct dans if name = main
image_name_tp = 'abba.png'
# image_name_tp = '2021 06 19 - 202339 - Etretat.jpg'

image_path_tp = 'draft/images_test'
image_dict_tp = {'image_name': image_name_tp,
                 'image_array': cv2.imread(os.path.join(image_path_tp, image_name_tp))}


# Paramètres
min_face_surface_in_image_tp = 0.004  # Surface minimale d'un visage pour être recevable
threshold_tp = 0.9    # Seuil minimal de taille de détection des visages pour les fonctions detect_faces et extract_faces


# metrics pour les fonctions .find() et .verify()
# distance_metric = metrics[1]
metrics = ["cosine", "euclidean", "euclidean_l2"]    # Il semble qu'il faille préférer metrics[2]

# Modèle pour les fonctions .find() et .verify()
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]

# Paramètre qui définit le fond posé derrière la personne découpée ; par défaut, 'opencv' ;
# il semble que 'retinaface' soit très performant, soit backends[4]
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']


def face_detecting_DeepFace(image_dict, min_face_surface_in_image, visualize=False):
    # Fonction de détection de visages dans une image via DeepFace
    # xxxsss voir si on réintroduit min_face_surface_in_image ;
    # Input : un dictionnaire contenant a minima
    #         'image_name' : du type 'nom_fichier.jpg'
    #         'image_array' : l'image en format np.array
    # Output : dictionnaire contenant, outre les éléments précédents
    #     'nb_faces' : nb de visages ;
    #     'image_with_faces' : image avec les cadres
    #     'cropped_faces', qui est une liste de dictionnaires, où chaque dico donne pour chaque visage :
    #         cropped_face : le visage découpé
    #         surface relative du visage
    #         emotion parmi : ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


    # ----- Travail sur la partie picturale des visages
    faces_pict_RetinaFace = []
    faces_pict_RetinaFace = RetinaFace.extract_faces(image_dict['image_array'], align = False, \
        allow_upscaling=False)

    image_dict['nb_faces'] = 0
    image_dict['cropped_faces']=[]

    for face_pict in faces_pict_RetinaFace:

        # filtre qui élimine les visages trop petits
        face_pict_surface = round((face_pict.shape[0] * face_pict.shape[1]) \
                / (image_dict['image_array'].shape[0] * image_dict['image_array'].shape[1]), 4)

        if face_pict_surface > min_face_surface_in_image:
            image_dict['nb_faces'] += 1

            cropped_face_dict = {}
            cropped_face_dict['cropped_face'] = face_pict

            plt.imshow(face_pict)
            plt.savefig("array")
            emotions = {}
            emotions = DeepFace.analyze(img_path = "array.png", \
                detector_backend = backends[4], actions = ["emotion"], prog_bar=False, enforce_detection=False)
            cropped_face_dict['emotion'] = emotions['dominant_emotion']
            plt.close()
            os.remove('array.png')

            cropped_face_dict['relative_face_surface'] = face_pict_surface

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

            x = face_info["facial_area"][0]
            y = face_info["facial_area"][1]
            w = face_info["facial_area"][2] - face_info["facial_area"][0]
            h = face_info["facial_area"][3] - face_info["facial_area"][1]

            face_info_surface = round((w * h) \
                / (image_dict['image_array'].shape[0] * image_dict['image_array'].shape[1]), 4)

            if face_info_surface > min_face_surface_in_image:
                image_with_faces = cv2.rectangle(image_with_faces, \
                    (x, y), (x+w, y+h), (0, 255, 0), 2)

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
    # ATTENTION à la gestion des variables ci-dessous pour l'appel de face_detecting
    image_dict_test_name = face_detecting_DeepFace(image_dict_tp, min_face_surface_in_image_tp, visualize=False)
    print(image_dict_test_name['nb_faces'])
    print('Fin test face_detecting')
