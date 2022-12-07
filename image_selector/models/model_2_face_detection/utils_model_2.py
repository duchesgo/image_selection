# IMPORTS
import os
import matplotlib.pyplot as plt


def visualize_cropped_faces(image_dict):
    # Fonction qui, à partir du dictionnaire image_dict, imprime un fichier contenant
    #    l'image d'origine
    #    l'image avec les cadres autour des visages
    #    la photo de tous les visages identifiés

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

    path_draft = os.getcwd()
    path_mkdir = f"DeepFace"
    path_full = os.path.join(path_draft, path_mkdir)

    fig.savefig(os.path.join(path_full, f"DeepFace_{image_dict['image_name']}.jpg"))

    plt.close()

    return None
