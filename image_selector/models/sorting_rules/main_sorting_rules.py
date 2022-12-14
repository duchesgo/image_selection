import math
from image_selector.models.sorting_rules.utils_sorting_rules import cluster_list, sorting_without_cluster_landscape, sorting_without_cluster_humain, sorting_cluster
from image_selector.models.sorting_rules.scores_sorting_rules import calculating_scores

def sorting_img(list_dict_img, category):
    """This fonction sort the name of the image into excellent, good or rubbish list.
    It return 3 lists given a list of image dictionnary and a category (LANDSCAPE or PORTRAIT or GROUP):
    1- list of the name of the excellent pictures
    2- list of the name of the good pictures
    3- list of the name of the rubbish pictures """

    # Creating the 3 empty lists, which will be append with the name of the pictures
    excellent_list = list()
    good_list = list()
    rubbish_list = list()

    # Sorting pictures from LANSCAPE category
    if category == "LANDSCAPE":

        # 1. Calculating the number of clusters
        n_cluster = len(set([img["cluster"] for img in list_dict_img]))

        # 2. Iterating on each cluster to sort the pictures
        for num in range(n_cluster):
            # Creating a list with image-dictionnaries for each cluster
            list_cluster = cluster_list(list_dict_img, num)
            # If the cluster number = 0 (no cluster), the pictures are sorted by their MOS score
            # Adding the name of the picture to the 3 list (EXCELLENT, GOOD, RUBBISH)
            if num == 0:
                excellent_img, good_img, rubbish_img = sorting_without_cluster_landscape(list_cluster)
                excellent_list += excellent_img
                good_list += good_img
                rubbish_list += rubbish_img
            # If the pictures are part of a cluster, 10% of the pictures with the best MOS score goes to the GOOD folder, the others to RUBBISH folder
            # Adding the name of the picture to the 2 list (GOOD, RUBBISH)
            else:
                pourcentage = math.ceil(len(list_cluster)*0.1)
                good_img, rubbish_img = sorting_cluster(list_cluster, pourcentage, "MOS")
                good_list += good_img
                rubbish_list += rubbish_img


    # Sorting pictures from PORTRAIT or GROUP categories
    if category == "PORTRAIT" or category == "GROUP":
        # 1. Calculating and adding the final score to the dictionnaries for each picture
        list_dict_img = calculating_scores(list_dict_img)

        # 2. Calculating the number of clusters
        n_cluster = len(set([img["cluster"] for img in list_dict_img]))

        # 3. Iterating on each cluster to sort the pictures
        for num in range(n_cluster):
            # Creating a list with image-dictionnaries for each cluster
            list_cluster = cluster_list(list_dict_img, num)
            # If the cluster number = 0 (no cluster), the pictures are sorted by their final score
            # Adding the name of the picture to the 3 list (EXCELLENT, GOOD, RUBBISH)
            if num == 0:
                excellent_img, good_img, rubbish_img = sorting_without_cluster_humain(list_cluster)
                excellent_list += excellent_img
                good_list += good_img
                rubbish_list += rubbish_img
            # If the pictures are part of a cluster, 10% of the pictures with the best MOS score goes to the GOOD folder, the others to RUBBISH folder
            # Adding the name of the picture to the 2 list (GOOD, RUBBISH)
            else:
                pourcentage = math.ceil(len(list_cluster)*0.1)
                good_img, rubbish_img = sorting_cluster(list_cluster, pourcentage, "final_score")
                good_list += good_img
                rubbish_list += rubbish_img


    return excellent_list, good_list, rubbish_list






# if __name__ == '__main__':
#     print("*********************** START *****************************")

#     dict_portrait = [{'image_name': '20200830_103939.jpg',
#     'cluster': 0,
#     'nb_faces': 1,
#     'cropped_faces': [{'emotion': 'happy', 'nb_eyes': 2}],
#     'MOS': 69,},
#     {'image_name': '20200830_103931.jpg',
#     'cluster': 0,
#     'nb_faces': 1,
#     'cropped_faces': [{'emotion': 'happy', 'nb_eyes': 1}],
#     'MOS': 43,},
#     {'image_name': '20200830_103932.jpg',
#     'cluster': 0,
#     'nb_faces': 1,
#     'cropped_faces': [{'emotion': 'sad', 'nb_eyes': 1}],
#     'MOS': 61,},
#     {'image_name': '20200908_163321.jpg',
#     'cluster': 1,
#     'nb_faces': 3,
#     'cropped_faces': [{'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'surprise', 'nb_eyes': 1},
#     {'emotion': 'neutral', 'nb_eyes': 1}],
#     'MOS': 72},
#     {'image_name': '20220925_115256.jpg',
#     'cluster': 1,
#     'nb_faces': 3,
#     'cropped_faces': [{'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'neutral', 'nb_eyes': 1}],
#     'MOS': 42},
#     {'image_name': '20220925_115259.jpg',
#     'cluster': 3,
#     'nb_faces': 5,
#     'cropped_faces': [{'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'sad', 'nb_eyes': 0},
#     {'emotion': 'sad', 'nb_eyes': 2},
#     {'emotion': 'neutral', 'nb_eyes': 0}],
#     'MOS': 89},
#     {'image_name': '20200908_163333.jpg',
#     'cluster': 2,
#     'nb_faces': 5,
#     'cropped_faces': [{'emotion': 'happy', 'nb_eyes': 1},
#     {'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'sad', 'nb_eyes': 2},
#     {'emotion': 'neutral', 'nb_eyes': 2},
#     {'emotion': 'neutral', 'nb_eyes': 2}],
#     'MOS': 75},
#     {'image_name': '20200908_163330.jpg',
#     'cluster': 2,
#     'nb_faces': 5,
#     'cropped_faces': [{'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'happy', 'nb_eyes': 1},
#     {'emotion': 'happy', 'nb_eyes': 2}],
#     'MOS': 33},
#     {'image_name': '20220925_115258.jpg',
#     'cluster': 2,
#     'nb_faces': 5,
#     'cropped_faces': [{'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'neutral', 'nb_eyes': 2},
#     {'emotion': 'sad', 'nb_eyes': 1},
#     {'emotion': 'happy', 'nb_eyes': 2}],
#     'MOS': 40,},
#     {'image_name': '20220925_115659.jpg',
#     'cluster': 3,
#     'nb_faces': 1,
#     'cropped_faces': [{'emotion': 'sad', 'nb_eyes': 2}],
#     'MOS': 89},
#     {'image_name': '20200908_163331.jpg',
#     'cluster': 3,
#     'nb_faces': 1,
#     'cropped_faces': [{'emotion': 'happy', 'nb_eyes': 0}],
#     'MOS': 70,},
#     {'image_name': '20200908_163337.jpg',
#     'cluster': 3,
#     'nb_faces': 1,
#     'cropped_faces': [{'emotion': 'neutral', 'nb_eyes': 1}],
#     'MOS': 65}]
#     print('dict OK')
#     print('-------')

#     excellent_list, good_list, rubbish_list = sorting_img(dict_portrait, "GROUP")
#     print(f'Excellent photos: {excellent_list}')
#     print('-----')
#     print(f'Good photos: {good_list}')
#     print('-----')
#     print(f'Rubbish photos: {rubbish_list}')

#     dict_paysage =[
#         {"image_name" : "20220925_115258.jpg",
#          "cluster" : 0,
#          "nb_faces" : 0,
#          "MOS" : 72},
#         {"image_name" : "20220925_115251.jpg",
#          "cluster" : 0,
#          "nb_faces" : 0,
#          "MOS" : 82},
#         {"image_name" : "20220925_115252.jpg",
#          "cluster" : 0,
#          "nb_faces" : 0,
#          "MOS" : 56},
#         {"image_name" : "20220925_115253.jpg",
#          "cluster" : 1,
#          "nb_faces" : 0,
#          "MOS" : 71},
#         {"image_name" : "20220925_115254.jpg",
#          "cluster" : 1,
#          "nb_faces" : 0,
#          "MOS" : 55},
#         {"image_name" : "20220925_115255.jpg",
#          "cluster" : 1,
#          "nb_faces" : 0,
#          "MOS" : 66},
#          {"image_name" : "20220925_115256.jpg",
#          "cluster" : 2,
#          "nb_faces" : 0,
#          "MOS" : 45},
#          {"image_name" : "20220925_115257.jpg",
#          "cluster" : 2,
#          "nb_faces" : 0,
#          "MOS" : 56},
#          {"image_name" : "20220925_115260.jpg",
#          "cluster" : 3,
#          "nb_faces" : 0,
#          "MOS" : 88},
#          {"image_name" : "20220925_115261.jpg",
#          "cluster" : 3,
#          "nb_faces" : 0,
#          "MOS" : 87},
#          {"image_name" : "20220925_115262.jpg",
#          "cluster" : 0,
#          "nb_faces" : 0,
#          "MOS" : 40}]

#     excellent_list, good_list, rubbish_list = sorting_img(dict_paysage, "LANDSCAPE")
#     print(f'Excellent photos: {excellent_list}')
#     print('-----')
#     print(f'Good photos: {good_list}')
#     print('-----')
#     print(f'Rubbish photos: {rubbish_list}')


#     print("*********************** END *****************************")
