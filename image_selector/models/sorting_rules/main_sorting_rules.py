import math
from image_selector.models.sorting_rules.utils_sorting_rules import cluster_list, sorting_without_cluster, sorting_cluster

def sorting_img(list_dict_img):
    excellent_list = list()
    good_list = list()
    rubbish_list = list()

    n_cluster = len(set([img["cluster"] for img in list_dict_img]))
    for num in range(n_cluster):
        list_cluster = cluster_list(list_dict_img, num)
        if num == 0:
            excellent_img, good_img, rubbish_img = sorting_without_cluster(list_cluster)
            excellent_list += excellent_img
            good_list += good_img
            rubbish_list += rubbish_img
        else :
            pourcentage = math.ceil(len(list_cluster)*0.1)
            good_img, rubbish_img = sorting_cluster(list_cluster, pourcentage)
            good_list += good_img
            rubbish_list += rubbish_img

    return excellent_list, good_list, rubbish_list


# if __name__ == '__main__':
#     print("*********************** START *****************************")
#     test_cluster = [{"image_name" : "20200830_103939.jpg", "nb_faces" : 0, "mos_score" : 69, "cluster" : 0},
#         {"image_name" : "20200908_163321.jpg", "nb_faces" : 0, "mos_score" : 72, "cluster" : 1},
#         {"image_name" : "20220925_115256.jpg", "nb_faces" : 0, "mos_score" : 42, "cluster" : 1},
#         {"image_name" : "20200830_103936.jpg", "nb_faces" : 0, "mos_score" : 82, "cluster" : 0},
#         {"image_name" : "20220925_115304.jpg", "nb_faces" : 0, "mos_score" : 84, "cluster" : 2},
#         {"image_name" : "20220925_124332.jpg", "nb_faces" : 0, "mos_score" : 85, "cluster" : 2},
#         {"image_name" : "20220925_115258.jpg", "nb_faces" : 0, "mos_score" : 72, "cluster" : 2},
#         {"image_name" : "20220925_115259.jpg", "nb_faces" : 0, "mos_score" : 65, "cluster" : 2},
#         {"image_name" : "20200908_163333.jpg", "nb_faces" : 0, "mos_score" : 64, "cluster" : 0},
#         {"image_name" : "20200908_163330.jpg", "nb_faces" : 0, "mos_score" : 35, "cluster" : 0},
#         {"image_name" : "20220925_115258.jpg", "nb_faces" : 0, "mos_score" : 72, "cluster" : 3},
#         {"image_name" : "20220925_115659.jpg", "nb_faces" : 0, "mos_score" : 65, "cluster" : 3},
#         {"image_name" : "20200908_163331.jpg", "nb_faces" : 0, "mos_score" : 64, "cluster" : 4},
#         {"image_name" : "20200908_163338.jpg", "nb_faces" : 0, "mos_score" : 35, "cluster" : 4},
#         {"image_name" : "20200908_163337.jpg", "nb_faces" : 0, "mos_score" : 35, "cluster" : 4}]
#     excellent_list, good_list, rubbish_list = sorting_img(test_cluster)
#     print(f'Excellent photos: {excellent_list}')
#     print('-----')
#     print(f'Good photos: {good_list}')
#     print('-----')
#     print(f'Rubbish photos: {rubbish_list}')
#     print("*********************** END *****************************")
