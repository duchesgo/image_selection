def cluster_list(list_dict_img, number_cluster):
    """Function that returns a list of image dictionnary from the same cluster,
    given a list of image dictionnary and the number of the cluster """

    cluster_list = list()
    for image in list_dict_img:
        if image["cluster"] == number_cluster:
            cluster_list.append(image)

    return cluster_list


def sorting_without_cluster(cluster_list):
    """Function that returns 3 lists :
    1- list of names of the BEST images to save (mos scores >= 70),
    1- list of names of the  images to save (60 <= mos scores < 70),
    2- list of names of the images to delete (mos score < 60)
    given a list of image dictionary"""
    delete_img = list()
    save_img = list()
    bestof_img = list()

    for image in cluster_list:

            if image['MOS'] < 60:
                delete_img.append(image['image_name'])
            elif image['MOS'] >= 60 and image['MOS'] < 70 :
                save_img.append(image['image_name'])
            else:
                bestof_img.append(image['image_name'])

    return bestof_img, save_img, delete_img



def sorting_cluster(cluster_list, number):
    """Function that returns 2 lists :
    1- list of names of the images to save (best mos scores),
    2- list of names of the images to delete
    given a cluster (format: list of image dictionary) and the number of best images to return"""

    list_mos = [image["MOS"] for image in cluster_list]
    if number == 1:
        max_mos = max(list_mos)
        good_img = [img["image_name"] for img in cluster_list if img['MOS'] == max_mos]
        rubbish_img = [img["image_name"] for img in cluster_list if img['MOS'] != max_mos]
    else:
        list_mos.sort(reverse = True)
        max_mos = list_mos[:number]
        good_img = [img["image_name"] for img in cluster_list if img["MOS"] in max_mos]
        rubbish_img = [img["image_name"] for img in cluster_list if img["MOS"] not in max_mos]

    return good_img, rubbish_img




# if __name__ == '__main__':

#     print("*********************** START - list of cluster 1/3 *****************************")
#     test_cluster = [{"image_name" : "20200830_103939.jpg", "nb_faces" : 0, "MOS" : 69, "cluster" : 0},
#         {"image_name" : "20200908_163321.jpg", "nb_faces" : 0, "MOS" : 72, "cluster" : 1},
#         {"image_name" : "20220925_115256.jpg", "nb_faces" : 0, "MOS" : 42, "cluster" : 1},
#         {"image_name" : "20200830_103939.jpg", "nb_faces" : 0, "MOS" : 82, "cluster" : 1},
#         {"image_name" : "20220925_115304.jpg", "nb_faces" : 0, "MOS" : 84, "cluster" : 2},
#         {"image_name" : "20220925_124332.jpg", "nb_faces" : 0, "MOS" : 85, "cluster" : 2},
#         {"image_name" : "20220925_115258.jpg", "nb_faces" : 0, "MOS" : 72, "cluster" : 2},
#         {"image_name" : "20220925_115260.jpg", "nb_faces" : 0, "MOS" : 65, "cluster" : 2},
#         {"image_name" : "20200908_163333.jpg", "nb_faces" : 0, "MOS" : 64, "cluster" : 0},
#         {"image_name" : "20220925_115261.jpg", "nb_faces" : 0, "MOS" : 35, "cluster" : 2},
#         {"image_name" : "20220925_115262.jpg", "nb_faces" : 0, "MOS" : 54, "cluster" : 2},
#         {"image_name" : "20200908_163330.jpg", "nb_faces" : 0, "MOS" : 35, "cluster" : 0},
#         {"image_name" : "20200908_163335.jpg", "nb_faces" : 0, "MOS" : 87, "cluster" : 0}]
#     list_cluster1 = cluster_list(test_cluster, 1)
#     list_cluster2 = cluster_list(test_cluster, 2)
#     print(f'Cluster 1: {list_cluster1}')
#     print(f'Cluster 2: {list_cluster2}')
#     print("*********************** END - list of cluster 1/3 *****************************")
#     print("*")
#     print("*********************** START - sorting cluster 2/3 *****************************")
#     save_test1, delete_test1 = sorting_cluster(list_cluster1, 1)
#     save_test2, delete_test2 = sorting_cluster(list_cluster2, 2)
#     print(f'Cluster 1 - images to save: {save_test1}')
#     print(f'Cluster 1 - images to delete: {delete_test1}')
#     print('------')
#     print(f'Cluster 2 - images to save: {save_test2}')
#     print(f'Cluster 2 - images to delete: {delete_test2}')
#     print("*********************** END - sorting cluster 2/3 *****************************")
#     print("*")
#     print("*********************** START - sorting without cluster 3/3 *****************************")
#     bestof_img_test, save_img_test, delete_img_test = sorting_without_cluster(test_cluster)
#     print(f'Without cluster - BEST images to save: {bestof_img_test}')
#     print(f'Without cluster - images to save: {save_img_test}')
#     print(f'Without cluster - images to delete: {delete_img_test}')
#     print("*********************** START - sorting without cluster 3/3 *****************************")
