def cluster_list(list_dict_img, number_cluster):
    """
    The function returns a list of image-dictionnaries from the same cluster,
    given a list of image-dictionnaries and the cluster number.
    """
    cluster_list = list()
    for img in list_dict_img:
        if img["cluster"] == number_cluster:
            cluster_list.append(img)
    return cluster_list



def sorting_cluster(cluster_list, number, scores):
    """
    The function returns 2 lists :
    1- a list of the names of the GOOD images
    2- a list of the names of the RUBBISH images
    given a cluster (format: list of image-dictionaries) and the number of GOOD images to choose.
    """
    list_scores = [img[scores] for img in cluster_list]
    if number == 1:
        max_scores = max(list_scores)
        good_img = [img["image_name"] for img in cluster_list if img[scores] == max_scores]
        rubbish_img = [img["image_name"] for img in cluster_list if img[scores] != max_scores]
    else:
        list_scores.sort(reverse = True)
        max_scores = list_scores[:number]
        good_img = [img["image_name"] for img in cluster_list if img[scores] in max_scores]
        rubbish_img = [img["image_name"] for img in cluster_list if img[scores] not in max_scores]
    return good_img, rubbish_img



def sorting_without_cluster_landscape(cluster_list):
    """
    The function returns 3 lists :
    1- a list of the names of the EXCELLENT images (mos scores >= 70),
    1- a list of the names of the GOOD images (60 <= mos scores < 70),
    2- a list of the names of the RUBBISH images (mos score < 60)
    given a list of image-dictionaries
    """
    excellent_img = list()
    good_img = list()
    rubbish_img = list()
    for img in cluster_list:
            if img['MOS'] < 60:
                rubbish_img.append(img['image_name'])
            elif img['MOS'] >= 60 and img['MOS'] < 70 :
                good_img.append(img['image_name'])
            else:
                excellent_img.append(img['image_name'])
    return excellent_img, good_img, rubbish_img



def sorting_without_cluster_humain(cluster_list):
    """
    The function returns 3 lists :
    1- a list of the names of the EXCELLENT images,
    1- a list of the names of the GOOD images,
    2- a list of the names of the RUBBISH images
    given a list of image-dictionaries.
    Following their final score, the best 10% goes to EXCELLENT, the following 50% to GOOD and the last 40% to RUBBISH.
    """
    img_scores = [img["final_score"] for img in cluster_list]
    img_scores.sort(reverse = True)

    excellent_scores = img_scores[:int(len(img_scores) * 0.1)]
    good_scores = img_scores[int(len(img_scores) * 0.1) : int(len(img_scores) * 0.5)]
    rubbish_scores = img_scores[int(len(img_scores)* 0.5):]

    excellent_img = [img["image_name"] for img in cluster_list if img["final_score"] in excellent_scores]
    good_img = [img["image_name"] for img in cluster_list if img["final_score"] in good_scores]
    rubbish_img = [img["image_name"] for img in cluster_list if img["final_score"] in rubbish_scores]
    return excellent_img, good_img, rubbish_img




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
