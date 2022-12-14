import numpy as np


def emotion_score(list_dict_img):
    """
    The function append the list of the image-dictionnaries by adding an EMOTION SCORE.
    List of emotions that the function receives : ['happy', 'surprise', 'neutral', 'angry', 'disgust', 'fear', 'sad']
    Calcul of the emotion scores :
    - 'happy, 'surprise' => score of 1.2
    - 'neutral' => score of 1.0
    - 'angry', 'disgust', 'fear', 'sad' => score of 0.8
    """
    for img in list_dict_img:
        emotion_list_score = list()
        for emotion in img["cropped_faces"]:
            'angry', 'disgust', 'fear', 'sad'
            if emotion["emotion"] in ["happy", "surprise"]:
                emotion_list_score.append(1.2)
            elif emotion["emotion"] == "neutral":
                emotion_list_score.append(1.0)
            else:
                emotion_list_score.append(0.8)
        em_score = sum(emotion_list_score)/img["nb_faces"]
        img["score_emotion"] = em_score
    return list_dict_img



def eyes_score(list_dict_img):
    """
    The function appends the list of the image-dictionnaries by adding an EYES SCORE.
    The function loops through the number of eyes for each face, received in the list of image_dictionnary,
    and gives a score following the number of eyes:
    - 2 eyes (or more !) => score of 1.1
    - 1 eye => score of 0.9
    - 0 eye => score of 0.8
    """
    for img in list_dict_img:
        eyes_list_score = list()
        for eyes in img["cropped_faces"]:
            if eyes["nb_eyes"] >= 2:
                eyes_list_score.append(1.1)
            elif eyes["nb_eyes"] == 1:
                eyes_list_score.append(0.9)
            else:
                eyes_list_score.append(0.8)
        sc_eyes = sum(eyes_list_score)/img["nb_faces"]
        img["score_eyes"] = sc_eyes
    return list_dict_img



def standardize_score(list_dict_img, score_clé):
    """
    The function returns the list of image-dictionnaries by replacing the MOS, emotion and eyes scores by their standardized score.
    """
    score_list = [img[score_clé] for img in list_dict_img if score_clé in img]
    score_mean = sum(score_list)/len(score_list)
    score_calcul = [(s - score_mean) for s in score_list]
    sum_minmax = abs(np.min(score_calcul)) + abs(np.max(score_calcul))
    for img in list_dict_img:
        if score_clé in img :
            score_standardize = ((img[score_clé]-score_mean)/sum_minmax) + 1
            img[score_clé] = score_standardize
    return list_dict_img



def final_score(list_dict_img, k1=1, k2=1, k3=1):
    """
    The function appends the list of the image-dictionnaries by adding an FINAL SCORE.
    Final score = (mos_score ** k1) * (emotion_score ** k2) * (eyes_score ** k3)
    k1, k2, k3 = hyperparameters that can be modified to give more importance to a score or another.
    """
    for img in list_dict_img:
        scores = list()
        scores.append(img["MOS"])
        scores.append(img["score_emotion"])
        scores.append(img["score_eyes"])
        score_final = (scores[0]**k1) * (scores[1]**k2) * (scores[2]**k3)
        img["final_score"] = score_final
    return list_dict_img



def calculating_scores(list_dict_img):
    """
    The fonction returns the list of the image-dictionnaries it received, appending the following scores:
    standardize mos, emotion and eyes scores, and the final score.
    """
    list_dict_img = emotion_score(list_dict_img)
    list_dict_img = eyes_score(list_dict_img)
    list_dict_img = standardize_score(list_dict_img, "score_emotion")
    list_dict_img = standardize_score(list_dict_img, "score_eyes")
    list_dict_img = standardize_score(list_dict_img, "MOS")
    list_dict_img = final_score(list_dict_img)
    return list_dict_img



# if __name__ == '__main__' :
#     print('*********** START ***************')
#     print('*********** START dict 1/4 ***************')
#     dict_G = [{'image_name': '20200830_103939.jpg',
#     'cluster': 0,
#     'nb_faces': 1,
#     'cropped_faces': [{'emotion': 'happy', 'nb_eyes': 6}],
#     'MOS': 69,
#     'score_emotion': 1.2,
#     'score_eyes': 1.1},
#     {'image_name': '20200908_163321.jpg',
#     'cluster': 1,
#     'nb_faces': 3,
#     'cropped_faces': [{'emotion': 'happy', 'nb_eyes': 1},
#     {'emotion': 'surprise', 'nb_eyes': 1},
#     {'emotion': 'neutral', 'nb_eyes': 1}],
#     'MOS': 72,
#     'score_emotion': 1.1333333333333333,
#     'score_eyes': 0.9666666666666667},
#     {'image_name': '20220925_115256.jpg',
#     'cluster': 1,
#     'nb_faces': 3,
#     'cropped_faces': [{'emotion': 'happy', 'nb_eyes': 10},
#     {'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'neutral', 'nb_eyes': 1}],
#     'MOS': 42,
#     'score_emotion': 1.1333333333333333,
#     'score_eyes': 1.0333333333333334},
#     {'image_name': '20200830_103936.jpg', 'cluster': 0, 'nb_faces': 0, 'MOS': 42},
#     {'image_name': '20220925_115304.jpg', 'cluster': 2, 'nb_faces': 0, 'MOS': 84},
#     {'image_name': '20220925_124332.jpg', 'cluster': 2, 'nb_faces': 0, 'MOS': 85},
#     {'image_name': '20220925_115258.jpg', 'cluster': 2, 'nb_faces': 0, 'MOS': 72},
#     {'image_name': '20220925_115259.jpg',
#     'cluster': 3,
#     'nb_faces': 5,
#     'cropped_faces': [{'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'sad', 'nb_eyes': 0},
#     {'emotion': 'sad', 'nb_eyes': 2},
#     {'emotion': 'neutral', 'nb_eyes': 0}],
#     'MOS': 89,
#     'score_emotion': 1.0,
#     'score_eyes': 0.9799999999999999},
#     {'image_name': '20200908_163333.jpg',
#     'cluster': 3,
#     'nb_faces': 5,
#     'cropped_faces': [{'emotion': 'happy', 'nb_eyes': 1},
#     {'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'sad', 'nb_eyes': 2},
#     {'emotion': 'neutral', 'nb_eyes': 2},
#     {'emotion': 'neutral', 'nb_eyes': 2}],
#     'MOS': 75,
#     'score_emotion': 1.04,
#     'score_eyes': 1.06},
#     {'image_name': '20200908_163330.jpg',
#     'cluster': 3,
#     'nb_faces': 5,
#     'cropped_faces': [{'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'happy', 'nb_eyes': 1},
#     {'emotion': 'happy', 'nb_eyes': 2}],
#     'MOS': 33,
#     'score_emotion': 1.2,
#     'score_eyes': 1.06},
#     {'image_name': '20220925_115258.jpg',
#     'cluster': 3,
#     'nb_faces': 5,
#     'cropped_faces': [{'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'neutral', 'nb_eyes': 2},
#     {'emotion': 'sad', 'nb_eyes': 1},
#     {'emotion': 'happy', 'nb_eyes': 2}],
#     'MOS': 40,
#     'score_emotion': 1.08,
#     'score_eyes': 1.06},
#     {'image_name': '20220925_115659.jpg',
#     'cluster': 4,
#     'nb_faces': 1,
#     'cropped_faces': [{'emotion': 'sad', 'nb_eyes': 2}],
#     'MOS': 89,
#     'score_emotion': 0.8,
#     'score_eyes': 1.1},
#     {'image_name': '20200908_163331.jpg',
#     'cluster': 4,
#     'nb_faces': 1,
#     'cropped_faces': [{'emotion': 'happy', 'nb_eyes': 0}],
#     'MOS': 70,
#     'score_emotion': 1.2,
#     'score_eyes': 0.8},
#     {'image_name': '20200908_163337.jpg',
#     'cluster': 4,
#     'nb_faces': 1,
#     'cropped_faces': [{'emotion': 'neutral', 'nb_eyes': 1}],
#     'MOS': 65,
#     'score_emotion': 1.0,
#     'score_eyes': 0.9}]
#     print('dict OK')
#     print('*********** END dict 1/4 ***************')
#     print('-------')

#     print('*********** START MOS standardize 2/4 ***************')
#     dict_test = standardize_score(dict_G, "MOS")
#     s_mos_stand = [img["MOS"] for img in dict_test if "MOS" in img]
#     print(s_mos_stand)
#     print('*********** END MOS standardize 2/4 ***************')
#     print('-------')

#     print('*********** START emotion standardize 3/4 ***************')
#     dict_test = standardize_score(dict_G, "score_emotion")
#     s_mos_stand = [img["score_emotion"] for img in dict_test if "score_emotion" in img]
#     print(s_mos_stand)
#     print('*********** END emotion standardize 3/4 ***************')
#     print('-------')

#     print('*********** START eyes standardize 4/4 ***************')
#     dict_test = standardize_score(dict_G, "score_eyes")
#     s_mos_stand = [img["score_eyes"] for img in dict_test if "score_eyes" in img]
#     print(s_mos_stand)
#     print('*********** END eyes standardize 4/4 ***************')
#     print('-------')

#     print('*********** START global scores 4/4 ***************')
#     dict_scores = calculating_scores(dict_G)
#     s_emotion = [img["score_emotion"] for img in dict_G if "score_emotion" in img]
#     s_eyes = [img["score_eyes"] for img in dict_G if "score_eyes" in img]
#     s_mos = [img["MOS"] for img in dict_G]
#     s_final = [img["final_score"] for img in dict_G]
#     print(f'score emotion: {s_emotion}')
#     print('-------')
#     print(f'score eyes: {s_eyes}')
#     print('-------')
#     print(f'score mos: {s_mos}')
#     print('-------')
#     print(f'score final: {s_final}')
#     print('-------')
#     print(dict_scores)
#     print('*********** END global scores 4/4 ***************')



#     print('*********** START ***************')
#     print('*********** START dict 1/2 ***************')
#     dict_test = [{'image_name': '20200830_103939.jpg',
#     'cluster': 0,
#     'nb_faces': 1,
#     'cropped_faces': [{'emotion': 'happy', 'nb_eyes': 6}],
#     'MOS': 69},
#     {'image_name': '20200908_163321.jpg',
#     'cluster': 1,
#     'nb_faces': 3,
#     'cropped_faces': [{'emotion': 'happy', 'nb_eyes': 1},
#     {'emotion': 'surprise', 'nb_eyes': 1},
#     {'emotion': 'neutral', 'nb_eyes': 0}],
#     'MOS': 72},
#     {'image_name': '20220925_115256.jpg',
#     'cluster': 1,
#     'nb_faces': 3,
#     'cropped_faces': [{'emotion': 'happy', 'nb_eyes': 10},
#     {'emotion': 'happy', 'nb_eyes': 2},
#     {'emotion': 'neutral', 'nb_eyes': 1}],
#     'MOS': 42}]
#     print('*********** END dict 1/2 ***************')
#     print('-------')
#     print('*********** START score eyes 2/2 ***************')
#     dict_emotion = eyes_score(dict_test)
#     for img in dict_emotion:
#         print(img["image_name"], img["score_eyes"])
#     print('*********** END score eyes 2/2 ***************')
#     print('-------')
#     print('*********** END ***************')
