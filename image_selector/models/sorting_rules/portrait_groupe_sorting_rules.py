def emotion_score(list_dict_img):
    for img in list_dict_img:
        emotion_list_score = list()

        if img["nb_faces"] != 0:
            for emotion in img["cropped_face"]:
                # si 'happy' ou 'surprise' => note de 1,2
                if emotion["emotion"] in ["happy", "surprise"]:
                    emotion_list_score.append(1.2)
                # si 'neutral' => note de 1
                elif emotion["emotion"] == 'neutral':
                    emotion_list_score.append(1.0)
                # si 'angry', 'disgust', 'fear', 'sad' => note 0.8
                else:
                    emotion_list_score.append(0.8)
            emotion_score = sum(emotion_list_score)/img["nb_faces"]
            img["emotion_score"] = emotion_score
            return list_dict_img
