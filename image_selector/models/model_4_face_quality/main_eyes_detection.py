import cv2
import os

eye_cascPath = os.path.join(os.environ.get("LOCAL_PROJECT_PATH"),"registry","eye_cascade")
eyeCascade = cv2.CascadeClassifier(eye_cascPath)


def eyes_detector(image):
    eyes = eyeCascade.detectMultiScale(
                    image,
                    scaleFactor=1.1,
                    minNeighbors=2,
                    minSize=(10,8),
                    # flags = cv2.CV_HAAR_SCALE_IMAGE
                )

    return len(eyes)

def pred_eyes(dico):
    """
    Takes dico with cropped faces
    return append dico with key "nb_eyes" -> number of eyes
    """

    # Getting list of dictionnaries / 1 dictionnary = 1 face + information
    faces_list = dico.get("cropped_faces", "No such key as 'cropped_faces'")


    if faces_list == "No such key as 'cropped_faces'":
        print("Wrong input for model 4")

    elif len(faces_list)==0: # Condition à faire sur le %age de la face
        print("No relevant face to analyse by model 4")
        return dico


    #In the case of a simple portrait (only one face detected)
    elif len(faces_list)==1:
        input_face = faces_list[0].get("cropped_face", "Review key names of input dictionnary") #modif JB, changer face pour cropped_face

        if input_face == "Review key names of input dictionnary":
            print("MODEL 4 -> Review key names of input dictionnary")


        #How many eyes ?
        nb_eyes = eyes_detector(input_face)
        dico["cropped_faces"][0]["nb_eyes"] = nb_eyes
        #print("\n✅ prediction done for 1 face: ", dico)
        return dico

    #In the case of a group portrait (more than one face detected)
    elif len(faces_list)>1:

        for face in faces_list :
            input_face = face.get("cropped_face", "Review key names of input dictionnary") #face=np.array - modif JB, changer face pour cropped_face
            nb_eyes = eyes_detector(input_face)
            face["associated_position_proba"] = nb_eyes

        #print("\n✅ prediction done for more than 1 face: ", dico)
        return dico
