from colorama import Fore, Style
import pandas as pd
import numpy as np
import os

from utils_model_4 import initialize_model_4, compile_model_4, train_model_4, evaluate_model_4
from preprocessor_model_4 import preprocess_model_4
from get_training_data_CMU import get_data_folder_CMU
from registry_model_4 import save_model_4, load_model_4

from tensorflow.keras.utils import to_categorical
from tensorflow import expand_dims, convert_to_tensor
from tensorflow.keras import Model #TODO : voir si utile


from sklearn.model_selection import train_test_split


encoding_dico = {"straight" :0,
                 "up" : 1 ,
                 "left" : 2,
                 "right" : 3}

def full_train_model_4(train_data_path):

    #Look for CMU training dataset
    print(Fore.BLUE + "\nGetting CMU dataset for training..." + Style.RESET_ALL)

    X,y = get_data_folder_CMU(train_data_path)

    print(f"\n✅ {len(X)} pictures retrieved from CMU dataset.")

    #preprocessing target
    y_new=[]
    for dico in y :
        y_new.append(dico["pose"])

    y_df = pd.DataFrame (y_new, columns = ['pose'])
    y_df["pose"] = y_df["pose"].map(encoding_dico)

    y_cat = to_categorical(y_df)


    #preprocessing images (X)
    normalization_coefficient = np.max(X)
    X_normalized = [expand_dims(tensor, 2)/normalization_coefficient for tensor in X]

    # Split data into train & test sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_cat,
                                                        test_size = 0.3,
                                                        random_state = 42)

    #Initialize model
    model = initialize_model_4()

    #compile
    model = compile_model_4(model)


    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

    model, history = train_model_4(model,
                X_train,
                y_train,
                batch_size=16,
                patience=1, #### A CHANGER POUR patience=4
                validation_split=0.3)

    print(f"\n✅ model trained on {len(X_train)} pictures.")

    print(Fore.BLUE + f"\nEvaluate model ..." + Style.RESET_ALL)

    metrics = evaluate_model_4(model, X_test, y_test)

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"\n✅ model evaluated: loss: {round(loss, 2)} accuracy: {round(accuracy, 2)}")

    #save trained model and evaluation
    save_model_4(model = model, params = None, metrics = metrics)


    return model, history





def pred_model_4(dico):
    """
    Make a prediction using the saved trained model in folder trained_model_4
    Input :
    return append dico
    - orientation key in list of dico
    """

    print("\n⭐️ Use case: predict")

    #Loading pre-trained model
    model = load_model_4()


    # Getting list of dictionnaries / 1 dictionnary = 1 face + information
    faces_list = dico.get("cropped_faces", "No such key as 'cropped_faces'")

    if faces_list == "No such key as 'cropped_faces'":
        print("Wrong input for model 4")

    if len(faces_list)==0: # Condition à faire sur le %age de la face
        print("No relevant face to analyse by model 4")

    #In the case of a simple portrait (only one face detected)
    #ATTENTION

    if len(faces_list)==1:
        input_face = faces_list[0].get("face", "Review key names of input dictionnary")

        if input_face == "Review key names of input dictionnary":
            print("MODEL 4 -> Review key names of input dictionnary")

        face_processed = preprocess_model_4(input_face)
        y_pred_proba = model.predict(convert_to_tensor([face_processed]), verbose=0) #import à faire pour le .predict ?
        associated_position_proba = {"straight_proba" :y_pred_proba[0][0],
                                        "up_proba" : y_pred_proba[0][1] ,
                                        "left_proba" : y_pred_proba[0][2],
                                        "right_proba" : y_pred_proba[0][3]}

        dico["cropped_faces"][0]["associated_position_proba"] = associated_position_proba
        print("\n✅ prediction done: ", dico)
        return dico



    #In the case of a group portrait (more than one face detected)
    elif len(faces_list)>1:
        print("Wrong input for model 4")

        for dico in faces_list :

            pass






if __name__ == '__main__':
    pass


    #Test training - REMOVE PASS
    #train_data_path = "/Users/jeannebaron/code/duchesgo/image_selection/draft/CMU_data"
    #full_train_model_4(train_data_path)


    #Test prediction - REMOVE PASS
    #dico_test_1 = {'initial_image' : None,
    #            'image_with_faces ' : None,
    #            'nb_faces' : 1,
    #            'cropped_faces' : [{"face_id" : "face_1",
    #                                "face" : "/Users/jeannebaron/code/duchesgo/image_selection/draft/image_test.jpeg",
    #                                "relative_surface": 0.5},
    #                               ]
    #            }
    #dico_test_2= {'initial_image' : None,
    #            'image_with_faces ' : None,
    #            'nb_faces' : 1,
    #            'cropped_faces' : [{"face_id" : "face_1",
    #                                "face" : "/Users/jeannebaron/code/duchesgo/image_selection/draft/image_test.jpeg",
    #                                "relative_surface": 0.5},
    #                               {"face_id" : "face_2",
    #                                "face" : None,
    #                                "relative_surface": 0.5}]
    #            }
    #print(dico_test_1)
    #pred_model_4(dico_test_1)
