
import os
import pickle
from tensorflow.keras import Model, models

from colorama import Fore, Style
import time
import glob


def save_model_4(model: Model = None,
               params: dict = None,
               metrics: dict = None) -> None:
    """
    persist trained model, params and metrics
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    print(Fore.BLUE + "\nSave model 4 to local disk..." + Style.RESET_ALL)

    local_path_model_4 = os.path.join(os.environ.get("LOCAL_PROJECT_PATH"),"registry","trained_model_4")

    # save params
    if params is not None:
        params_path = os.path.join(local_path_model_4, "params", timestamp + ".pickle")
        print(f"- params path: {params_path}")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # save metrics
    if metrics is not None:
        metrics_path = os.path.join(local_path_model_4, "metrics", timestamp + ".pickle")
        print(f"- metrics path: {metrics_path}")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    # save model
    if model is not None:
        model_path = os.path.join(local_path_model_4, "models", timestamp)
        print(f"- model path: {model_path}")
        model.save(model_path)

    print("\n✅ model saved locally")

    return None

def load_model_4() -> Model:
    """
    load the latest saved model, return None if no model found
    """
    print(Fore.BLUE + "\nLoad model 4 from local disk..." + Style.RESET_ALL)

    local_path_model_4 = os.path.join(os.environ.get("LOCAL_PROJECT_PATH"),"registry","trained_model_4")

    # get latest model version
    model_directory = os.path.join(local_path_model_4, "models")

    results = glob.glob(f"{model_directory}/*")
    if not results:
        return None

    model_path = sorted(results)[-1]
    print(f"- path: {model_path}")

    model = models.load_model(model_path)
    print("\n✅ model loaded from disk")

    return model
