from colorama import Fore, Style
import numpy as np


print(Fore.BLUE + "\nLoading tensorflow..." + Style.RESET_ALL)

from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import convert_to_tensor, expand_dims

print(f"\n✅ tensorflow loaded ")

from typing import Tuple

def initialize_model_4() -> Model :
    """
    Initialize model 4
    """
    print(Fore.BLUE + "\nInitialize model..." + Style.RESET_ALL)

    model = Sequential()

    model.add(layers.Conv2D(120, (2,2), input_shape=(120, 128, 1), activation="relu", padding = "same"))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    model.add(layers.Conv2D(50, (2,2), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(32, (2,2), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(16, (2,2), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation = "relu"))
    model.add(layers.Dense(4, activation = "softmax"))

    return model

def compile_model_4(model) :
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics = 'accuracy')

    return model

def train_model_4(model: Model,
                X: list,
                y: list,
                batch_size=16,
                patience=4,
                validation_split=0.3) -> Tuple[Model, dict] :

    es = EarlyStopping(patience = patience)

    """
    Fit model and return a the tuple (fitted_model, history)
    """


    history = model.fit(convert_to_tensor(X),
                        convert_to_tensor(y),
                        validation_split=validation_split, # /!\ LAST 30% of train indexes are used as validation
                        batch_size=batch_size,
                        epochs=100,
                        callbacks = [es])

    return model, history

def evaluate_model_4(model: Model,
                   X,
                   y):
    """
    Evaluate trained model performance on dataset
    """

    if model is None:
        print(f"\n❌ no model to evaluate")
        return None

    metrics = model.evaluate(
        x=convert_to_tensor(X),
        y=y,
        verbose=1,
        # callbacks=None,
        return_dict=True)

    return metrics
