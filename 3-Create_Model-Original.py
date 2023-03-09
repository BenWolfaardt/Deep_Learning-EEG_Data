"""
Created on 01/10/2018
Updated on 07/09/2022
Authors: William Geuns & Ben Wolfaardt
"""

import numpy as np
import pickle

from envyaml import EnvYAML
from keras.backend import clear_session
from keras.callbacks import EarlyStopping,  ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from nptyping import NDArray
from sklearn.model_selection import StratifiedKFold
from typing import Any


class Create:
    def __init__(self) -> None:
        # config
        self.os: str = ""
        self.config: EnvYAML = None
        self.experiment: str = ""
        # paths
        self.pickles: str = ""
        self.models: str = ""
        # experiment details
        self.name: str = ""
        self.participants: list[int] = []
        self.participant: int = 0
        self.triggers: list[str] = []
        self.version: str = ""
        self.comparison: int = 0
        # model
        self.kfolds: int = 0
        self.epochs: int = 0
        self.callbacks: list[Any] = []
        self.model: Sequential = Sequential()
        self.X: NDArray = None
        self.y: NDArray = None
        # filenames
        self.filename_load: str = ""
        self.filename_save: str = ""

    def set_callbacks(self) -> None:
        # Callback for writing checkpoints during training
        path_checkpoint = 'checkpoint.keras'
        callback_checkpoint = ModelCheckpoint(
            filepath=path_checkpoint,
            monitor='val_loss',
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
        )
        # Callback for stopping the optimization when performance worsens on the validation-set
        callback_early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=7,
            verbose=1,
            # restore_best_weights=True,
        )
        # Callback for writing the TensorBoard log during training.
        callback_tensorboard = TensorBoard(
            log_dir='./logs/original/',
            histogram_freq=0,
            write_graph=False,
        )
        # This callback reduces the learning-rate for the optimizer if the validation-loss has not improved
        # since the last epoch (as indicated by patience=0). The learning-rate will be reduced by multiplying
        # it with the given factor. We set a start learning-rate of 1e-3 above, so multiplying it by 0.1 gives
        # a learning-rate of 1e-4. We don't want the learning-rate to go any lower than this.
        callback_reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            min_lr=1e-4,
            patience=0,
            verbose=1,
        )

        self.callbacks = [
            callback_early_stopping,
            # callback_checkpoint,
            # callback_tensorboard,
            # callback_reduce_lr
        ]

    def load_yaml(self) -> None:
        self.config = EnvYAML("./setup/config.yaml", strict=False)

    def populate_config(self) -> None:
        # TODO
        #   Parse in as arg using argparse
        self.os = "mac_m1"
        self.experiment = "libet"

        self.pickles = self.config[f"os.{self.os}.io_paths.pickle_files"]
        self.models = self.config[f"os.{self.os}.io_paths.model_files"]
        # TODO use name for folder or something
        self.name = self.config[f"experiment.details.{self.experiment}.name"]
        self.participants = self.config[f"experiment.details.{self.experiment}.participants"]
        self.triggers = self.config[f"experiment.details.{self.experiment}.triggers"]
        self.version = self.config[f"experiment.details.{self.experiment}.version"]
        self.kfolds = self.config[f"model_parameters.kfolds"]
        self.epochs = self.config[f"model_parameters.epochs"]
        self.comparison = self.config[f"model_parameters.comparison"]

    # Load preprocessed training/test pickles
    def load_data(self) -> None:
        # TODO
        #   Don't save self.comparison as number but rather as value of dict {0: "All", 1: "Single"}
        # 0: "All" (all participants' data combined)
        if self.comparison == 0:
            self.filename_load = ""
        # 1: "Single" (each participant's data separate)
        elif self.comparison == 1:
            self.filename_load = f"{self.participant}-"

        with open(f"{self.pickles}/{self.version}/{self.comparison}/X-{self.filename_load}Training.pickle", 'rb') as f:
            self.X = pickle.load(f)  # shape: (1369, 63, 450, 1)
            self.X = np.asarray(self.X)
        with open(f"{self.pickles}/{self.version}/{self.comparison}/y-{self.filename_load}Training.pickle", 'rb') as f:
            self.y = pickle.load(f)  # shape: (#TODO)
            self.y = np.transpose(self.y)

    def create_model(self) -> None:
        self.model.add(Conv2D(512, (3, 3), input_shape=self.X.shape[1:]))  # input_shape: (63, 450, 1)
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        
        self.model.add(Conv2D(256, (3, 3)))
        self.model.add(Activation('relu')) 
        
        self.model.add(Conv2D(128, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
            
        self.model.add(Flatten())   
        
        self.model.add(Dense(32))
        self.model.add(Activation('relu'))
        
        self.model.add(Dense(16))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2)) 
        
        # Last dense 1ayers must have number of classes in data in the parenthesis
        self.model.add(Dense(len(self.triggers)))

        # TODO
        #   The activation function is a result of the amount of Triggers in the last layer
        #       For a binary classification: sigmoid
        #       For more than 2x trigger classification: softmax
        #   Add automatic logic for this
        self.model.add(Activation('sigmoid'))

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        # TODO
        #   Graphical depict model - it was wanting more libraries to be installed
        #       https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/
        #       https://www.graphviz.org/
        # from keras.utils.vis_utils import plot_model
        # plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        # print(model.summary())

    # Train model with K-fold cross validation with early stopping
    def kfold_cross_validation(self) -> None:
        kfold: StratifiedKFold = StratifiedKFold(n_splits=self.kfolds, shuffle=True, random_state=42)
        cvscores: list = []

        if self.comparison == 1:
            print(f"\nTraining model for participant: {self.participant}/{len(self.participants)}")

        for index, (train_index, validation_index) in enumerate(kfold.split(self.X, self.y)):
            print(f"Training on fold: {str(index+1)}/{self.kfolds}\n")
            
            # Generate batches
            Xtrain, Xval = self.X[train_index], self.X[validation_index]
            ytrain, yval = self.y[train_index], self.y[validation_index]

            self.model.fit(
                Xtrain,
                ytrain,
                epochs=self.epochs,
                batch_size=10,
                verbose=1,
                validation_data=(Xval, yval),
                callbacks=self.callbacks,
                shuffle=True
            )
            # Should you want an array of some of the below standard output
            # eg. 1s 137ms/step - loss: 2.0092e-04 - accuracy: 1.0000 - val_loss: 5.2966e-04 - val_accuracy: 1.0000
            # you can make use of:
            #   history = self.model.fit(...
            #   accuracy_history = history.history['accuracy']
            #   val_accuracy_history = history.history['val_accuracy']

            scores = self.model.evaluate(Xval, yval, verbose=1)
            cvscores.append(scores[1] * 100)

        print(f"\nEvaluated accuracy: {np.mean(cvscores):.2f}% (+/- {np.std(cvscores):.2f}%) on {self.kfolds} k-folds")

    def save_model(self) -> None:
        # 0: "All" (all participants' data combined)
        if self.comparison == 0:
            self.filename_save = f"{self.experiment}"
        # 1: "Single" (each participant's data separate)
        elif self.comparison == 1:
            self.filename_save = f"{self.participant}"

        self.model.save(f"{self.models}/{self.version}/{self.comparison}/{self.filename_save}.h5")
        clear_session()
    
    def setup_and_test_data(self) -> None:
        # 0: "All" (all participants' data combined)
        if self.comparison == 0:
            self.load_data()
            self.create_model()
            self.kfold_cross_validation()
            self.save_model()
        # 1: "Single" (each participant's data separate)
        elif self.comparison == 1:
            for self.participant in self.participants:
                self.set_callbacks()
                self.load_data()
                self.create_model()
                self.kfold_cross_validation()
                self.save_model()


if __name__ == '__main__':
    app = Create()
    app.load_yaml()
    app.populate_config()
    app.setup_and_test_data()
