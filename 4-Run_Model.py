"""
Created on 01/10/2018
Updated on 07/09/2022
Authors: William Geuns & Ben Wolfaardt
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from envyaml import EnvYAML
from keras.models import load_model
from nptyping import NDArray, Shape, Float64
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Optional, Any


class Test:
    def __init__(self) -> None:
        # config
        self.os: str = ""
        self.config: EnvYAML = None
        self.experiment: str = ""
        # paths
        self.pickles: str = ""
        self.models: str = ""
        self.confusion_matrices: str = ""
        # experiment details
        self.name: str = ""
        self.participants: list[int] = []
        self.participant: int = 0
        self.triggers: list[str] = []
        self.version: str = ""
        self.comparison: int = 0
        # model
        self.model: Optional[Any] = None
        self.Xtest: NDArray = None
        self.ytest: NDArray = None
        self.normalize: bool = False
        self.confusion_matrix: Any = None
        self.classes: bool = False
        # filenames
        self.filename_load: str = ""
        self.filename_save: str = ""

    def load_yaml(self) -> None:
        self.config = EnvYAML("./setup/config.yaml", strict=False)

    def populate_config(self):
        # TODO
        #   Parse in as arg using argparse
        self.os = "mac_m1"
        self.experiment = "libet"

        self.pickles = self.config[f"os.{self.os}.io_paths.pickle_files"]
        self.models = self.config[f"os.{self.os}.io_paths.model_files"]
        self.confusion_matrices = self.config[f"os.{self.os}.io_paths.confusion_matrices"]
        self.name = self.config[f"experiment.details.{self.experiment}.name"]
        self.participants = self.config[f"experiment.details.{self.experiment}.participants"]
        self.triggers = self.config[f"experiment.details.{self.experiment}.triggers"]
        self.version = self.config[f"experiment.details.{self.experiment}.version"]
        self.comparison = self.config[f"model_parameters.comparison"]

    # Load test pickles (unseen)
    # TODO:
    #   The type hint is dynamic for the method and is dependant on the pickle shape
    #   eg. def load_data(self) -> tuple[NDArray[Shape['24,128,257,1'], Float64], NDArray[Shape['1'], Float64]]:
    #   Determine if there is a dynamic way to set this shape
    def load_data(self) -> None:
        # TODO
        #   Don't save self.comparison as number but rather as value of dict {0: "All", 1: "Single"}
        # 0: "All" (all participants' data combined)
        if self.comparison == 0:
            self.filename_load = ""
        # 1: "Single" (each participant's data separate)
        elif self.comparison == 1:
            self.filename_load = f"{self.participant}-"

        with open(f"{self.pickles}/{self.version}/{self.comparison}/X-{self.filename_load}Testing.pickle", 'rb') as f:
        # with open(f"{self.pickles}/{self.version}/X-{filename}Testing.pickle", 'rb') as f:
            print(f"X: {self.pickles}/{self.version}/X-{self.filename_load}Testing.pickle")
            self.Xtest = pickle.load(f)  # shape: (1369, 63, 450, 1)
            self.Xtest = np.asarray(self.Xtest)
        with open(f"{self.pickles}/{self.version}/{self.comparison}/y-{self.filename_load}Testing.pickle", 'rb') as f:
        # with open(f"{self.pickles}/{self.version}/y-{filename}Testing.pickle", 'rb') as f:
            print(f"y: {self.pickles}/{self.version}/y-{self.filename_load}Testing.pickle")
            self.ytest = pickle.load(f)  # shape: (
            self.ytest = np.transpose(self.ytest)

    def load_model(self) -> None:
        # TODO
        #   Don't save self.comparison as number but rather as value of dict {0: "All", 1: "Single"}
        # 0: "All" (all participants' data combined)
        if self.comparison == 0:
            # filename = "inter_participant"
            self.filename_load = "libet"
        # 1: "Single" (each participant's data separate)
        elif self.comparison == 1:
            self.filename_load = f"{self.participant}"

        # filename = "_1-1"

        # self.model = load_model(f"{self.models}/{self.version}/{self.comparison}/{self.experiment}{filename}.h5")
        self.model = load_model(f"{self.models}/{self.version}/{self.comparison}/{self.filename_load}.h5")
        print(f"Model: {self.models}/{self.version}/{self.comparison}/{self.comparison}/{self.filename_load}.h5")
        print("Model Loaded")

    def run_model_on_unseen_data(self) -> None:
        scores = self.model.evaluate(self.Xtest, self.ytest, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

    def predict_model_on_unseen_data(self) -> None:
        self.normalize = False

        self.generate_confusion_matrix()
        self.plot_confusion_matrix()

    def generate_confusion_matrix(self) -> NDArray[Shape[Any], Float64]:
        # TODO
        #   Make shape dynamic such that the other experiment board size will also work
        #       def generate_confusion_matrix(self) -> NDArray[Shape['2,2'], Float64]:
        #   and in
        #       self.confusion_matrix: NDArray[Shape['2,2'], Float64] = None

        self.confusion_matrix: NDArray[Shape[Any], Float64] = None

        if self.classes:
            predictions = (self.model.predict_classes(self.Xtest, batch_size=10, verbose=0) > 0.9).astype("int32")
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = (self.model.predict(self.Xtest, batch_size=10, verbose=0) > 0.9).astype("int32")
            predictions = np.argmax(predictions, axis=1)

        self.confusion_matrix = confusion_matrix(predictions, self.ytest)

    def plot_confusion_matrix(self) -> None:
        if self.normalize:
            # TODO
            #   Investigate the below cm replacement
            self.confusion_matrix = self.confusion_matrix.astype('float') / self.confusion_matrix.sum(axis=1)[:, np.newaxis]
            title = "Normalised Confusion Matrix"
            print(title)
        else:
            title = "Non-normalised Confusion Matrix"
            print(title)

        plt.figure(figsize=(6.2, 5))
        plt.matshow(self.confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues, fignum=plt.gcf().number)
        plt.title(title, loc='center', y=1.05)
        plt.colorbar()
        tick_marks = np.arange(len(self.triggers))
        plt.xticks(tick_marks, self.triggers)
        plt.yticks(tick_marks, self.triggers, rotation=90)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tick_params(top=False, labelleft=True, labelbottom=True, labeltop=False)

        # Add values to confusion matrices
        fmt = '.2f' if self.normalize else 'd'
        thresh = self.confusion_matrix.max() / 2.
        for i, j in itertools.product(range(self.confusion_matrix.shape[0]), range(self.confusion_matrix.shape[1])):
            plt.text(
                j,
                i,
                format(self.confusion_matrix[i, j], fmt),
                horizontalalignment="center",
                color="white" if self.confusion_matrix[i, j] > thresh else "black"
            )

        try:
            os.makedirs(f"{self.confusion_matrices}/{self.version}")
        except Exception as e:
            print(f"{self.confusion_matrices}/{self.version} already exists: {e}")

        if self.comparison == 0:
            self.filename_save = "inter_participant"
        elif self.comparison == 1:
            self.filename_save = f"{self.participant}"

        try:
            os.makedirs(f"{self.confusion_matrices}/{self.version}/{self.comparison}")
        except Exception as e:
            print(f"{self.confusion_matrices}/{self.version}/{self.comparison}already exists: {e}")

        plt.savefig(f"{self.confusion_matrices}/{self.version}/{self.comparison}/{self.filename_save}")
        # plt.show(block=True)

    def setup_and_test_data(self) -> None:
        # 0: "All" (all participants' data combined)
        # TODO
        #   Try adapting code such that duplicate parts are removed in the case of self.participant = 1
        if self.comparison == 0:
            self.load_data()
            self.load_model()
            self.run_model_on_unseen_data()
            self.classes = False
            self.predict_model_on_unseen_data()
            # TODO
            #   Test the below on a TensorFlow 1 setup
            #       self.classes = True
            #       self.predict_model_on_unseen_data()
        # 1: "Single" (each participant's data separate)
        elif self.comparison == 1:
            for self.participant in self.participants:
                self.load_data()
                self.load_model()
                self.run_model_on_unseen_data()
                self.classes = False
                self.predict_model_on_unseen_data()
                # TODO
                #   Test the below on a TensorFlow 1 setup
                #       self.classes = True
                #       self.predict_model_on_unseen_data()


if __name__ == '__main__':
    app = Test()
    app.load_yaml()
    app.populate_config()
    app.setup_and_test_data()
