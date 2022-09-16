"""
Created on 01/10/2018
Updated on 07/09/2022
Authors: William & Ben
"""
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pickle

from envyaml import EnvYAML
from keras.models import load_model, Sequential
from nptyping import NDArray, Shape, Float64
from numpy.core.shape_base import block
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Test:
    def __init__(self) -> None:
        # config
        self.os: str = None
        self.config: EnvYAML = None
        self.experiment: str = None
        # paths
        self.pickles: str = None
        self.models: str = None
        self.confusion_matrixes: str = None
        # experiment details
        self.name: str = None
        self.participants: list[int] = []
        self.triggers: list[str] = []
        self.comparison: int = None

    def load_yaml(self) -> None:
        self.config = EnvYAML("./setup/config.yaml", strict=False)

    def populate_config(self):
        # TODO parse in as cli command from make file
        self.os = "mac_m1"
        self.experiment = "libet"

        # TODO add name into folder save or something
        self.pickles = self.config[f"os.{self.os}.io_paths.pickle_files"]
        self.models = self.config[f"os.{self.os}.io_paths.model_files"]
        self.confusion_matrixes = self.config[f"os.{self.os}.io_paths.confusion_matrices"]
        self.name = self.config[f"experiment.details.{self.experiment}.name"]
        self.participants = self.config[f"experiment.details.{self.experiment}.participants"]
        self.triggers = self.config[f"experiment.details.{self.experiment}.triggers"]
        self.comparison = self.config[f"model_parameters.comparison"]
    
    
    # Load test pickles (unseen)
    # TODO is this a dynamic shape dependant on pickle?
    # def load_data(self) -> tuple[NDArray[Shape['24,128,257,1'], Float64], NDArray[Shape['1'], Float64]]:
    def load_data(self) -> None:
        # TODO assign type hints
        self.Xtest: NDArray = None
        self.ytest: NDArray = None
        
        with open(f"{self.pickles}/X-{str(self.participant)}-Test.pickle", 'rb') as f:
            self.Xtest = pickle.load(f) # shape: (24, 128, 257, 1) - Participant: 2
            self.Xtest = np.asarray(self.Xtest)
        with open(f"{self.pickles}/y-{str(self.participant)}-Test.pickle", 'rb') as f:
            self.ytest = pickle.load(f) # len: 24
            self.ytest = np.asarray(self.ytest)

    # Load model
    def load_model(self) -> None:
        self.filename = f"{self.experiment}_{self.comparison}-{self.participant}"
        self.model = load_model(f"{self.models}/{self.filename}.h5")
        print("Model Loaded")

    # TODO tidy up
    # Run model on unseen data
    def run_model_on_unseen_data(self) -> None:
        scores = self.model.evaluate(self.Xtest, self.ytest, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))

    def predict_model_on_unseen_data(self) -> None:
        self.normalize = True
        
        self.generate_confussion_matrix()
        self.plot_confusion_matrix()

    def generate_confussion_matrix(self) -> NDArray[Shape['2,2'], Float64]:
        # TODO make dynamic for other experiment board size as well =
        self.confusion_matrix: NDArray[Shape['2,2'], Float64] = None
        
        if self.classes:
            predictions = (self.model.predict_classes(self.Xtest, batch_size=10, verbose=0) > 0.9).astype("int32")
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = (self.model.predict(self.Xtest, batch_size=10, verbose=0) > 0.9).astype("int32")
            predictions = np.argmax(predictions, axis=1)

        self.confusion_matrix = confusion_matrix(predictions, self.ytest)

    def plot_confusion_matrix(self) -> None:
        if self.normalize:
            # Not sure about the below cm replacement
            self.confusion_matrix = self.confusion_matrix.astype('float') / self.confusion_matrix.sum(axis=1)[:, np.newaxis]
            title = "Normalised Confusion Natrix"
            print(title)
        else:
            title = "Non-normalised Confusion Natrix"
            print(title)

        plt.figure(figsize=(6.2,5))
        plt.matshow(self.confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues, fignum=plt.gcf().number)
        plt.title(title, loc='center', y=1.05)
        plt.colorbar()
        tick_marks = np.arange(len(self.triggers))
        plt.xticks(tick_marks, self.triggers)
        plt.yticks(tick_marks, self.triggers, rotation=90)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tick_params(top=False, labelleft=True, labelbottom=True, labeltop=False)

        # Add values to confusion matrix
        fmt = '.2f' if self.normalize else 'd'
        thresh = self.confusion_matrix.max() / 2.
        for i, j in itertools.product(range(self.confusion_matrix.shape[0]), range(self.confusion_matrix.shape[1])):
            plt.text(j, i, format(self.confusion_matrix[i, j], fmt), horizontalalignment="center", color="white" if self.confusion_matrix[i, j] > thresh else "black")
    
        plt.show(block=True)

    def setup_and_test_data(self) -> None:
        for self.participant in self.participants:
            self.load_data()
            self.load_model()
            self.run_model_on_unseen_data()
            self.classes = False
            self.predict_model_on_unseen_data()
            # TODO test the below on a TensorFlow 1 setup
            # self.classes = True
            # self.predict_model_on_unseen_data()

# Main function
if __name__ == '__main__':
    app = Test()
    app.load_yaml()
    app.populate_config()
    app.setup_and_test_data()
