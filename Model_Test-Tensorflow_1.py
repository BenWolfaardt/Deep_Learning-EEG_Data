"""
Created on 01/10/2018
Updated on 07/09/2022
Authors: William & Ben
"""
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pickle

from keras.models import load_model, Sequential
from nptyping import NDArray, Shape, Float64
from numpy.core.shape_base import block
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Tuple

EXPERIMENT = "Siobhan"
TRIGGERS = ['Left','Right']
PARTICIPANTS = {0: "All", 1: "Single"}
NAME = f"{EXPERIMENT}-{PARTICIPANTS}_participant(s)"
KFOLD_SPLITS=5

class Test:
    def __init__(self) -> None:
        pass
    
    # Load test pickles (unseen)
    # TODO is this a dynamic shape dependant on pickle?
    def load_data(self) -> Tuple[NDArray[Shape['24,128,257,1'], Float64], NDArray[Shape['1'], Float64]]:
        with open(f"X-Test.pickle", 'rb') as f:
            Xtest = pickle.load(f) # shape: (24, 128, 257, 1) - Participant: 2
        with open(f"Y-Test.pickle", 'rb') as f:
            ytest = pickle.load(f) # len: 24
        print("Pickles loaded")
        return Xtest, ytest

    # Load model
    def load_model(self, name) -> Sequential:
        model = load_model(name)
        print("Model Loaded")
        return model

    # Predict model performance
    def predict_model_performance(self, model, Xval, yval) -> None:
        scores = model.evaluate(Xval, yval, verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # Run model on unseen data
    def run_model_on_unseen_data(self, model, Xtest, ytest)  -> None:
        scores = model.evaluate(Xtest, ytest, verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    def predict_model_on_unseen_data(self, model, Xtest, ytest, classes) -> None:
        cm = self.generate_confussion_matrix(model, Xtest, ytest, classes)
        self.plot_confusion_matrix(cm, triggers=TRIGGERS, normalize=True)

    def generate_confussion_matrix(self, model, X_test, y_test, classes) -> NDArray[Shape['2,2'], Float64]:
        if classes:
            predictions = (model.predict_classes(X_test, batch_size=10, verbose=0) > 0.9).astype("int32")
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = (model.predict(X_test, batch_size=10, verbose=0) > 0.9).astype("int32")
            predictions = np.argmax(predictions, axis=1)

        return confusion_matrix(predictions, y_test)

    def plot_confusion_matrix(self, cm, normalize, triggers) -> None:
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = "Normalised Confusion Natrix"
            print(title)
        else:
            title = "Non-normalised Confusion Natrix"
            print(title)

        plt.figure(figsize=(6.2,5))
        plt.matshow(cm, interpolation='nearest', cmap=plt.cm.Blues, fignum=plt.gcf().number)
        plt.title(title, loc='center', y=1.05)
        plt.colorbar()
        tick_marks = np.arange(len(triggers))
        plt.xticks(tick_marks, triggers)
        plt.yticks(tick_marks, triggers, rotation=90)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tick_params(top=False, labelleft=True, labelbottom=True, labeltop=False)

        # Add values to confusion matrix
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
        plt.show()

    def setup_and_test_data(self) -> None:
        Xtest, ytest = self.load_data()
        model = self.load_model(f"{EXPERIMENT}.h5")
        self.predict_model_on_unseen_data(model, Xtest, ytest, classes=False)
        # TODO test the below on a TensorFlow 1 setup
        # self.predict_model_on_unseen_data(self, model, Xtest, ytest, classes=True)

# Main function
if __name__ == '__main__':
    app = Test()
    app.setup_and_test_data()
