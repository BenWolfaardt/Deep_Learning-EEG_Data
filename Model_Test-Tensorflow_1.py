"""
Created on Mon Oct 01 22:10:52 2018
Updated on Sun Aug 14 18:22:12 2022
Authors: William & Ben
"""
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pickle

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model, Sequential
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

EXPERIMENT = "Siobhan"
TRIGGERS = ['Left','Right']
PARTICIPANTS = {0: "All", 1: "Single"}
NAME = f"{EXPERIMENT}-{PARTICIPANTS}_participant(s)"
KFOLD_SPLITS=5

# Load preprocessed training/test pickles
def load_data():
    with open(f"X-Training.pickle", 'rb') as f:
        X = pickle.load(f) # shape: (1369, 63, 450, 1)
    with open(f"Y-Training.pickle", 'rb') as f:
        y = pickle.load(f)
        y = np.transpose(y)
    with open(f"X-Test.pickle", 'rb') as f:
        Xtest = pickle.load(f) # shape: (158, 63, 450, 1)
    with open(f"Y-Test.pickle", 'rb') as f:
        ytest = pickle.load(f)
    return X, y, Xtest, ytest

# Create model 
def create_model(X) -> Sequential():
    model = Sequential()
    model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:])) # input_shape: (63, 450, 1)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu')) 
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
        
    model.add(Flatten())   
    
    model.add(Dense(32))
    model.add(Activation('relu'))
    
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dropout(0.2)) 
    
    # Last dense 1ayers must have number of classes in data in the parenthesis
    # Also must be softmax
    model.add(Dense(2))
    # For a binary classification you should have a sigmoid last layer
    # See 2nd answer for more details: https://stackoverflow.com/questions/68776790/model-predict-classes-is-deprecated-what-to-use-instead
    model.add(Activation('sigmoid'))

    model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    print(model.summary())

    return model

# Train model with K-fold cross validation with early stopping
def kfold_cross_validation_earlystopping(
    model: Sequential(),
    X,
    y,
    kfold: StratifiedKFold,
) -> Sequential():
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    cvscores = []
    for index, (train_index, validation_index) in enumerate(kfold.split(X, y)):
        print("Training on fold: " + str(index+1)+"/{}".format(KFOLD_SPLITS))
        
        #Generate batches
        Xtrain, Xval = X[train_index], X[validation_index]
        ytrain, yval = y[train_index], y[validation_index]

        # Xtrain = np.asarray(Xtrain)
        # ytrain = np.asarray(ytrain)
        # Xval = np.asarray(Xval)
        # yval = np.asarray(yval)

        history = model.fit(Xtrain,ytrain, epochs=20, batch_size=10, verbose=1, validation_data=(Xval,yval), callbacks=[early_stopping], shuffle=True)
        accuracy_history = history.history['accuracy']
        val_accuracy_history = history.history['val_accuracy']
        print ("Last training accuracy: " + str(accuracy_history[-1]) 
        + ", last validation accuracy: " + str(val_accuracy_history[-1]))


    #     visualise_model_performance(history)

    #     scores = model.evaluate(xval, yval, verbose=1)
    #     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    #     cvscores.append(scores[1] * 100)
    # print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    save_model(model)
    return model, Xval, yval

# Save model to disk
def save_model(model: Sequential()) -> None:
    model.save(f"{EXPERIMENT}.h5")
    print("Saved model to disk")

# Visualise model performance
def visualise_model_performance(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    return None

# Predict model performance
def predict_model_performance(model, Xval, yval):
    scores = model.evaluate(Xval, yval, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return model

# Run model on unseen data
def run_model_on_unseen_data(model,Xtest,ytest):
    scores = model.evaluate(Xtest, ytest, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return model

# Predict model on unseen data
def predict_model_on_unseen_data(model,Xtest,ytest):
    cm = generate_confussion_matrix(model,Xtest,ytest)
    plot_confusion_matrix(cm, classes=TRIGGERS, normalize=True)

# Predict model classes on unseen data
# Rounded prediction? (old notes)
def predict_model_classes_on_unseen_data(model,Xtest,ytest):
    y_pred = model.predict_classes(Xtest)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(ytest, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(confusion_matrix(y_test, y_pred))
    return cm

# Confusion matrix
def generate_confussion_matrix(model, X_test, y_test):
    predictions = (model.predict(X_test, batch_size=10, verbose=0) > 0.9).astype("int32")
    predictions = np.argmax(predictions, axis=1)

    return confusion_matrix(predictions, y_test)

def plot_confusion_matrix(cm, normalize, classes):
    # Print the confusion matrix as text.
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalised Confusion Natrix"
        print(title)
    else:
        title = "Non-normalised Confusion Natrix"
        print(title)
    
    print(cm)

    # Dispaly the confusion matrix as a plot
    plt.matshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.tight_layout()

    # Add values to confusion matrix
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
 
    plt.show()

# Main function
def main():
    X, y, Xtest, ytest = load_data()
    model = create_model(X)
    kfold = StratifiedKFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=42)
    model, Xval, yval = kfold_cross_validation_earlystopping(model,X,y,kfold)
    save_model(model)

    print("Predict model performance")
    predict_model_performance(model, Xval, yval)
    # print("Run model on unseen data")
    # run_model_on_unseen_data(model,Xtest,ytest)
    predict_model_on_unseen_data(model, Xtest, ytest)


    # cm = None
    # cm = predict_model_classes_on_unseen_data(model,Xtest,ytest)
    # plot_confusion_matrix(cm, classes=CATEGORIES, normalize=False, title='Confusion matrix, without normalization')
    # plot_confusion_matrix(cm, classes=CATEGORIES, normalize=True, title='Confusion matrix')

if __name__ == '__main__':
    main()
    print("Done")

# ----------------------------------------------------------------------------------------------------------------------
# Miscelaneous fiddling
# ----------------------------------------------------------------------------------------------------------------------

# Load model from disk
def load_model():
    model = load_model(EXPERIMENT)
    print("Loaded model from disk")
    return model
