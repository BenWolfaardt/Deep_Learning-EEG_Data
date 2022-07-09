# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 22:10:52 2018

@author: Ben
"""
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard , EarlyStopping
import itertools
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix
#import itertools
#import matplotlib.pyplot as plt

NAME = "CNN train 20 2 Architecture"
# Loading training data from preprocessing
# Change these numbers
pickle_in = open("X_train_20_2.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open("y_train_20_2.pickle","rb")
y = pickle.load(pickle_in)

# Change these numbers
pickle_in = open("X_test_20_2.pickle","rb")
X_test = pickle.load(pickle_in)
pickle_in = open("y_test_20_2.pickle","rb")
y_test = pickle.load(pickle_in)

CATEGORIES = ['Lavendar','Full Screen','Word']

# KFold pick your folds here
kfold_splits=3
skf = StratifiedKFold(n_splits=kfold_splits, shuffle=True)
y = np.transpose(y)

def create_model():
    # Give it unique name for tensorboard and also save
   
    model = Sequential()
         
    model.add(Conv2D(512, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu')) 
    
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(48, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
        
    model.add(Flatten())   
    
    model.add(Dense(36))
    model.add(Activation('relu'))
    
    model.add(Dense(18))
    model.add(Activation('relu'))
    model.add(Dropout(0.2)) 
    
    # Last dense 1ayers must have number of classes in data in the parenthesis
    # Also must be softmax
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model

def train_model(model,xtrain,ytrain,xval,yval):
                
    # Preventing overfitting through 'earlystopping' it will monitor val_loss and stop
    # computing when val_loss goes up even though there are more epochs
      earlystopping = EarlyStopping(monitor= 'val_loss',
                                    min_delta = 0, 
                                    patience= 2, 
                                    verbose = 0, 
                                    mode ='auto'
                                    )
    
      accuracy= model.fit(xtrain,ytrain,
              epochs = 8, 
              validation_data= (xval,yval),
              batch_size=10,
              shuffle=True
              )    
      return accuracy

Total_accuracy = []
for index, (train_indices, val_indices) in enumerate(skf.split(X,y)):
    print("Training on fold: " + str(index+1)+"/{}".format(kfold_splits))
    
    #Generate batches
    xtrain, xval = X[train_indices], X[val_indices]
    ytrain, yval = y[train_indices], y[val_indices]

    # Clear model, and create it
    model = None
    model = create_model()

    # Debug message I guess
    print ("Training new iteration on " + str(xtrain.shape[0]) + " training samples, " 
    + str(xval.shape[0]) + " validation samples, this may be a while...")
    
    
    history = train_model(model, xtrain, ytrain, xval, yval)
    accuracy_history = history.history['acc']
    val_accuracy_history = history.history['val_acc']
    print ("Last training accuracy: " + str(accuracy_history[-1]) 
    + ", last validation accuracy: " + str(val_accuracy_history[-1]))
    
    
    Total_accuracy.append(val_accuracy_history[-1])

accuracy_Array = np.asarray(Total_accuracy)
print("%.2f%% (+/- %.2f%%)" % (np.mean(accuracy_Array*100), np.std(accuracy_Array*100)))
print(accuracy_Array)

# model = model_load(NAME)
model.save(NAME)

# Loading the trained model for testing 
#model = load_model("NAME")

# Letting the above trained model predict on unseen data 
prediction = model.predict(X_test, batch_size=10, verbose=0)
# Visualizing these predictions
for i in prediction:
    print(i)

# Rounding the prediction to the classes
rounded_prediction = model.predict_classes(X_test, batch_size=10, verbose=0)
# Visualizing the best model
for i in rounded_prediction:
    print(i)

# Confusion matrix to visualize how the model was performing SKLEARN
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float32') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, rounded_prediction)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=CATEGORIES,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=CATEGORIES, normalize=True,
                      title='Normalized confusion matrix')
plt.show()
model.summary()