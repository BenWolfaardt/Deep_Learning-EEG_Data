# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 01:42:47 2018

@author: Ben
"""

import numpy as np
import os
from tqdm import tqdm
import random
import pickle
from sklearn.preprocessing import StandardScaler

DATADIR = 'D:\\AAA DATA\\Tristancsv\\Pen'
CATEGORIES = ['T6','T8','T12']
TESTINGCAT = ['T6Test','T8Test','T12Test']

# If you manualy divided the data in different folders to test it after training (take 10 files each class)
sc = StandardScaler()

# Creating training data just reading it and putting it in an array
training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for epoch in tqdm(os.listdir(path)):
            try:
                epoch_array = np.genfromtxt(os.path.join(path,epoch),delimiter=',')
                new_array = sc.fit_transform(epoch_array)
                reshape_array = new_array[:,100:550]
                training_data.append([reshape_array, class_num])
            except Exception as e:
                pass 
create_training_data()
print(len(training_data))

# Testing data to validate the model not sure if this is going to be the final way to test it
# but we'll see
testing_data = []
def create_testing_data():    
    for category in TESTINGCAT:
        path = os.path.join(DATADIR,category)
        class_num_test = TESTINGCAT.index(category)
        for epoch in tqdm(os.listdir(path)):
            try:
                epoch_array = np.genfromtxt(os.path.join(path,epoch),delimiter=',')
                new_array = sc.transform(epoch_array)
                reshape_array = new_array[:,100:550]
                testing_data.append([reshape_array, class_num_test])
            except Exception as e:
                pass
             
create_testing_data()
print(len(testing_data))
#
X_test = []
y_test = []
for features, label in testing_data:
    X_test.append(features)
    y_test.append(label) 
# Parenthesis depend on the input data -1 being batch size, channels, datasamples, idk
X_test = np.array(X_test).reshape(-1,63,450,1)

# Second we do the training data, for training we randomly shuffle it
# This is also done when we fit the model but shuffling twice is even better then once I guess
random.shuffle(training_data)

X = []
y = []
for features,label in training_data:
        X.append(features)
        y.append(label)
X = np.array(X).reshape(-1,63,450,1)

# Saving the preprocessed data to feed in to algorithm 
# This way you dont have to load all the csv files every time you change your algorithm
# Training data
pickle_out = open("X_train_20_6.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y_train_20_6.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# Testing data is saved in a different pickle, this pickle will be loaded
# when you start fitting the model to test it and generate a confusion matrix (I think)
pickle_out = open("X_test_20_6.pickle","wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test_20_6.pickle","wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()
