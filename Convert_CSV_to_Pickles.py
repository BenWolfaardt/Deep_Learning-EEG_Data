"""
Author: Ben Wolfaardt

DESCRIPTION: 

# If you manualy divided the data in different folders to test it after training (take 10 files each class)

# Saving the preprocessed data to feed in to algorithm 
# This way you dont have to load all the csv files every time you change your algorithm

# Testing data is saved in a different pickle, this pickle will be loaded
# when you start fitting the model to test it and generate a confusion matrix (I think)
"""

import os
import random
import fnmatch
import pickle
from textwrap import indent
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

DIRECTORY_CSV_DATA_ROOT = 'E:\\Skripsie\\Data\\New\\2-CSV'
DIRECTORY_PICKLE_DATA_OUTPUT = "E:\\Skripsie\\Data\\New\\3-Pickles"
PERCENTAGE_TRAINING_AND_VALIDATION = float(1)

Participants = ['A', 'B', 'C', 'D']
# Participant D, Trigger 11's data is incorrect and only has 59 channels instead of 63.
# Grouped_triggers without the "Seven" experiment
Grouped_triggers =  [
                        [1,2,7],    # Colours - Purple
                        [3,4,5],    # Colours - Red
                        # [9,11,14],  # Numbers - Seven
                        [10,13,15], # Numbers - Two
                        [16,17,18], # Objects - Ball
                        [6,8,12]    # Objects - Pen
                    ]
# Experiment_number = {
#                         0: "Purple", 
#                         1: "Red", 
#                         2: "Seven", 
#                         3: "Two", 
#                         4: "Ball", 
#                         5: "Pen"
#                     }
# Experiment_number without the "Seven" experiment
Experiment_number = {
                        0: "Purple", 
                        1: "Red", 
                        2: "Two", 
                        3: "Ball", 
                        4: "Pen", 
                    }
scaler = StandardScaler()

# Split the CSV data into [training & validation (grouped)] and 
# testing epochs based on the PERCENTAGE_TRAINING_AND_VALIDATION
def generate_split_data_type_epoch_list(path, trigger, LEADING_ZERO):
    try:
        # Random split of data into [training & validation (grouped)] and testing data
        list_csv_files = fnmatch.filter(os.listdir(path), '*.csv')
        k = len(list_csv_files) * PERCENTAGE_TRAINING_AND_VALIDATION // 100
        k = round(k)
        k = int(k)
        indicies = random.sample(range(len(list_csv_files)), k)
        indicies.sort()
        
        training_and_validation_list = [list_csv_files[i] for i in indicies]
        testing_list = []

        i = 0
        for csv_file in range(len(list_csv_files)):
            if csv_file < len(indicies):
                i = exists_in_list(testing_list, int(indicies[csv_file]), i, trigger, LEADING_ZERO)
            else:
                if i < len(indicies) - 1:
                    testing_list.append(f"T{LEADING_ZERO}{trigger}-{(i + 1):03d}.csv")
                    i += 1
                else:
                    break
        
        return training_and_validation_list, testing_list
        
    except Exception as e:
        print(f"Error: {e}")

# TODO description of this function
# TODO rename funciton to something more descriptive
def exists_in_list(testing_list, x, i, trigger, LEADING_ZERO):
    while x != i:
        if x == i:
            pass
        else:
            testing_list.append(f"T{LEADING_ZERO}{trigger}-{(i + 1):03d}.csv")
            i += 1
    i += 1
    return i

# Create and populate the Training/Test data
def populate_data_type_epoch_lists(selected_epochs, data, data_type, path, participant, grouped_triggers, trigger):
    print(f"Generating {data_type} data (consisting of {len(selected_epochs)} epochs) for participant: {participant}")
    
    trigger_instance = grouped_triggers.index(trigger)

    try:
        for epoch in tqdm(selected_epochs):
            epoch_array = np.genfromtxt(f"{path}\{epoch}", delimiter=',')
            new_array = scaler.fit_transform(epoch_array, None)
            # Instead of having the full 0:750 ms now we are reshaping it to 100:550 ms
            reshape_array = new_array[:,100:550]
            data.append([reshape_array, trigger_instance])
        
        return data

    except Exception as e:
        print(f"An error occurded whilst creating the {data_type} data for trigger: {trigger} epoch {epoch}.")
        # TODO raise flag so that this is printed at the end. 
        # print("Remeber to check back and correct this!")

# Create and populate the pickles from the Training/Testing data
def create_data_type_pickles(data, data_type, experiment):
    print(f"Generating Pickles for {data_type} data")
    
    x = []
    y = []

    random.shuffle(data)

    for features, label in data:
        x.append(features)
        y.append(label)
    # Parenthesis depend on the input data -1 being batch size, channels, datasamples, idk
    x = np.array(x).reshape(-1,63,450,1)

    pickle_out = open(f"{experiment}-X-{data_type}.pickle","wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()

    pickle_out = open(f"{experiment}-Y-{data_type}.pickle","wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

for grouped_triggers in Grouped_triggers:
    training_data = []
    testing_data = []

    experiment_name = Experiment_number[Grouped_triggers.index(grouped_triggers)]
    print(f"Experiment: {experiment_name}\n")

    for trigger in grouped_triggers:
        print(f"Trigger: {trigger}\n")

        for participant in Participants:
            # Set LEADING_ZERO prefix based on trigger naming convention 
            if trigger <= 9:
                LEADING_ZERO = "0"
            else:
                LEADING_ZERO = ""
            # Directory manipulation
            path = f"{DIRECTORY_CSV_DATA_ROOT}\{participant}\T{LEADING_ZERO}{trigger}"

            # Create data split for training and validation data
            training_and_validation_epochs = []
            testing_epochs = []
            training_and_validation_epochs , testing_epochs = generate_split_data_type_epoch_list(path, trigger, LEADING_ZERO)

            training_data = populate_data_type_epoch_lists(training_and_validation_epochs, training_data, "Training", path, participant, grouped_triggers, trigger)
            # testing_data = populate_data_type_epoch_lists(testing_epochs, testing_data, "Test", path, participant, grouped_triggers, trigger)
            print()

    create_data_type_pickles(training_data, "Training", experiment_name)
    create_data_type_pickles(testing_data, "Test", experiment_name)
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------\n")

# Testing a recursive implementation of training/test split
# def exists_in_list(testing_list, x, i):
#     print(f"testing_list: {testing_list}, x: {x}, i:{i}")
#     if x == i:
#         i += 1
#     else:
#         testing_list.append(i)
#         i += 1
#         exists_in_list(testing_list, x, i)
#     return i
