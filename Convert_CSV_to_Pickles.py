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
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

DIRECTORY_CSV_DATA_ROOT = 'E:\\Skripsie\\Data\\New\\2-CSV'
DIRECTORY_PICKLE_DATA_OUTPUT = "E:\\Skripsie\\Data\\New\\3-Pickles"
PERCENTAGE_TRAINING_AND_VALIDATION = float(90)

# Participants = ['A', 'B', 'C', 'D']
Participants = ['C']
# Grouped_triggers =  [
#                         [1,2,7],    # Colours - Purple
#                         [4,3,5],    # Colours - Red
#                         [14,11,9],  # Numbers - Seven
#                         [15,13,10], # Numbers - Two
#                         [17,18,16], # Objects - Ball
#                         [8,6,12]    # Objects - Pen
#                     ]
Grouped_triggers =  [[8], [9]]

# TODO check the standardScalar thing I deleted
sc = StandardScaler()

# Split the CSV data into [training & validation (grouped)] and testing data
def create_data_split(Participants):
    for grouped_triggers in Grouped_triggers:
        for trigger in grouped_triggers:
            for participant in Participants:
                try:
                    # Set LEADING_ZERO prefix based on trigger naming convention 
                    if trigger <= 9:
                        LEADING_ZERO = "0"
                    else:
                        LEADING_ZERO = ""

                    # Directory manipulation
                    directory_trigger = f"{DIRECTORY_CSV_DATA_ROOT}\{participant}\T{LEADING_ZERO}{trigger}"
                    # Random split of data into [training & validation (grouped)] and testing data
                    list_csv_files = fnmatch.filter(os.listdir(directory_trigger), '*.csv')
                    # list_csv_files = [1,2,3,4,5]
                    k = len(list_csv_files) * PERCENTAGE_TRAINING_AND_VALIDATION // 100
                    k = round(k)
                    k = int(k)
                    indicies = random.sample(range(len(list_csv_files)), k)
                    indicies.sort()
                    training_and_validation_list = [list_csv_files[i] for i in indicies]
                    # print(f"{training_and_validation_list}")

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
                    # print(f"Answer: {testing_list}")
                except Exception as e:
                    print(f"Error: {e}")

                try: 
                    # Training Data
                    training_data = []
                    training_data = create_data(directory_trigger, training_and_validation_list, "Training", grouped_triggers, trigger)

                    x_training = []
                    y_training = []
                    create_pickle(training_data, x_training, y_training, "Training", participant, category)

                    # Test Data
                    testing_data = []
                    testing_data = create_data(directory_trigger, testing_list, "Test", grouped_triggers, trigger)

                    x_test = []
                    y_test = []
                    create_pickle(testing_data, x_test, y_test, "Test", participant, category)

                except Exception as e:
                    print(f"Error: {e}")    

    return training_and_validation_list, testing_list

def exists_in_list(testing_list, x, i, trigger, LEADING_ZERO):
    while x != i:
        if x == i:
            pass
        else:
            testing_list.append(f"T{LEADING_ZERO}{trigger}-{(i + 1):03d}.csv")
            i += 1
    i += 1
    return i

# Create and populate the Training/Testing data
def create_data(path, data, data_type, grouped_triggers, trigger):
    trigger_instance = grouped_triggers.index(trigger)

    for count, epoch in enumerate(tqdm(data)):
        try:
            if count == 49:
                pass
            if count < len(data) - 1:
                epoch_array = np.genfromtxt(f"{path}\{epoch}", delimiter=',')
                new_array = sc.fit_transform(epoch_array)
                # Instead of having the full 0:750 ms now we are reshaping it to 100:550 ms
                reshape_array = new_array[:,100:550]
                data.append([reshape_array, trigger_instance])
            else:
                break
        except Exception as e:
            print(f"Error: {e}")
            print("")
            print(f"An error occurded on {data_type} trigger: {trigger}.")
            print("Remeber to check back and correct this!")
            
    print(f"{data_type} data conists of {len(training_data)}")
    return data

# Create and populate the pickles from the Training/Testing data
def create_pickle(data, x, y, data_type, participant, category):
    random.shuffle(data)

    for features, label in data:
        x.append(features)
        y.append(label)
    # Parenthesis depend on the input data -1 being batch size, channels, datasamples, idk
    x = np.array(x).reshape(-1,63,450,1)

    pickle_out = open(f"{participant}-{category}-X-{data_type}.pickle","wb")
    pickle.dump(X_test, pickle_out)
    pickle_out.close()

    pickle_out = open(f"{participant}-{category}-Y-{data_type}.pickle","wb")
    pickle.dump(y_test, pickle_out)
    pickle_out.close()

# Generate the list of [training & validation (grouped)] and testing data
# training_and_validation_list, testing_list = create_data_split()
create_data_split(Participants)


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