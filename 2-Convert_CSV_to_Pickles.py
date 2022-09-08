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
import random
from textwrap import indent
from typing import List
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

DIRECTORY_CSV_DATA_ROOT = '/Users/james.wolfaardt/code/__ben/Code/Siobhan_Data'
# TODO output to correct directory and not project root.
DIRECTORY_PICKLE_DATA_OUTPUT = "/Users/james.wolfaardt/code/__ben/Code/Deep_Learning-EEG_Data/outputs/pickles"
# TODO fiddle with ratio
PERCENTAGE_TRAINING_AND_VALIDATION = float(80)

# 1, 3 don't exist
# Participants = [2]
Participants = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
Triggers =  ['L', 'R']

scaler = StandardScaler()

class Pickles:
    # def __init__(self, participants, triggers) -> None:
    def __init__(self) -> None:
        # TODO figure out correct parsing
        # self.participants: list[int] = Participants,
        # self.triggers: list[str] = Triggers,
        self.training_data: list[str] = None
        self.testing_data: list[str] = None
        self.path: str = None

    # Split the CSV data into [training & validation (grouped)] and 
    # testing epochs based on the PERCENTAGE_TRAINING_AND_VALIDATION
    # TODO seems like the random generator's rounding makes us loose 1 or 2 files, to investigate
    def generate_split_data_type_epoch_list(self, trigger, participant):
        try:
            # Random split of data into [training & validation (grouped)] and testing data
            list_csv_files = fnmatch.filter(os.listdir(self.path), '*.csv')
            k = len(list_csv_files) * PERCENTAGE_TRAINING_AND_VALIDATION // 100
            k = round(k)
            k = int(k)
            indicies = random.sample(range(len(list_csv_files)), k)
            indicies.sort()
            
            training_and_validation_list = [list_csv_files[i] for i in indicies]
            testing_list = []

            i = 1
            for csv_file in range(len(list_csv_files)):
                if csv_file < len(indicies):
                    i = self.exists_in_list(testing_list, int(indicies[csv_file] + 1), i, trigger, participant)
                else:
                    if i < len(indicies): # no minus 1 as counting starts at 1 and not at 0
                        testing_list.append(f"P{participant}{trigger}{i}.csv")
                        i += 1
                    else:
                        break
            
            return training_and_validation_list, testing_list
            
        except Exception as e:
            print(f"Error: {e}")

    # TODO description of this function
    # TODO rename funciton to something more descriptive
    def exists_in_list(self, testing_list, x, i, trigger, participant):
        while x != i:
            if x == i:
                pass
            else:
                testing_list.append(f"P{participant}{trigger}{i}.csv")
                i += 1
        i += 1
        return i

    # Create and populate the Training/Test data
    def populate_data_type_epoch_lists(self, selected_epochs, data, data_type, trigger):
        print(f"Generating {data_type} data (consisting of {len(selected_epochs)} epochs) for the {trigger} trigger.")
        
        trigger_instance = Triggers.index(trigger)

        try:
            for epoch in tqdm(selected_epochs):
                epoch_array = np.genfromtxt(f"{self.path}/{epoch}", delimiter=',')
                new_array = scaler.fit_transform(epoch_array, None)
                # TODO check the reshape logic for Siobhan
                # Instead of having the full 0:750 ms now we are reshaping it to 100:550 ms
                # reshape_array = new_array[:,100:550]
                data.append([new_array, trigger_instance]) # where the label is added to the feature
            
            return data

        except Exception as e:
            print(f"An error occurded whilst creating the {data_type} data for trigger: {trigger} epoch {epoch}.")
            quit()
            # TODO raise flag so that this is printed at the end. 
            # print("Remeber to check back and correct this!")

    # Create and populate the pickles from the Training/Testing data
    def create_data_type_pickles(self, data, data_type, participant):
        print(f"Generating Pickles for {data_type} data")
        
        x = []
        y = []

        random.shuffle(data)

        for features, label in data:
            x.append(features)
            y.append(label)
        # Parenthesis depend on the input data -1 being batch size, channels, datasamples, idk
        x = np.array(x).reshape(-1,128,257,1)

        pickle_out = open(f"{DIRECTORY_PICKLE_DATA_OUTPUT}/X-{participant}-{data_type}.pickle","wb")
        pickle.dump(x, pickle_out)
        pickle_out.close()

        pickle_out = open(f"{DIRECTORY_PICKLE_DATA_OUTPUT}/y-{participant}-{data_type}.pickle","wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()

    def generate_data_split_and_create(self):
        self.training_data = []
        self.testing_data = []
        for participant in Participants:
            print(f"Participant: {participant}\n")

            for trigger in Triggers:
                # Directory manipulation
                self.path = f"{DIRECTORY_CSV_DATA_ROOT}/{trigger}/{participant}"

                # Create data split for training and validation data
                training_and_validation_epochs = []
                testing_epochs = []
                training_and_validation_epochs, testing_epochs = self.generate_split_data_type_epoch_list(trigger, participant)

                training_data = self.populate_data_type_epoch_lists(training_and_validation_epochs, self.training_data, "Training", trigger)
                testing_data = self.populate_data_type_epoch_lists(testing_epochs, self.testing_data, "Test", trigger)

                random.shuffle(self.training_data)
                random.shuffle(self.testing_data)
                print()

            random.shuffle(training_data)
            random.shuffle(testing_data)

            self.create_data_type_pickles(self.training_data, "Training", participant)
            self.create_data_type_pickles(self.testing_data, "Test", participant)

            print("------------------------------------------------------------------------------------------------------------------------------------------------------------\n")

if __name__ == '__main__':
    # app = Pickles(Participants, Triggers)
    app = Pickles()
    app.generate_data_split_and_create()
