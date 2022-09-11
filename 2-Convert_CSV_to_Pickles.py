"""
Author: Ben Wolfaardt

DESCRIPTION: 

# If you manualy divided the data in different folders to test it after training (take 10 files each class)

# Saving the preprocessed data to feed in to algorithm 
# This way you dont have to load all the csv files every time you change your algorithm

# Testing data is saved in a different pickle, this pickle will be loaded
# when you start fitting the model to test it and generate a confusion matrix (I think)
"""

import argparse
import os
import random
import fnmatch
import pickle
import random
import numpy as np

from envyaml import EnvYAML
from io import BufferedWriter
from nptyping import NDArray
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class Pickles:
    def __init__(self) -> None:
        self.os: str = None
        self.experiment_name: str = None
        self.csvs: str = None
        self.pickles: str = None
        self.epochs: str = None
        self.participants: list[int] = None
        self.participant: int = None
        self.triggers: list[str] = None
        self.trigger: str = None
        self.split: float = None
        self.config: EnvYAML = None

    def load_yaml(self) -> None:
        self.config = EnvYAML("./setup/config.yaml", strict=False)

    def populate_settings(self):
        # TODO parse in as cli command from make file
        self.experiment = "libet"
        self.os = "mac_m1"

        self.name = self.config[f"experiment.details.{self.experiment}.name"]
        self.csvs = self.config[f"os.{self.os}.io_paths.csv_files"]
        self.pickles = self.config[f"os.{self.os}.io_paths.pickle_files"]
        self.participants = self.config[f"experiment.details.{self.experiment}.participants"]
        self.triggers = self.config[f"experiment.details.{self.experiment}.triggers"]
        self.split = self.config["model_parameters.percentage_training_&_validation_to_testing_split"]
    
    # Split the CSV data into [training & validation (grouped)] and 
    # testing epochs based on the PERCENTAGE_TRAINING_AND_VALIDATION
    # TODO seems like the random generator's rounding makes us loose 1 or 2 files, to investigate
    def generate_split_data_type_epoch_list(self):
        try:
            # Random split of data into [training & validation (grouped)] and testing data
            list_csv_files = fnmatch.filter(os.listdir(self.epochs), '*.csv')
            k = len(list_csv_files) * self.split // 100
            k = round(k)
            k = int(k)
            indicies = random.sample(range(len(list_csv_files)), k)
            indicies.sort()
            
            training_and_validation_list = [list_csv_files[i] for i in indicies]
            testing_list = []

            i = 1
            for csv_file in range(len(list_csv_files)):
                if csv_file < len(indicies):
                    i = self.exists_in_list(testing_list, int(indicies[csv_file] + 1), i, self.trigger, self.participant)
                else:
                    if i < len(indicies): # no minus 1 as counting starts at 1 and not at 0
                        testing_list.append(f"P{self.participant}{self.trigger}{i}.csv")
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
    def populate_data_type_epoch_lists(self, selected_epochs, data, data_type):
        print(f"Generating {data_type} data (consisting of {len(selected_epochs)} epochs) for the {self.trigger} trigger.")
        
        trigger_instance: int = self.triggers.index(self.trigger)
        scaler = StandardScaler()

        try:
            for epoch in tqdm(selected_epochs):
                epoch_array: NDArray = np.genfromtxt(f"{self.epochs}/{epoch}", delimiter=',')
                new_array: NDArray = scaler.fit_transform(epoch_array, None)
                # TODO check the reshape logic for Siobhan
                # Instead of having the full 0:750 ms now we are reshaping it to 100:550 ms
                # reshape_array = new_array[:,100:550]
                data.append([new_array, trigger_instance]) # where the label is added to the feature
            
            return data

        except Exception as e:
            print(f"An error occurded whilst creating the {data_type} data for trigger: {self.trigger} epoch {epoch}.")
            quit()
            # TODO raise flag so that this is printed at the end. 
            # print("Remeber to check back and correct this!")

    # Create and populate the pickles from the Training/Testing data
    def create_data_type_pickles(self, data, data_type):
        print(f"Generating Pickles for {data_type} data")
        
        x = []
        y = []

        random.shuffle(data)

        for features, label in data:
            x.append(features)
            y.append(label)
        # Parenthesis depend on the input data -1 being batch size, channels, datasamples, idk
        x = np.array(x).reshape(-1,128,257,1)

        pickle_out: BufferedWriter = open(f"{self.pickles}/X-{self.participant}-{data_type}.pickle","wb")
        pickle.dump(x, pickle_out)
        pickle_out.close()

        pickle_out: BufferedWriter = open(f"{self.pickles}/y-{self.participant}-{data_type}.pickle","wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()

    def generate_data_split_and_create(self):
        training_data: list[str] = []
        testing_data: list[str] = []

        for self.participant in self.participants:
            print(f"Participant: {self.participant}\n")

            for self.trigger in self.triggers:
                # Directory manipulation
                self.epochs = f"{self.csvs}/{self.trigger}/{self.participant}"

                # Create data split for training and validation data
                training_and_validation_epochs: list[str] = []
                testing_epochs: list[str] = []
                training_and_validation_epochs, testing_epochs = self.generate_split_data_type_epoch_list()

                training_data = self.populate_data_type_epoch_lists(training_and_validation_epochs, training_data, "Training")
                testing_data = self.populate_data_type_epoch_lists(testing_epochs, testing_data, "Test")

                random.shuffle(training_data)
                random.shuffle(testing_data)
                print()

            random.shuffle(training_data)
            random.shuffle(testing_data)

            self.create_data_type_pickles(training_data, "Training")
            self.create_data_type_pickles(testing_data, "Test")

            print("---------------------------------------------------------------------------------------------------------------------------------------------------\n")

if __name__ == '__main__':
    app = Pickles()
    app.load_yaml()
    app.populate_settings()
    app.generate_data_split_and_create()
