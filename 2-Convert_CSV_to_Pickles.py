"""
Author: Ben Wolfaardt

DESCRIPTION: 

Convert our CSV files to Pickles

This eliminates the necessity of having to load all the csv files every time you change your model.
Our pickles are split into training and validation (grouped) as well as testing pickles for both 
features, X, and labels, y. 

The ratio of training and validation (grouped) to testing split is set in the config.yaml file: 
    model_parameters -> percentage_training_&_validation_to_testing_split

Considering our small dataset a value of between 70 and 80 percent is suggested

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
        # config
        self.os: str = None
        self.config: EnvYAML = None
        self.experiment: str = None
        # paths
        self.csvs: str = None
        self.pickles: str = None
        # experiment details
        self.name: str = None
        self.participants: list[int] = []
        self.triggers: list[str] = []
        self.comparison: int = None
        # model
        self.split: float = None

    def load_yaml(self) -> None:
        self.config = EnvYAML("./setup/config.yaml", strict=False)

    def populate_config(self):
        # TODO parse in as cli command from make file
        self.os = "mac_m1"
        self.experiment = "libet"

        # TODO add name into folder save or something
        self.csvs = self.config[f"os.{self.os}.io_paths.csv_files"]
        self.pickles = self.config[f"os.{self.os}.io_paths.pickle_files"]
        self.name = self.config[f"experiment.details.{self.experiment}.name"]
        self.participants = self.config[f"experiment.details.{self.experiment}.participants"]
        self.triggers = self.config[f"experiment.details.{self.experiment}.triggers"]
        self.comparison = self.config[f"model_parameters.comparison"]
        self.split = self.config["model_parameters.percentage_training_&_validation_to_testing_split"]
    
    # Split the CSV data into [training & validation (grouped)] and 
    # testing epochs based on the PERCENTAGE_TRAINING_AND_VALIDATION
    # TODO seems like the random generator's rounding makes us loose 1 or 2 files, to investigate
    def generate_split_data_type_epoch_list(self) -> None:
        self.training_and_validation_epoch_list: list[str] = []
        self.testing_epoch_list: list[str] = []

        # TODO have better try catch logic so that you know exactly where the error happened.
        try:
            # Random split of data into [training & validation (grouped)] and testing data
            list_csv_files = fnmatch.filter(os.listdir(self.epochs), '*.csv')
            k = len(list_csv_files) * self.split // 100
            k = round(k)
            k = int(k)
            indicies = random.sample(range(len(list_csv_files)), k)
            indicies.sort()
            
            self.training_and_validation_epoch_list = [list_csv_files[i] for i in indicies]

            i = 1
            for csv_file in range(len(list_csv_files)):
                if csv_file < len(indicies):
                    i = self.exists_in_list(self.testing_epoch_list, int(indicies[csv_file] + 1), i, self.trigger, self.participant)
                else:
                    if i < len(indicies): # no minus 1 as counting starts at 1 and not at 0
                        self.testing_epoch_list.append(f"P{self.participant}{self.trigger}{i}.csv")
                        i += 1
                    else:
                        break
            
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
    def populate_data_type_epoch_lists(self):
        print(f"Generating {self.data_type} data (consisting of {len(self.selected_epochs)} epochs) for the {self.trigger} trigger.")
        
        trigger_instance: int = self.triggers.index(self.trigger)
        scaler = StandardScaler()
        data: list[NDArray, int] = []

        # TODO have better try catch logic so that you know exactly where the error happened.
        try:
            for epoch in tqdm(self.selected_epochs):
                epoch_array: NDArray = np.genfromtxt(f"{self.epochs}/{epoch}", delimiter=',')
                new_array: NDArray = scaler.fit_transform(epoch_array, None)
                # TODO check the reshape logic for Siobhan
                # Instead of having the full 0:750 ms now we are reshaping it to 100:550 ms
                # reshape_array = new_array[:,100:550]
                data.append([new_array, trigger_instance]) # where the label is added to the feature
            
            return data

        except Exception as e:
            print(f"An error occurded whilst creating the {self.data_type} data for trigger: {self.trigger} epoch {epoch}.")
            quit()
            # TODO raise flag so that this is printed at the end. 
            # print("Remeber to check back and correct this!")

    # Create and populate the pickles from the Training/Testing data
    def create_data_type_pickles(self, data):
        print(f"Generating Pickles for {self.data_type} data")
        
        X: list = []
        y: list = []

        random.shuffle(data)

        for features, label in data:
            X.append(features)
            y.append(label)
        # Parenthesis depend on the input data -1 being batch size, channels, datasamples, idk
        # TODO reshape size adjusted based on experiment
        X = np.array(X).reshape(-1,128,257,1)

        pickle_out: BufferedWriter = open(f"{self.pickles}/X-{self.participant}-{self.data_type}.pickle","wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

        pickle_out: BufferedWriter = open(f"{self.pickles}/y-{self.participant}-{self.data_type}.pickle","wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()
        
        print("---------------------------------------------------------------------------------------------------------------------------------------------------\n")

    def generate_data_split_and_create(self):
        self.participant: int = None
        self.trigger: str = None
        self.data_type: str = None
        data_types = ["Training", "Testing"]

        for self.participant in self.participants:
            print(f"Participant: {self.participant}\n")

            for self.trigger in self.triggers:
                # Directory manipulation
                self.epochs = f"{self.csvs}/{self.trigger}/{self.participant}"

                # Create data split for training and validation data
                self.generate_split_data_type_epoch_list()

                for self.data_type in data_types:
                    if self.data_type == "Training":
                        self.selected_epochs = self.training_and_validation_epoch_list
                        training_and_validation_data = self.populate_data_type_epoch_lists()
                    elif self.data_type == "Testing":
                        self.selected_epochs = self.testing_epoch_list
                        testing_epoch_list_data = self.populate_data_type_epoch_lists()

                random.shuffle(training_and_validation_data)
                random.shuffle(testing_epoch_list_data)
                print()

            if self.comparison == 1:
                self.create_data_type_pickles(training_and_validation_data)
        
        if self.comparison == 0:
            self.create_data_type_pickles(testing_epoch_list_data)


if __name__ == '__main__':
    app = Pickles()
    app.load_yaml()
    app.populate_config()
    app.generate_data_split_and_create()
