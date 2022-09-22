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
        self.version: str = None
        self.comparison: int = None
        # model
        self.split: float = None

    def load_yaml(self) -> None:
        self.config = EnvYAML("./setup/config.yaml", strict=False)

    def populate_config(self) -> None:
        # TODO parse in as cli command from make file (argparse)
        self.os = "mac_m1"
        self.experiment = "libet"

        self.csvs = self.config[f"os.{self.os}.io_paths.csv_files"]
        self.pickles = self.config[f"os.{self.os}.io_paths.pickle_files"]
        self.name = self.config[f"experiment.details.{self.experiment}.name"]
        self.participants = self.config[f"experiment.details.{self.experiment}.participants"]
        self.triggers = self.config[f"experiment.details.{self.experiment}.triggers"]
        self.version = self.config[f"experiment.details.{self.experiment}.version"]
        self.comparison = self.config[f"model_parameters.comparison"]
        self.split = self.config["model_parameters.percentage_training_&_validation_to_testing_split"]
    
    # Split the CSV data into [training & validation (grouped)] and testing epochs:
    #   based on the percentage_training_&_validation_to_testing_split in config.yaml
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
                    i = self.exists_in_list(int(indicies[csv_file] + 1), i)
                else:
                    if i < len(indicies): # no minus 1 as counting starts at 1 and not at 0
                        self.testing_epoch_list.append(f"{self.trigger}{i}.csv")
                        i += 1
                    else:
                        break
            
        except Exception as e:
            print(f"Error: {e}")

    # TODO description of this function
    # TODO rename funciton to something more descriptive
    def exists_in_list(self, x, i):
        while x != i:
            if x == i:
                pass
            else:
                self.testing_epoch_list.append(f"{self.trigger}{i}.csv")
                i += 1
        i += 1
        return i

    # Create and populate the Training/Test data
    # TODO add type hints for function
    def populate_data_type_epoch_lists(self, epoch_list, data, data_type: str):
        print(f"Generating {data_type} data (consisting of {len(epoch_list)} epochs) for the {self.trigger} trigger.")
        
        trigger_instance: int = self.triggers.index(self.trigger)
        scaler = StandardScaler()

        # TODO have better try catch logic so that you know exactly where the error happened.
        # return listed in func name
        # return after exception? 
        try:
            for epoch in tqdm(epoch_list):
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
    # TODO add type hints for function
    def create_data_type_pickles(self, data, data_type: str):
        print(f"Generating Pickles for {data_type} data")
        
        X: list = []
        y: list = []

        random.shuffle(data)

        for features, label in data:
            X.append(features)
            y.append(label)

        # Parenthesis depend on the input data -1 being batch size, channels, datasamples, idk
        # TODO reshape size adjusted based on experiment (automated)
        #   Original: 128 x 257
        #   New: 128 x 129
        # Idea: shape of X
        X = np.array(X).reshape(-1,128,129,1)

        # TODO Improve below logic to only run once per version
        try:
            os.makedirs(f"{self.pickles}/{self.version}")
        except:
            print("already exists")

        pickle_out: BufferedWriter = open(f"{self.pickles}/{self.version}/X-{self.participant}-{data_type}.pickle","wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

        pickle_out: BufferedWriter = open(f"{self.pickles}/{self.version}/y-{self.participant}-{data_type}.pickle","wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()
        
        print("---------------------------------------------------------------------------------------------------------------------------------------------------\n")

    def generate_data_split_and_create(self):
        self.participant: int = None
        self.trigger: str = None
        # TODO add type hints
        self.training_data = []
        self.testing_data = []

        for self.participant in self.participants:
            print(f"Participant: {self.participant}\n")
            # TODO this would need to be called differently (earlier) when the self.comparison == 0:
            #   Or else we will be throwing away data
            # TODO add type hints
            training_and_validation_data, testing_data = [], []

            for self.trigger in self.triggers:
                # Directory manipulation
                self.epochs = f"{self.csvs}/{self.trigger}/P{self.participant}"

                # Create data split for training and validation data
                self.generate_split_data_type_epoch_list()

                # Save each selected epoch into a list for both groups of data listed below
                training_and_validation_data = self.populate_data_type_epoch_lists(self.training_and_validation_epoch_list, self.training_data, "Training")
                testing_data = self.populate_data_type_epoch_lists(self.testing_epoch_list, self.testing_data, "Testing")                             

                random.shuffle(training_and_validation_data)
                random.shuffle(testing_data)
                print()

            if self.comparison == 1:
                self.create_data_type_pickles(training_and_validation_data, "Training")
                self.create_data_type_pickles(testing_data, "Testing")
        
        # TODO change naming convention for these
        #   As it will be all participants in the same pickles.
        if self.comparison == 0:
            random.shuffle(training_and_validation_data)
            random.shuffle(testing_data)

            self.create_data_type_pickles(training_and_validation_data, "Training")
            self.create_data_type_pickles(testing_data, "Testing")


if __name__ == '__main__':
    app = Pickles()
    app.load_yaml()
    app.populate_config()
    app.generate_data_split_and_create()
