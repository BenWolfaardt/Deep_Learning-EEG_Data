"""
Author: Ben Wolfaardt

DESCRIPTION: 
The following script has been written to convert SET EEG files into CSV files such 
that they can be used to train our Deep Learning model.

PEASE NOTE: 
That this project assumes that you have your SET files stored in the 
DIRECTORY_SET_DATA_ROOT variable and that you have sub-folders per participant 
in which you have the SET files stored based on their triggers.

eg.

C:\<DIRECTORY_SET_DATA_ROOT>\<Names>\<Trigger>
C:\SET_DATA\A\T01.SET

FURTHER NOTE:
Triggers smaller than 9 have a 0 prefix 

eg: 

T01.SET, T09.SET, T12.SET
"""

import sys
import os
import mne
import pandas as pd
import numpy as np

DIRECTORY_SET_DATA_ROOT = "E:\\Skripsie\\Data\\1-Pre_Processed"
DIRECTORY_CSV_DATA_OUTPUT = "E:\\Skripsie\\Data\\2-CSV"

Participants = ['A', 'B', 'C', 'D']
Triggers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

for participant in Participants:
    for trigger in Triggers:
        try:
            # Set LEADING_ZERO prefix based on trigger naming convention 
            if trigger <= 9:
                LEADING_ZERO = "0"
            else:
                LEADING_ZERO = ""

            # Read in the SET file
            directory = f"{DIRECTORY_SET_DATA_ROOT}\{participant}"
            filename = f"T{LEADING_ZERO}{trigger}.set"
            df = mne.io.read_epochs_eeglab(f"{directory}\{filename}")
 
            i = 0

            for epoch in df:
                # Create output directory if it does not exist
                if i == 0:
                    try: 
                        os.makedirs(f"{DIRECTORY_CSV_DATA_OUTPUT}\{participant}", exist_ok=True)
                    except:
                        print("The path already exists")
                
                i += 1

                # Save CSV in desired location
                dfEpoched = pd.DataFrame(epoch)
                directory = f"{DIRECTORY_CSV_DATA_OUTPUT}\{participant}"
                filename = f"T{LEADING_ZERO}{trigger}_{i:03d}.csv"
                np.savetxt(f"{directory}\{filename}", dfEpoched, delimiter=',')

        except IOError as e:
            # Print error message and remind user that an error needs to be adressed afterwards
            print(f"Error: {e}")
            print("")
            print(f"An error occurded for participant {participant} in trigger {trigger}.")
            print("Remeber to check back and correct this!")

print("Sucessfully converted SET files to CSVs")
