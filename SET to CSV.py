import sys
import os
import mne
import pandas as pd
import numpy as np

DIRECTORY_SET_DATA_ROOT = "E:\\Skripsie\\Data\\1-Pre_Processed-edited"
DIRECTORY_CSV_DATA_OUTPUT = "E:\\Skripsie\\Data\\2-CSV-edited"

Names = ['A', 'B', 'C', 'D']
Triggers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

for name in Names:
    for trigger in Triggers:
        try:
            # Set LEADING_ZERO based on naming convention 
            # a 0 prefix has been used for triggers < 9
            # eg: 01
            if trigger <= 9:
                LEADING_ZERO = "0"
            else:
                LEADING_ZERO = ""

            # Read in the SET file
            directory = f"{DIRECTORY_SET_DATA_ROOT}\{name}"
            filename = f"T{LEADING_ZERO}{trigger}.set"
            df = mne.io.read_epochs_eeglab(f"{directory}\{filename}")
 
            i = 0

            for epoch in df:
                # Create output directory if it does not exist
                if i == 0:
                    try: 
                        os.makedirs(f"{DIRECTORY_CSV_DATA_OUTPUT}\{name}", exist_ok=True)
                    except:
                        print("The path already exists")
                
                i += 1

                # Save CSV in desired location
                dfEpoched = pd.DataFrame(epoch)
                directory = f"{DIRECTORY_CSV_DATA_OUTPUT}\{name}"
                filename = f"T{LEADING_ZERO}{trigger}_{i:03d}.csv"
                np.savetxt(f"{directory}\{filename}", dfEpoched, delimiter=',')

        except IOError as e:
            # Print error message and remind user that an error needs to be adressed afterwards
            print(f"Error: {e}")
            print("")
            print(f"An error occurded for participant {name} in trigger {trigger}.")
            print("Remeber to check back and correct this!")

print("Sucessfully converted SET files to CSVs")
