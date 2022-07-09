import mne
import pandas as pd
import numpy as np

Numbers = [1,2,3,4,5,6]
Triggers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

for number in Numbers:
    for trigger in Triggers:
        try:
            if trigger <= Triggers[8]:
                df = mne.io.read_epochs_eeglab('G:\\a Data\\0 Preprocessed\\Leroy\\epoch\\L Test{} T0{}.set'.format((number),(trigger)))
            else:
                df = mne.io.read_epochs_eeglab('G:\\a Data\\0 Preprocessed\\Leroy\\epoch\\L Test{} T{}.set'.format((number),(trigger)))
                
            i = 0
            for epoch in df:
                i=i+1
                dfNew1 = pd.DataFrame(epoch)
                if trigger <= Triggers[8]:
                    np.savetxt("L Test"+str(number)+" T0"+str(trigger)+" E"+format(i,'02d')+".csv",dfNew1,delimiter=',')
                else:
                    np.savetxt("L Test"+str(number)+" T"+str(trigger)+" E"+format(i,'02d')+".csv",dfNew1,delimiter=',')

        except IOError:
            print("Error")
print("DONE")  