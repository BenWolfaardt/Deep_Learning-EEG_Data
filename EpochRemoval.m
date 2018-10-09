[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

for test = 1:5
    EEG = pop_loadset('filename',[num2str(test),'.set'],'filepath','C:\\Ben\\eeglab\\external\\data\\');
    
    if test == 1
        [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, test);
    else
        [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, test);
    end
    
    stimuli = 1;
    for stimuli = 1:18
        [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG, [CURRENTSET+1]);
        
        if stimuli < 10
            EEG = pop_epoch( EEG, {['T  ',num2str(stimuli)]}, [-0.25        1.25], 'newname', ['T',num2str(test),' epochs ',num2str(stimuli)], 'epochinfo', 'yes');
        else
            EEG = pop_epoch( EEG, {['T ',num2str(stimuli)]}, [-0.25        1.25], 'newname', ['T',num2str(test),' epochs ',num2str(stimuli)], 'epochinfo', 'yes');
        end
     
        EEG = pop_eegthresh(EEG,1,[1:63] ,-100,100,-0.25,1.248,2,0);
        EEG = pop_jointprob(EEG,1,[1:63] ,6,2,0,0,0,[],0);
        EEG = pop_jointprob(EEG,1,[1:63] ,6,2,0,0,1,[],0);
        EEG = pop_rejkurt(EEG,1,[1:63] ,5,5,0,0,0,[],0);
        EEG = pop_rejkurt(EEG,1,[1:63] ,5,5,0,0,1,[],0);
        EEG = pop_rejtrend(EEG,1,[1:63] ,750,50,0.3,2,0);
        uiwait;
        pop_rejmenu(EEG, 0)
        uiwait;
        [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
        EEG = eeg_rejsuperpose( EEG, 1, 1, 1, 1, 1, 1, 1, 1);  %gives a variable
        %named as follows with all epochs to reject: EEG.reject.rejglobal
        EEG = pop_rejepoch( EEG, EEG.reject.rejglobal ,0);
        [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
        EEG = pop_saveset( EEG, 'filename', ['N Test',num2str(test),' T',num2str(stimuli,'%02.f')], 'filepath', '/Ben/eeglab/external/data/epoch');
        [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET,'retrieve',test,'study',0);
    end
end