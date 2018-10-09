%% Example of batch code to preprocess multiple subjects

% Step 1: Change the option to use double precision.

cd('/Ben/eeglab/external/data')
rawDataFiles = dir('*.vhdr');
for subjID = 1:length(rawDataFiles) 
    loadName = rawDataFiles(subjID).name;
    dataName = loadName(1:end-4);
    
    % Step2: Import data.
    EEG = pop_loadbv('/Ben/eeglab/external/data',loadName);
    EEG.setname = dataName;

    % Step 3: Downsample the data.
    %EEG = pop_resample(EEG, 250);
    
    % Step 4: High-pass filter the data at 1-Hz. Note that EEGLAB uses pass-band edge, therefore 1/2 = 0.5 Hz.
    EEG = pop_eegfiltnew(EEG, 1, 0, 1650, 0, [], 0);
    
    % Step 5: Import channel info
    EEG = pop_chanedit(EEG, 'lookup','C:\Ben\eeglab\plugins\dipfit2.3\standard_BEM\elec\standard_1005.elc','eval','chans = pop_chancenter( chans, [],[]);');
    
    % Step 6: Remove line noise using CleanLine
    EEG = pop_cleanline(EEG, 'bandwidth', 2,'chanlist', [1:EEG.nbchan], 'computepower', 0, 'linefreqs', [50 100 150 200 250],...
        'normSpectrum', 0, 'p', 0.01, 'pad', 2, 'plotfigures', 0, 'scanforlines', 1, 'sigtype', 'Channels', 'tau', 100,...
        'verb', 1, 'winsize', 4, 'winstep', 4);
    
    % Step 7: Apply clean_rawdata() to reject bad channels and correct continuous data using Artifact Subspace Reconstruction (ASR)
    originalEEG = EEG;
    EEG = clean_rawdata(EEG, 5, -1, 0.85, 4, 20, 0.25);
    
    % Step 8: Interpolate all the removed channels
    EEG = pop_interp(EEG, originalEEG.chanlocs, 'spherical');

    % Step 9: Re-reference the data to average
    EEG.nbchan = EEG.nbchan+1;
    EEG.data(end+1,:) = zeros(1, EEG.pnts);
    EEG.chanlocs(1,EEG.nbchan).labels = 'initialReference';
    EEG = pop_reref(EEG, []);
    EEG = pop_select( EEG,'nochannel',{'initialReference'});
    
    % Step 10: Run AMICA using calculated data rank with 'pcakeep' option
    if isfield(EEG.etc, 'clean_channel_mask')
        dataRank = min([rank(double(EEG.data')) sum(EEG.etc.clean_channel_mask)]);
    else
        dataRank = rank(double(EEG.data'));
    end
    runamica15(EEG.data, 'num_chans', EEG.nbchan,...
        'outdir', ['/Ben/eeglab/external/data/' dataName],...
        'pcakeep', dataRank, 'num_models', 1,...
        'do_reject', 1, 'numrej', 15, 'rejsig', 3, 'rejint', 1);
    EEG.etc.amica  = loadmodout15(['/Ben/eeglab/external/data/' dataName]);
    EEG.etc.amica.S = EEG.etc.amica.S(1:EEG.etc.amica.num_pcs, :); % Weirdly, I saw size(S,1) be larger than rank. This process does not hurt anyway.
    EEG.icaweights = EEG.etc.amica.W;
    EEG.icasphere  = EEG.etc.amica.S;
    EEG = eeg_checkset(EEG, 'ica');

    % Step 11: Estimate single equivalent current dipoles
    % Note: if ft_datatype_raw() complains about channel numbers, comment out (i.e. put % letter in the line top) line 88 as follows
    % assert(size(data.trial{i},1)==length(data.label), 'inconsistent number of channels in trial %d', i);
    
    % CASE1: If you are using template channel locations for all the subjects:
    %        -> Perform 'Head model and settings' from GUI on one of the
    %           subjects, then type 'eegh' to obtain 9 parameters called 'coord_transform'
    %           For example, my 32ch data has [0.68403 -17.0438 0.14956 1.3757e-07 1.0376e-08 -1.5708 1 1 1], therefore
    %
    %           coordinateTransformParameters = [0.68403 -17.0438 0.14956 1.3757e-07 1.0376e-08 -1.5708 1 1 1];
    %
    % CASE2: If you are using digitized (i.e. measured) channel locations that have 10-20 system names (Fz, Cz, ...) 
    %        -> Calculate the invidivualized transform parameters usign all scalp channels in this way
             
              [~,coordinateTransformParameters] = coregister(EEG.chanlocs, 'C:\Ben\eeglab\plugins\dipfit2.3\standard_BEM\elec\standard_1005.elc', 'warp', 'auto', 'manual', 'off')
    
    % CASE3: If you are using digitized channel locations that do NOT have 10-20 system names
    %        -> Identify several channels that are compatible with 10-20 channel locations, rename them with 10-20 system labels, perform CASE2, then rename them back.
    %           Alternatively, rename the fiducial channels under EEG.chaninfo.nodatchans into 'Nz', 'LPA', 'RPA' accordingly, then use these fiducial channels to perform CASE2 in this way
    %
    %           [~,coordinateTransformParameters] = coregister(EEG.chaninfo.nodatchans, '[EEGLABroot]/eeglab/plugins/dipfit2.3/standard_BEM/elec/standard_1005.elc', 'warp', 'auto', 'manual', 'off')

    templateChannelFilePath = 'C:\Ben\eeglab\plugins\dipfit2.3\standard_BEM\elec\standard_1005.elc';
    hdmFilePath             = 'C:\Ben\eeglab\plugins\dipfit2.3\standard_BEM\standard_vol.mat';
    EEG = pop_dipfit_settings( EEG, 'hdmfile', hdmFilePath, 'coordformat', 'MNI',...
        'mrifile', 'C:\Ben\eeglab\plugins\dipfit2.3\standard_BEM\standard_mri.mat',...
        'chanfile', templateChannelFilePath, 'coord_transform', coordinateTransformParameters,...
        'chansel', 1:EEG.nbchan);
    EEG = pop_multifit(EEG, 1:EEG.nbchan,'threshold', 100, 'dipplot','off','plotopt',{'normlen' 'on'});
    
    % Step 12: Search for and estimate symmetrically constrained bilateral dipoles
    EEG = fitTwoDipoles(EEG, 'LRR', 35);

    % Save the dataset
    EEG = pop_saveset( EEG, 'filename', dataName, 'filepath', '/Ben/eeglab/external/data');
end
