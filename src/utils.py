import mne
from scipy.stats import zscore
from colorama import Fore, Style, init
from pathlib import Path
import os
import src.config as config
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import pywt
import multiprocessing as mp
import src.config as config

init(autoreset=True)



def computeSegmentFeatures(segment, sfreq, scales, freqBands):
    nChannels = segment.shape[0]
    
    means = np.mean(segment, axis=1)
    stds = np.std(segment, axis=1)
    skewnesses = skew(segment, axis=1)
    kurts = kurtosis(segment, axis=1)
    
    freqs, psds = welch(segment, sfreq, axis=1, nperseg=2*sfreq)
    
    bandPowers = [np.sum(psds[:, (freqs >= low) & (freqs < high)], axis=1) for low, high in freqBands.values()]
    bandPowers = np.array(bandPowers).T
    
    morletFeatures = [np.sum(np.abs(pywt.cwt(segment[ch], scales, 'cmor1.5-1.0', sampling_period=1/sfreq)[0])**2, axis=1) for ch in range(nChannels)]
    morletFeatures = np.array(morletFeatures)
    
    segmentFeatures = np.hstack((
        means[:, np.newaxis],
        stds[:, np.newaxis],
        skewnesses[:, np.newaxis],
        kurts[:, np.newaxis],
        bandPowers,
        morletFeatures
    ))

    return segmentFeatures

def extractFeatures(epochs, sfreq=config.samplingFrequency):
    printHeader('Extracting Features')
    
    segmentedData = epochs.get_data()
    nSegments, nChannels, nTimes = segmentedData.shape
    
    print(f"{Fore.CYAN}Input shape: {Fore.YELLOW}{nSegments} segments, {nChannels} channels, {nTimes} time points{Style.RESET_ALL}")
    
    freqBands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100),
    }
    
    print(f"{Fore.CYAN}Frequency bands: {Fore.YELLOW}{', '.join(freqBands.keys())}{Style.RESET_ALL}")
    
    frequencies = np.linspace(1, sfreq / 2, 50)
    scales = pywt.scale2frequency('cmor1.5-1.0', frequencies) * sfreq
    
    print(f"{Fore.CYAN}Computing features using {Fore.YELLOW}{config.numJobs} processes{Style.RESET_ALL}")
    
    with mp.Pool(processes=config.numJobs) as pool:
        results = pool.starmap(
            computeSegmentFeatures, 
            [(segmentedData[i], sfreq, scales, freqBands) for i in range(nSegments)]
        )
    
    features = np.array(results)
    
    print(f"{Fore.CYAN}Features shape: {Fore.YELLOW}{features.shape}{Style.RESET_ALL}")
    
    return features









def printHeader(message):
    print("\n" + "="*50)
    print(f'\033[1m{Fore.BLACK}{message}{Style.RESET_ALL}\033[0m]')
    print("="*50)

def printFooter(message):
    print("\n" + "="*50)
    print(f'\033[1m{Fore.GREEN}{message}{Style.RESET_ALL}\033[0m]')
    print("="*50)

def cleanData(mneData):
    printHeader("Preprocessing the data")
    data = mneData.copy()
    
    print(f"{Fore.CYAN}1. {Style.BRIGHT}Applying notch filter{Style.RESET_ALL}")
    data.notch_filter([50, 100])
    
    print(f"{Fore.CYAN}2. {Style.BRIGHT}Applying bandpass filter{Style.RESET_ALL}")
    data.filter(l_freq=0.5, h_freq=150)
    
    print(f"{Fore.CYAN}3. {Style.BRIGHT}Setting EEG reference{Style.RESET_ALL}")
    data.set_eeg_reference("average", projection=True)
    
    print(f"{Fore.CYAN}4. {Style.BRIGHT}Performing ICA{Style.RESET_ALL}")
    ica = mne.preprocessing.ICA(
        n_components=20, 
        random_state=97,
        max_iter='auto'
    )

    print(f"{Fore.CYAN}5. {Style.BRIGHT}Fitting ICA{Style.RESET_ALL}")
    ica.fit(data)
    
    print(f"{Fore.CYAN}6. {Style.BRIGHT}Applying ICA{Style.RESET_ALL}")
    ica.apply(data)

    printFooter("Data preprocessing completed")
    return data

def getAllFifFilesFromFolder(directory):
    printHeader(f"Searching for .fif files in: {directory}")
    fifFiles = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.fif') and 'eeg-1' not in file:
                filePath = os.path.join(root, file)
                fifFiles.append(filePath)
    printFooter(f"Found {len(fifFiles)} .fif files")
    return fifFiles

def getDirFromFolder(folderPath):
    printHeader(f"Getting directories from: {folderPath}")
    entries = os.listdir(folderPath)
    directories = [entry for entry in entries if os.path.isdir(os.path.join(folderPath, entry))]
    directoriesWithPaths = [Path(folderPath, folder) for folder in directories]
    printFooter(f"Found {len(directories)} directories")
    return directories, directoriesWithPaths

def getAllPreprocessedFiles(folderPath=config.rawDataDir):
    printHeader(f"Collecting all preprocessed files from: {folderPath}")
    allFilePaths = []
    _, subjectFolders = getDirFromFolder(folderPath)
    for subject in subjectFolders:
        _, sessionFolders = getDirFromFolder(subject)
        for session in sessionFolders:
            dirPath = Path(session, 'eeg')
            files = getAllFifFilesFromFolder(dirPath)
            for file in files:
                allFilePaths.append(file)
                
    allFilePaths = [filepath for filepath in allFilePaths if 'sub-013' not in filepath]
    
    printFooter(f"Found {len(allFilePaths)} preprocessed files")
    return allFilePaths