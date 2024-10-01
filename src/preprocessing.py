import os
import numpy as np
from pathlib import Path
import mne
from sklearn.model_selection import train_test_split
import src.config as config
from src.utils import preprocess_data, get_all_processed_files, extract_features
import pdb
import warnings

warnings.filterwarnings('ignore')

class PerceptionImaginationDataProcessor:
    def __init__(self):
        self.name = 'PerceptionImagination'
        self.destinationDir = Path(config.dataDir, self.name)  

    def get_perception_imagination_data_subject(self, filepath):
        print('\033[1;34m***************** Loading Perception/Imagination Data *********************\033[0m')
        print(f'\033[0;36mFile: {filepath}\033[0m')
        data = mne.io.read_raw_fif(filepath, verbose=False, preload=True)  

        #data = cleanData(data)
        events, eventIds = mne.events_from_annotations(data, verbose=False)
        eventIdsReversed = {str(value): key for key, value in eventIds.items()}
        codes, eventTimings = [], []
        for event in events:
            eventCode = eventIdsReversed.get(str(event[2]), None)
            if eventCode:
                code = self._getCode(eventCode)
                if code:
                    codes.append(code)
                    eventTimings.append(event[0])

        perceptionImaginationEvents = [[timing, 0, code] for timing, code in zip(eventTimings, codes)]
        perceptionImaginationEventIds = {'Perception': 1, 'Imagination': 2}
        
        epochs = mne.Epochs(
            data, perceptionImaginationEvents, 
            event_id=perceptionImaginationEventIds, 
            tmin=config.tmin, tmax=config.tmax, 
            preload=True, verbose=False
        )
        
        features = extract_features(epochs)
        perceptionIndexs = []
        imaginationIndexs = []
        for index in range(epochs.events.shape[0]):
            event = epochs.events[index]
            if event[2] == 1:
                perceptionIndexs.append(index)
            else:
                imaginationIndexs.append(index)


        perceptionData, imaginationData = epochs[perceptionIndexs].get_data(), epochs[imaginationIndexs].get_data()
        labels = np.concatenate(([0] * perceptionData.shape[0], [1] * imaginationData.shape[0]), axis=0)
        perceptionFeatures, imaginationFeatures = features[perceptionIndexs], features[imaginationIndexs]
        data = np.concatenate((perceptionData, imaginationData), axis=0)
        features = np.concatenate((perceptionFeatures, imaginationFeatures), axis=0)

        data = np.concatenate((data, features), axis=2)

        print(f'\033[0;32mX: {data.shape}, Y {labels.shape}\033[0m')
        print('\033[1;34m***************** Loaded Perception/Imagination Data *********************\033[0m')
        return data,  labels

    @staticmethod
    def _getCode(event):
        if 'Perception' in event:
            return 1
        elif 'Imagination' in event:
            return 2
        else:
            return None



    def preprocess_perception_imagination_data_all_Subjects(self):
        print('\033[1;35m****Perception/Imagination Data Preprocessing****\033[0m')
        #self.destinationDir = Path(self.destinationDir, "Cleaned")
        os.makedirs(self.destinationDir, exist_ok=True)
        filepaths = get_all_processed_files()
        subjectIds = []
        sessionIds = []
        trainData = None
        trainLabels = None
        testData = None
        testLabels = None
        testSizes = []
        print("\033[1;36mAll filepaths:\033[0m")  # Cyan color for the header
        for filepath in filepaths:
            print(f"\033[0;32m{filepath}\033[0m")
        for index in range(len(filepaths)):
            subjectIds.append(filepaths[index].split(config.seperator)[-4])
            sessionIds.append(filepaths[index].split(config.seperator)[-3])
            X, y = self.get_perception_imagination_data_subject(filepaths[index])
            xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

            
            if index == 0:
                trainData = xTrain
                trainLabels = yTrain
                testData = xTest
                testLabels = yTest
                testSizes.append(xTest.shape[0])   
            else:
                trainData =  np.concatenate((trainData, xTrain), axis=0)
                trainLabels = np.concatenate((trainLabels, yTrain))
                testData =  np.concatenate((testData, xTest), axis=0)
                testLabels = np.concatenate((testLabels, yTest), axis=0)
                testSizes.append(xTest.shape[0])
            
        print(f'\033[0;33mSaving files to {self.destinationDir} directory\033[0m')
        np.save(Path(self.destinationDir, 'xTrain.npy'), trainData)
        np.save(Path(self.destinationDir, 'yTrain.npy'), trainLabels)
        np.save(Path(self.destinationDir, 'xTest.npy'), testData)
        np.save(Path(self.destinationDir, 'yTest.npy'), testLabels)
        np.save(Path(self.destinationDir, 'SubjectIds.npy'), subjectIds)
        np.save(Path(self.destinationDir, 'SessionIds.npy'), sessionIds)
        np.save(Path(self.destinationDir, 'TestSizes.npy'), testSizes)

def perception_imagination_preprocessingPipeline():
    name = 'PerceptionImagination'
    print('\033[1;35m****Starting Perception/Imagination Preprocessing Pipeline****\033[0m')
    perceptionImaginationDataProcessor = PerceptionImaginationDataProcessor()
    perceptionImaginationDataProcessor.preprocess_perception_imagination_data_all_Subjects()
   
    print('\033[1;35m****Completed Perception/Imagination Preprocessing Pipeline****\033[0m')



