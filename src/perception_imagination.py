import os
import numpy as np
import pandas as pd
from pathlib import Path
import pdb
import src.config as config
import warnings
from termcolor import colored
from scipy.stats import zscore

from src.models import EEGNet, EegNetSsvepN, DeepConvNet, SegmentedSignalNet
from src.models import XGBoostModel, SVCModel, RandomForestModel
from src.trainer import EEGModelTrainer

warnings.filterwarnings('ignore')

class PerceptionImaginationData:
    def __init__(self):
        self.name = 'PerceptionImagination'
        self.dataDir = Path(config.dataDir, self.name)
        print(colored(' Loading processed data...', 'cyan', attrs=['bold']))
        self.destinationDir = config.resultsDir
        
        self.load_imagination_perception_data()
        
    def load_imagination_perception_data(self):
        files = os.listdir(self.dataDir)
        print(colored(f' Data directory: {self.dataDir}', 'yellow'))
        print(colored('易 Loading Perception-Imagination Data for All Subjects', 'green'))
        
        for file in files:
            filepath = Path(self.dataDir, file)
            if 'Session' in file:
                self.sessionIds = np.load(filepath)
            elif 'Subject' in file:
                self.subjectIds = np.load(filepath)
            elif 'xTrain' in file:
                self.xTrain = np.load(filepath)
            elif 'xTest' in file:
                self.xTest = np.load(filepath)
            elif 'yTrain' in file:
                self.yTrain = np.load(filepath)
            elif 'yTest' in file:
                self.yTest = np.load(filepath)
            elif 'TestSizes' in file:
                self.testSizes = np.load(filepath)
        print(colored('✅ Loaded Perception-Imagination for All Subjects', 'green', attrs=['bold']))

def train_models_perception_imagination():
    taskType = 'PerceptionImagination'
    numClasses =  2
    dataLoader = PerceptionImaginationData()
    xTrain, xTest = dataLoader.xTrain, dataLoader.xTest
    yTrain, yTest = dataLoader.yTrain, dataLoader.yTest
    print(colored(f' Xtrain: {xTrain.shape}, xTest: {xTest.shape}', 'magenta'))
    xTrainFeatures = xTrain[:,:,1537:]
    xTestFeatures = xTest[:,:,:1537:]
    xTrain = xTrain[:,:,537:1537]
    xTest = xTest[:,:,537:1537]

    xTrain = zscore(xTrain)
    xTrainFeatures = zscore(xTrainFeatures)
    xTestFeatures = zscore(xTestFeatures)
    xTest = zscore(xTest)
    
    print(colored(f' Xtrain: {xTrain.shape}, xTest: {xTest.shape}', 'magenta'))
         
    
    #Deep Learning Based Models Training
    
    modelBuilder = SegmentedSignalNet(numClasses=numClasses)
    model = modelBuilder.buildModel()
    trainner = EEGModelTrainer(
        model=model,
        modelName='CNNSegmented',
        taskType=taskType
    )
    
    trainner.train(
        xTrain=(xTrain, xTrainFeatures),
        yTrain=yTrain,
        xVal=(xTest, xTestFeatures),
        yVal=yTest
    )
    
    modelBuilder = EEGNet(numClasses=numClasses, samples=1000)
    model = modelBuilder.buildModel()
    trainner = EEGModelTrainer(
        model=model,
        modelName="EEGNet",
        taskType=taskType
    )
    
    trainner.train(
        xTrain=xTrain,
        yTrain=yTrain,
        xVal=xTest,
        yVal=yTest
    )
    
    modelBuilder = EegNetSsvepN(numClasses=numClasses, samples=1000)
    model = modelBuilder.buildModel()
    trainner = EEGModelTrainer(
        model=model,
        modelName="EEGNetSsvepN",
        taskType=taskType
    )
    trainner.train(
        xTrain=xTrain,
        yTrain=yTrain,
        xVal=xTest,
        yVal=yTest
    )
    
    modelBuilder = DeepConvNet(numClasses=numClasses, samples=1000)
    model = modelBuilder.buildModel()
    trainner = EEGModelTrainer(
        model=model,
        modelName="DeepConvNet",
        taskType=taskType
    )
    trainner.train(
        xTrain=xTrain,
        yTrain=yTrain,
        xVal=xTest,
        yVal=yTest
    )

def evaluateAllModelsForPerceptionAndImaginationDecoding():
    taskType = 'PerceptionImagination'
    destinationDir = Path(config.resultsDir, 'Reports')
    destinationDir = Path(destinationDir, taskType)
    os.makedirs(destinationDir, exist_ok=True)
    dataLoader = PerceptionImaginationData(cleaned=True)
    xTrain, xTest = dataLoader.xTrain, dataLoader.xTest
    yTrain, yTest = dataLoader.yTrain, dataLoader.yTest

    xTrainFeatures, xTestFeatures = loadExtractedFeatures(
        folder=Path(config.dataDir, taskType )
    )

    xTrain = np.concatenate((xTrain, xTrainFeatures), axis=2)
    xTest =  np.concatenate((xTest, xTestFeatures), axis=2)
    testSizes = dataLoader.testSizes
    sessionIds, subjectIds =  dataLoader.sessionIds, dataLoader.subjectIds
    
    loadedModels, modelNames = loadAllTrainedModels()

    for modelNo in range(0, len(loadedModels)):
        modelName = modelNames[modelNo]
        model = loadedModels[modelNo]
        
        if modelName == 'RF':
            continue
        machineLearningModels = ['XGB', 'SVC', 'RF']
        print(colored(f' Evaluating model: {modelName}', 'blue', attrs=['bold']))
        if modelName in machineLearningModels:
            continue
            dataDir = Path(config.dataDir,taskType, 'CSPFeatures')
            _, xTest1 = loadCSPFeatures(dataDir)
            
            reportOnAllSubjects = classificationReport(model, xTest1, yTest)
            reportOnIndividualSubjects = getIndividualSpecificClassificationReport(
                model=model, xTest=xTest1, yTest=yTest,
                subjectIds=subjectIds, sessionIds=sessionIds,
                testSizes=testSizes    
            )
        else:
           
            print(xTest.shape, yTest.shape)
            reportOnAllSubjects = classificationReport(model, xTest, yTest)
            trainAccuracy = classificationReport(model, xTrain, yTrain)
            reportOnIndividualSubjects = getIndividualSpecificClassificationReport(
                model=model, xTest=xTest, yTest=yTest,
                subjectIds=subjectIds, sessionIds=sessionIds, 
                testSizes=testSizes   
            )
            print(colored(' Test Report:', 'cyan'))
            print(reportOnAllSubjects)
            print(colored(' Train Report:', 'cyan'))
            print(trainAccuracy)
        
        reportOnAllSubjects = pd.DataFrame(reportOnAllSubjects)
        reportOnAllSubjects.to_csv(Path(destinationDir, f'{modelName}_AllSubjects.csv'))
        reportOnIndividualSubjects.to_csv(Path(destinationDir, f'{modelName}_IndividualSubjects.csv'))
        


def evaluateAllModelsForPerceptionAndImaginationDecoding():
    taskType = 'PerceptionImagination'
    destinationDir = Path(config.resultsDir, 'Reports')
    destinationDir = Path(destinationDir, taskType)
    os.makedirs(destinationDir, exist_ok=True)
    dataLoader = PerceptionImaginationData(cleaned=True)
    xTrain, xTest = dataLoader.xTrain, dataLoader.xTest
    yTrain, yTest = dataLoader.yTrain, dataLoader.yTest

    xTrainFeatures = xTrain[:,:,1537:]
    xTestFeatures = xTest[:,:,:1537:]
    xTrain = xTrain[:,:,537:1537]
    xTest = xTest[:,:,537:1537]

    xTrain = zscore(xTrain)
    xTest = zscore(xTest)

   
    testSizes = dataLoader.testSizes
    sessionIds, subjectIds =  dataLoader.sessionIds, dataLoader.subjectIds
    
    loadedModels, modelNames = loadAllTrainedModels()

    for modelNo in range(0, len(loadedModels)):
        modelName = modelNames[modelNo]
        model = loadedModels[modelNo]
        
        if modelName == 'RF':
            continue
        machineLearningModels = ['XGB', 'SVC', 'RF']
        print(f'****************Model name {modelName}')
        if modelName in machineLearningModels:
            continue
            dataDir = Path(config.dataDir,taskType, 'CSPFeatures')
            _, xTest1 = loadCSPFeatures(dataDir)
            
            reportOnAllSubjects = classificationReport(model, xTest1, yTest)
            reportOnIndividualSubjects = getIndividualSpecificClassificationReport(
                model=model, xTest=xTest1, yTest=yTest,
                subjectIds=subjectIds, sessionIds=sessionIds,
                testSizes=testSizes    
            )
        else:
           
            print(xTest.shape, yTest.shape)
            reportOnAllSubjects = classificationReport(model, xTest, yTest)
            trainAccuracy = classificationReport(model, xTrain, yTrain)
            reportOnIndividualSubjects = getIndividualSpecificClassificationReport(
                model=model, xTest=xTest, yTest=yTest,
                subjectIds=subjectIds, sessionIds=sessionIds, 
                testSizes=testSizes   
            )
            print('Test Report')
            print(reportOnAllSubjects)
            print('Train report')
            print(trainAccuracy)
        
        reportOnAllSubjects = pd.DataFrame(reportOnAllSubjects)
        reportOnAllSubjects.to_csv(Path(destinationDir, f'{modelName}_AllSubjects.csv'))
        reportOnIndividualSubjects.to_csv(Path(destinationDir, f'{modelName}_IndividualSubjects.csv'))






















    
        





















    
        
