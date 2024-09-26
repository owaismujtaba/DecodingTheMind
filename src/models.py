from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from pathlib import Path
import os
import pandas as pd
import pdb

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.keras.layers import DepthwiseConv2D, Activation
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D 
from tensorflow.keras.layers import SpatialDropout2D, Dropout, Concatenate
from tensorflow.keras.layers import  SeparableConv2D, Flatten, Dense
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.activations import swish

import src.config as config
from src.utils import saveTensorFlowModel

from colorama import Fore, Style
import emoji

def print_with_icon(message, icon):
    print(f"{Fore.CYAN}{emoji.emojize(icon)} {message}{Style.RESET_ALL}")



class SegmentedSignalNet:
    def __init__(self, numClasses, inputShape=(128, 128, 1), numAdditionalFeatures=59):
        """
        Initializes the model architecture.

        :param numClasses: Number of output classes (e.g., for classification)
        :param inputShape: Shape of the main input (time_steps, feature_dim, channels)
        :param numAdditionalFeatures: Number of additional features (e.g., 59 features sent to Dense1)
        """
        self.numClasses = numClasses
        self.inputShape = inputShape
        self.numAdditionalFeatures = numAdditionalFeatures
        print('Building SegmentedSignalNet Architecture...')

    def buildModel(self):
        """
        Builds the architecture based on the provided block diagram.

        :return: Compiled TensorFlow/Keras model
        """
        # Input for the main data
        inputMain = Input(shape=self.inputShape)

        # Input for the last 59 features
        morletFeatures = Input(shape=(self.numAdditionalFeatures,))

        # Block 1 (Convolutional and pooling layers for main data)
        block1 = Conv2D(64, kernel_size=(8, 8), activation='relu', padding='same')(inputMain)
        block1 = Conv2D(64, kernel_size=(4, 4), activation='relu', padding='same')(block1)
        block1 = DepthwiseConv2D(kernel_size=(12, 4), activation='relu', padding='same')(block1)
        block1 = BatchNormalization()(block1)
        block1 = AveragePooling2D(pool_size=(4, 4))(block1)
        block1 = SeparableConv2D(64, kernel_size=(8, 8), activation='relu', padding='same')(block1)
        block1 = BatchNormalization()(block1)
        block1 = AveragePooling2D(pool_size=(8, 8))(block1)
        block1 = Flatten()(block1)

        # Dense 2: Send Block 1 output to the first dense layer
        dense2 = Dense(1024, activation='relu')(block1)

        # Dense 1: Send the last 59 features to another dense layer
        dense1 = Dense(1024, activation='relu')(morletFeatures)

        # Concatenate the two dense outputs
        concatenated = Concatenate()([dense1, dense2])

        output = Dense(self.numClasses, activation='softmax')(concatenated)

        # Define and compile the model
        model = Model(inputs=[inputMain, morletFeatures], outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model




class DeepConvNet:
    def __init__(self, 
                 numClasses, chans=124, 
                 samples=1000, dropoutRate=0.5
        ):
        print_with_icon('Building DeepConvNet Architecture', ':computer::zap:')
        self.nbClasses = numClasses
        self.chans = chans
        self.samples = samples
        self.dropoutRate = dropoutRate

    def buildModel(self):
        inputMain = Input((self.chans, self.samples, 1))
        block1 = Conv2D(25, (1, 5), input_shape=(self.chans, self.samples, 1), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(inputMain)
        block1 = Conv2D(25, (self.chans, 1), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
        block1 = Activation('elu')(block1)
        block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
        block1 = Dropout(self.dropoutRate)(block1)

        block2 = Conv2D(50, (1, 5), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block2 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block2)
        block2 = Activation('elu')(block2)
        block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
        block2 = Dropout(self.dropoutRate)(block2)

        block3 = Conv2D(100, (1, 5), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
        block3 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block3)
        block3 = Activation('elu')(block3)
        block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
        block3 = Dropout(self.dropoutRate)(block3)

        block4 = Conv2D(200, (1, 5), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
        block4 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block4)
        block4 = Activation('elu')(block4)
        block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
        block4 = Dropout(self.dropoutRate)(block4)

        flatten = Flatten()(block4)
        dense = Dense(self.nbClasses, kernel_constraint=max_norm(0.5))(flatten)
        softmax = Activation('softmax')(dense)

        return Model(inputs=inputMain, outputs=softmax)

class EegNetSsvepN:
    def __init__(self, numClasses=12, numChannels=124, 
                 samples=1000, dropoutRate=0.5, kernLength=256, 
                 F1=96, D=1, F2=96, dropoutType='Dropout'
        ):
        print_with_icon('Building EEGNetSSVEPN Architecture', ':brain::chart_increasing:')
        self.numClasses = numClasses
        self.numChannels = numChannels
        self.numTimepoints = samples
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutType = dropoutType

    def buildModel(self):
        if self.dropoutType == 'SpatialDropout2D':
            dropoutLayer = SpatialDropout2D
        elif self.dropoutType == 'Dropout':
            dropoutLayer = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.')
        
        input1 = Input(shape=(self.numChannels, self.numTimepoints, 1))

        block1 = Conv2D(self.F1, (1, self.kernLength), padding='same', input_shape=(self.numChannels, self.numTimepoints, 1), use_bias=False)(input1)
        block1 = BatchNormalization()(block1)
        block1 = DepthwiseConv2D((self.numChannels, 1), use_bias=False, depth_multiplier=self.D, depthwise_constraint=max_norm(1.))(block1)
        block1 = BatchNormalization()(block1)
        block1 = Activation('elu')(block1)
        block1 = AveragePooling2D((1, 4))(block1)
        block1 = dropoutLayer(self.dropoutRate)(block1)
        
        block2 = SeparableConv2D(self.F2, (1, 16), use_bias=False, padding='same')(block1)
        block2 = BatchNormalization()(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((1, 8))(block2)
        block2 = dropoutLayer(self.dropoutRate)(block2)
        
        flatten = Flatten(name='flatten')(block2)
        dense = Dense(self.numClasses, name='dense')(flatten)
        softmax = Activation('softmax', name='softmax')(dense)
        
        return Model(inputs=input1, outputs=softmax)

class EEGNet:
    def __init__(self, numClasses, chans=124, samples=1000, 
                 dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=16, normRate=0.25, dropoutType='Dropout'):
        print_with_icon('Building EEGNet Architecture', ':brain::electric_plug:')
        self.nbClasses = numClasses
        self.chans = chans
        self.samples = samples
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.normRate = normRate
        self.dropoutType = dropoutType

        if self.dropoutType == 'SpatialDropout2D':
            self.dropoutType = SpatialDropout2D
        elif self.dropoutType == 'Dropout':
            self.dropoutType = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.')

    def buildModel(self):
        input1 = Input(shape=(self.chans, self.samples, 1))

        # Block 1
        block1 = Conv2D(self.F1, (1, self.kernLength), padding='same', use_bias=False)(input1)
        block1 = BatchNormalization()(block1)
        block1 = DepthwiseConv2D((self.chans, 1), use_bias=False, depth_multiplier=self.D, depthwise_constraint=max_norm(1.))(block1)
        block1 = BatchNormalization()(block1)
        block1 = Activation('elu')(block1)
        block1 = AveragePooling2D((1, 4))(block1)
        block1 = self.dropoutType(self.dropoutRate)(block1)

        # Block 2
        block2 = SeparableConv2D(self.F2, (1, 16), use_bias=False, padding='same')(block1)
        block2 = BatchNormalization()(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((1, 8))(block2)
        block2 = self.dropoutType(self.dropoutRate)(block2)

        # Flatten and Dense layers
        flatten = Flatten(name='flatten')(block2)
        dense = Dense(self.nbClasses, name='dense', kernel_constraint=max_norm(self.normRate))(flatten)
        softmax = Activation('softmax', name='softmax')(dense)

        return Model(inputs=input1, outputs=softmax)

class XGBoostModel:
    def __init__(self, numClasses, taskType):
        print_with_icon('XGB Model', ':deciduous_tree::rocket:')
        self.name = "XGB"
        self.destinationDir = Path(config.trainedModelsDir, taskType)
        self.numClasses = numClasses
        self.model = None
        self.bestModel = None
        self.bestAccuracy = 0
        self.report = None
        self.bestModelName = None

        
    def hyperParameterTunning(self, xTrain, yTrain, xTest, yTest):
        xTrain = xTrain.reshape(xTrain.shape[0], -1)
        xTest = xTest.reshape(xTest.shape[0], -1)
        
        print(f'xTrain:{xTrain.shape}, xTest:{xTest.shape}')
        
        model = xgb.XGBClassifier(
            objective='multi:softmax', 
            num_class=self.numClasses, 
            seed=42,
            nthread=config.numJobs
        )
        params = {
            'maxDepth': [10, 20, 30],
            'nEstimators': [50, 100, 200]
        }
        for maxDepth in params['maxDepth']:
            for nEstimators in params['nEstimators']:
                modelName = f'max_depth: {maxDepth} nEstimators:{nEstimators}'
                print(modelName)
                          
                model.set_params(
                    max_depth=maxDepth, 
                    n_estimators=nEstimators
                )
                model.fit(xTrain, yTrain)
                    
                yPred = model.predict(xTest)
                accuracy = accuracy_score(yTest, yPred)
                if accuracy > self.bestAccuracy:
                    self.bestAccuracy = accuracy
                    self.bestModel = model
                    self.report = classification_report(yTest, yPred, output_dict=True)
                    self.bestModelName = model
        os.makedirs(self.destinationDir, exist_ok=True)
        modelNameWithPath = Path(self.destinationDir, f"{self.name}.pkl")
        saveTensorFlowModel(model, modelNameWithPath, "pickle")
        bestReport = self.report
        bestReport = pd.DataFrame(bestReport).T
        bestReport.to_csv(Path(self.destinationDir, f'{self.name}.csv'))
        self.bestModelName = modelName

class SVCModel:
    def __init__(self, numClasses, taskType):
        print_with_icon('SVC Model', ':straight_ruler::chart_with_upwards_trend:')
        self.name = "SVC"
        self.destinationDir = Path(config.trainedModelsDir, taskType)
        self.numClasses = numClasses
        self.model = None
        self.bestModel = None
        self.bestAccuracy = 0
        self.report = None
        self.bestModelName = None

    def hyperParameterTunning(self, xTrain, yTrain, xTest, yTest):
        xTrain = xTrain.reshape(xTrain.shape[0], -1)
        xTest = xTest.reshape(xTest.shape[0], -1)
        print(f'xTrain:{xTrain.shape}, xTest:{xTest.shape}')
        paramGrid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.01, 0.1, 1],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
        }
        for c in paramGrid['C']:
            for gamma in paramGrid['gamma']:
                for kernel in paramGrid['kernel']:                        
                        modelName = f'C: {c}_ gamme: {gamma}_kernal: {kernel}'
                        print('fitting', modelName)       
                model = SVC(
                        C = c,
                        gamma=gamma,
                        kernel=kernel,
                )
                model.fit(xTrain, yTrain)
                    
                yPred = model.predict(xTest)
                accuracy = accuracy_score(yTest, yPred)
                if accuracy > self.bestAccuracy:
                    self.bestAccuracy = accuracy
                    self.bestModel = self.model
                    self.report = classification_report(yTest, yPred, output_dict=True)
                    self.bestModelName = modelName
        os.makedirs(self.destinationDir, exist_ok=True)
        modelNameWithPath = Path(self.destinationDir, f"{self.name}.pkl")
        saveTensorFlowModel(model, modelNameWithPath, "pickle")
        bestReport = self.report
        bestReport = pd.DataFrame(bestReport).T
        bestReport.to_csv(Path(self.destinationDir, f'{self.name}.csv'))
        self.bestModelName = modelName

class RandomForestModel:
    def __init__(self, numClasses, taskType):
        print_with_icon('Random Forest Model', ':evergreen_tree::deciduous_tree::palm_tree:')
        self.name = "RF"
        self.destinationDir = Path(config.trainedModelsDir, taskType)
        self.numClasses = numClasses
        self.model = None
        self.bestModel = None
        self.bestAccuracy = 0
        self.report = None
        self.bestModelName = None
    

    def hyperParameterTunning(self,xTrain, yTrain, xTest, yTest):
        xTrain = xTrain.reshape(xTrain.shape[0], -1)
        xTest = xTest.reshape(xTest.shape[0], -1)
        
        paramGrid = {
            'n_estimators': [20, 30, 50, 100, 500, 1000],
            'max_depth': [10, 20, 30],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [10, 15, 20, 25, 30]
        }
        for nEstimators in paramGrid['n_estimators']:
            for maxDepth in paramGrid['max_depth']:
                for maxFeatures in paramGrid['max_features']:
                    for minSamplesSplit in paramGrid['min_samples_split']:
                        modelName = f'{nEstimators}_{maxDepth}_{maxFeatures}_{minSamplesSplit}'
                        print(f'Fitting {modelName}')
                        model = RandomForestClassifier(
                            n_estimators=nEstimators,
                            max_depth=maxDepth,
                            min_samples_split=minSamplesSplit,
                            max_features=maxFeatures,
                            n_jobs=config.numJobs
                        )
                        model.fit(xTrain, yTrain)
                        yPred = model.predict(xTest)
                        
                        accuracy = accuracy_score(yTest, yPred)
                        if accuracy > self.bestAccuracy:
                            self.bestAccuracy = accuracy
                            self.bestModel = self.model
                            self.report = classification_report(yTest, yPred, output_dict=True)
                            self.bestModelName = modelName
                    
                
            
            
                
        os.makedirs(self.destinationDir, exist_ok=True)
        modelNameWithPath = Path(self.destinationDir, f"{self.name}.pkl")
        saveTensorFlowModel(model, modelNameWithPath, "pickle")

        bestReport = self.report
        bestReport = pd.DataFrame(bestReport).T
        bestReport.to_csv(Path(self.destinationDir, f'{self.name}.csv'))

