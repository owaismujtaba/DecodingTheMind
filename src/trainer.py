from sklearn.metrics import classification_report

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import pdb
from pathlib import Path
from src import config
from src.utils import save_model
import pandas as pd
import os
import tensorflow as tf
from termcolor import colored
if config.device == 'CPU':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class EEGModelTrainer:
    def __init__(self, model, modelName, taskType, numEpochs=config.epochs, batchSize=config.batchSize):
        self.model = model
        self.modelName = modelName
        self.taskType = taskType
        self.destinationDir = Path(config.trainedModelsDir, self.taskType)
        self.numEpochs = numEpochs
        self.batchSize = batchSize
        self.report=None
        self.history = None
        os.makedirs(self.destinationDir, exist_ok=True)

    def train(self, xTrain, yTrain, xVal, yVal):
        
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        )
        
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("\033[94m Model Summary:\033[0m")  
        print(self.model.summary())
        #print(colored(f' Xtrain: {xTrain.shape}, xTest: {xVal.shape}', 'magenta'))
        print("\033[92m Starting model training...\033[0m")
        if self.modelName == 'CNNSegmented':
            self.history = self.model.fit(xTrain, yTrain, 
                        epochs=self.numEpochs, 
                        batch_size=self.batchSize, 
                        validation_data=(xVal, yVal),
                        callbacks=[early_stopping]
                    )
        else:

            self.history = self.model.fit(xTrain, yTrain, 
                            epochs=self.numEpochs, 
                            batch_size=self.batchSize, 
                            validation_data=(xVal, yVal),
                            callbacks=[early_stopping]
                        )
        modelNameWithPath = Path(self.destinationDir, self.modelName)
        save_model(self.model, f"{modelNameWithPath}.h5")
        print(f"\033[92m Model saved to {modelNameWithPath}.h5\033[0m")
        
        self.history = pd.DataFrame(self.history.history)
        self.history.to_csv(f"{modelNameWithPath}.csv")
        print(f"\033[92m Training history saved to {modelNameWithPath}.csv\033[0m")
        
        self.performance(xVal, yVal)
    
    def performance(self, xTest, yTest):
        print("\033[93m Evaluating model performance...\033[0m")
        with tf.device('/CPU:0'):
            predictions = self.model.predict(xTest)
            predictions = np.argmax(predictions, axis=1)
            trueLabels = np.array(yTest)
            
            self.report = classification_report(trueLabels, predictions, output_dict=True)
            print("\033[94m Classification Report:\033[0m")
            print(self.report)
