import os
from pathlib import Path

currentDir = os.getcwd()
rawDataDir = Path(currentDir, 'RawData')
dataDir = Path(currentDir, 'Data')

seperator = "\\"
preprocessData = True
precptionImaginationProcessing = True


tmin=-0.5
tmax=1.0
samplingFrequency = 1000
numJobs=20