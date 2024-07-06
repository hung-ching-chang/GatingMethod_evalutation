#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import os.path
import sys
sys.path.insert(0,"auto-gating/deepcytof_ref/deepcytof-master/src/Util")
sys.path.insert(0,"auto-gating/deepcytof_ref/deepcytof-master/src")
import Util

from Util import DataHandler as dh
from Util import FileIO as io
from Util import feedforwadClassifier as net


dataSet = ['data']
numSample = [1]
relevantMarkers = [21]
hiddenLayersSizes = [12, 6, 3]
activation = 'softplus'
l2_penalty = 1e-4
choice = 0

# Generate the path of the chosen data set.
dataPath = os.path.join('auto-gating/deepcytof_ref/', dataSet[choice])

# Generate the output table.
acc = np.zeros(numSample[choice])
F1 = np.zeros(numSample[choice])

import numpy.matlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('Data set name: ', dataSet[choice])
for i in range(numSample[choice]):
    # Load sample.
    print('Load sample ', str(i+1))
    sample = dh.loadDeepCyTOFData(dataPath, i + 1,
                                  range(relevantMarkers[choice]), 'CSV',
                                  skip_header = 1) 
    # Pre-process sample.
    print('Pre-process sample ', str(i+1))
    sample = dh.preProcessSamplesCyTOFData(sample)
    sample, preprocessor = dh.standard_scale(sample, preprocessor = None)
    
    # Split data into training and testing.
    print('Split data into training and testing.')
    trainSample, testSample = dh.splitData(sample, test_size = .5)
    
    # Train a feed-forward neural net classifier on the training data.
    print('Train a feed-forward neural net classifier on the training data.')
    classifier = net.trainClassifier(trainSample, dataSet[choice], i,
                                     hiddenLayersSizes,
                                     activation = activation,
                                     l2_penalty = l2_penalty)
    
    # Run the classifier on the testing data.
    print('Run the classifier on the testing data.')
    acc[i-1], F1[i-1], y_test_pred = net.prediction(testSample, # line 159 of prediction function in feedforwardClassifier.py, I added y_test
                                                                dataSet[choice], i, classifier)





