#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Step 1
import sys
sys.path.insert(0,"/Users/downloads/ACDC")
from random_walk_classifier import *
from cell_type_annotation import *


## Step 2
import pandas as pd
import numpy as np
from collections import Counter
# data with cell type label
df = pd.read_csv('/Users/auto-gating/data.csv', sep=',', header = 0)
channels = [item for item in df.columns[:-1]] # markers
df.columns = channels + ['cell_type']
# marker for each cell type
table = pd.read_csv('/Users/auto-gating/marker_table.csv', sep=',', header=0, index_col=0)
table = table.fillna(0)
cts, channels = get_label(table)
X0 = np.arcsinh((df[channels].values - 1.0)/5.0)

## Step 3
idx2ct = [key for idx, key in enumerate(table.index)]
idx2ct.append('unknown')
ct2idx = {key:idx for idx, key in enumerate(table.index)}
ct2idx['unknown'] = len(table.index)
ct_score = np.abs(table.as_matrix()).sum(axis = 1)
## compute manual gated label (true label)
y0 = np.zeros(df.cell_type.shape)
for i, ct in enumerate(df.cell_type):
    if ct in ct2idx:
        y0[i] = ct2idx[ct]
    else:
        y0[i] = ct2idx['unknown']  
        
## Step 4
from sklearn.metrics import accuracy_score, confusion_matrix
import phenograph
from sklearn.model_selection import StratifiedKFold
import pickle
n_neighbor = 10
thres = 0.5       

        
## Step 5
result = []
score_final = []
process_time = []
c = 0
X = X0.copy()
y_true = y0.copy()

mk_model =  compute_marker_model(pd.DataFrame(X, columns = channels), table, 0.0)

## compute posterior probs
score = get_score_mat(X, [], table, [], mk_model)
score = np.concatenate([score, 1.0 - score.max(axis = 1)[:, np.newaxis]], axis = 1)    
    
## get indices     
ct_index = get_unique_index(X, score, table, thres)
    
## baseline - classify events    
y_pred_index = np.argmax(score, axis = 1)

## running ACDC
res_c = get_landmarks(X, score, ct_index, idx2ct, phenograph, thres)

landmark_mat, landmark_label = output_feature_matrix(res_c, [idx2ct[i] for i in range(len(idx2ct))]) 

landmark_label = np.array(landmark_label)

lp, y_pred = rm_classify(X, landmark_mat, landmark_label, n_neighbor)

