#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 06:24:51 2021

@author: july
"""

## MP reference code
# scipy-1.4.1
# pandas==0.25.3
# scikit-learn 0.24.1

import os
import sys
import glob
import itertools
import random

from IPython.display import Image

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.colors import ListedColormap
from scipy.stats import multivariate_normal

import numpy as np
import pandas as pd
from scipy.stats import beta

random.seed(1234)

%matplotlib inline


### SAMPLE MONDRIAN PROCESS ###
def draw_Mondrian(theta_space, budget=5):
    return draw_Mondrian_at_t(theta_space, 0, budget)
    
def draw_Mondrian_at_t(theta_space, t, budget):
    dists = theta_space[:,1] - theta_space[:,0]
    lin_dim = np.sum(dists)
    T = np.random.exponential(scale=1./lin_dim)
    
    if t+T > budget: 
        return (theta_space, None, None)
    
    d = np.argmax(np.random.multinomial(n=1, pvals=dists/lin_dim))
    x = np.random.uniform(low=theta_space[d,0], high=theta_space[d,1])
    
    theta_left = np.copy(theta_space)
    theta_left[d][1] = x 
    M_left = draw_Mondrian_at_t(theta_left, t+T, budget)
    
    theta_right = np.copy(theta_space)
    theta_right[d][0] = x 
    M_right = draw_Mondrian_at_t(theta_right, t+T, budget)
    
    return (theta_space, M_left, M_right)

def comp_log_p_sample(theta_space, data):
    if theta_space[1] == None and theta_space[2] == None:
        if data.shape[0] == 0:
            return 0
        else:
            mu = np.mean(data, axis = 0)
            residual = data - mu
            cov = np.dot(residual.T , residual) / data.shape[0] + np.identity(data.shape[1])*0.001
            return np.log(multivariate_normal.pdf(data, mean=mu, cov=cov)).sum()
    
    # find the dimension and location of first cut
    root_rec = theta_space[0]
    left_rec = theta_space[1][0]
    
    for _ in range(root_rec.shape[0]):
        if root_rec[_,1] != left_rec[_,1]:
            break
    
    dim, pos = _, left_rec[_,1]
    idx_left = data[:,dim] < pos
    idx_right = data[:,dim] >= pos
    log_len_left =  np.log(pos - root_rec[dim,0])
    log_len_right = np.log(root_rec[dim,1] - pos)
    return comp_log_p_sample(theta_space[1], data[idx_left]) + comp_log_p_sample(theta_space[2], data[idx_right])
                                                                             
### VISUALIZE 2D MONDRIAN PROCESS ###
def print_partitions(p, trans_level=1., color='k'):
    if not p[1] and not p[2]: 
        plt.plot([p[0][0,0], p[0][0,0]], [p[0][1,0], p[0][1,1]], color+'-', linewidth=5, alpha=trans_level)
        plt.plot([p[0][0,1], p[0][0,1]], [p[0][1,0], p[0][1,1]], color+'-', linewidth=5, alpha=trans_level)
        plt.plot([p[0][0,0], p[0][0,1]], [p[0][1,0], p[0][1,0]], color+'-', linewidth=5, alpha=trans_level)
        plt.plot([p[0][0,0], p[0][0,1]], [p[0][1,1], p[0][1,1]], color+'-', linewidth=5, alpha=trans_level)
    
    else:
        print_partitions(p[1], trans_level, color)
        print_partitions(p[2], trans_level, color)
        
        
### VISUALIZE 2D POSTERIOR WITH DATA###
def print_posterior(data, samples, trans_level=.05, color='k'):

    plt.figure()
    plt.scatter(data[:,0], data[:,1], c='k', edgecolors='k', s=5, alpha=.5)

    #print all samples
    for sample in samples:
        print_partitions(sample, trans_level, color)
        
        
def print_tree_at_leaf(mp_tree, table):

    if mp_tree[1] == None and mp_tree[2] == None: 
        print(table.shape)
        return 1
    
    
    # find the dimension and location of first cut
    root_rec = mp_tree[0]
    left_rec = mp_tree[1][0]
    
    for _ in range(root_rec.shape[0]):
        if root_rec[_,1] != left_rec[_,1]:
            break
    d, pos = _, left_rec[_,1]
    
    cut_type = ' '.join([str(int(x)) for x in sorted(set(table[table.columns[d]]))]) 
    
    if cut_type in {"-1 0 1", '-1 1'}: 
        idx_table_left = table[table.columns[d]] != 1
        table_left = table.loc[idx_table_left]

        idx_table_right = table[table.columns[d]] != -1
        table_right = table.loc[idx_table_right]
    
    if cut_type == '-1 0':
        idx_table_left = table[table.columns[d]] == -1
        table_left = table.loc[idx_table_left]

        idx_table_right = table[table.columns[d]] == 0
        table_right = table.loc[idx_table_right]
        

    if cut_type == '0 1':
        idx_table_left = table[table.columns[d]] == 0
        table_left = table.loc[idx_table_left]

        idx_table_right = table[table.columns[d]] == 1
        table_right = table.loc[idx_table_right]
    
    return print_tree_at_leaf(mp_tree[1], table_left) + print_tree_at_leaf(mp_tree[2], table_right)                                                                                
                                                                                                                                                              
### SAMPLE MONDRIAN PROCESS WITH PRIOR INFORMATION ###
def draw_informed_Mondrian(theta_space, table, budget=5):
    
    # INFORMATIVE PRIORS
    upper_cut = (5., 2.)
    lower_cut = (2., 5.)
    middle_cut = (5., 5.)
    neutral_cut = (2., 2.)
    priors_dict = { '-1':lower_cut, '0':neutral_cut, '1':upper_cut, 
                   '-1 0':lower_cut, '-1 1':middle_cut, '0 1':upper_cut,
                   '-1 0 1': middle_cut, '': neutral_cut
                  }
    
    cut_history = [1] * theta_space.shape[0]
    
    return draw_informed_Mondrian_at_t(theta_space, table, priors_dict, cut_history)
    

def draw_informed_Mondrian_at_t(theta_space, table, priors_dict, cut_history):    
    
    if sum(cut_history) == 0 or table.shape[0] == 1:
        return (theta_space, None, None)

    
    types_str = [' '.join([str(int(x)) for x in sorted(set(table[table.columns[d]]))]) 
                 for d in range(table.shape[1])]
    
    if set([types_str[d] for d in range(table.shape[1]) if cut_history[d] == 1]).issubset({'0','1','-1'}):
        return (theta_space, None, None)

    
    low, medium, high, very_high = 0, 1, 100, 1000
    priority_dict = {'-1': low , '0': low, '1': low, 
                   '-1 0': medium, '0 1': medium,
                   '-1 0 1': high, '-1 1':very_high
    }    
        
    types = np.array([priority_dict[_] for _ in types_str])
    

    dists = (theta_space[:,1] - theta_space[:,0])* types    
    lin_dim = np.sum(dists)
    # draw dimension to cut
    dim_probs = ((dists/lin_dim) * np.array(cut_history)) 
    dim_probs /= np.sum(dim_probs)
    d = np.argmax(np.random.multinomial(n=1, pvals=dim_probs))
    cut_history[d] = 0

    prior_type_str = ' '.join([str(int(x)) for x in sorted(set(table[table.columns[d]]))])
    prior_params = priors_dict[prior_type_str]
    
    # make scaled cut
    x = (theta_space[d,1] - theta_space[d,0]) * np.random.beta(prior_params[0], prior_params[1]) + theta_space[d,0]
    
    cut_type = types_str[d]
    
    if cut_type in {"-1 0 1", '-1 1'}: 
        idx_table_left = table[table.columns[d]] != 1
        table_left = table.loc[idx_table_left]

        idx_table_right = table[table.columns[d]] != -1
        table_right = table.loc[idx_table_right]
    
    if cut_type == '-1 0':
        idx_table_left = table[table.columns[d]] == -1
        table_left = table.loc[idx_table_left]

        idx_table_right = table[table.columns[d]] == 0
        table_right = table.loc[idx_table_right]
        

    if cut_type == '0 1':
        idx_table_left = table[table.columns[d]] == 0
        table_left = table.loc[idx_table_left]

        idx_table_right = table[table.columns[d]] == 1
        table_right = table.loc[idx_table_right]

    
    # make lower partition
    theta_left = np.copy(theta_space)
    theta_left[d][1] = x
    M_left = draw_informed_Mondrian_at_t(theta_left, table_left, priors_dict, list(cut_history))
    
    # make upper partition
    theta_right = np.copy(theta_space)
    theta_right[d][0] = x 
    M_right = draw_informed_Mondrian_at_t(theta_right, table_right, priors_dict,list(cut_history))
    
    return (theta_space, M_left, M_right)

def Mondrian_Gaussian_perturbation(theta_space, old_sample, stepsize):
    """
    Input: 
    theta_space: a rectangle
    old_sample: partioned theta_space of a mondrian process
    stepsize: gaussian std
    """
    if old_sample[1] == None and old_sample[2] == None:
        return (theta_space, None, None)
    
    # find the dimension and location of first cut in the old_sample
    for _ in range(old_sample[0].shape[0]):
        if old_sample[0][_,1] > old_sample[1][0][_,1]:
            break    
    dim, pos = _, old_sample[1][0][_,1]
    # propose position of new cut
    good_propose = False
    while good_propose == False:
        new_pos = pos + np.random.normal(0,(old_sample[0][dim,1] - old_sample[0][dim,0])*stepsize,1)[0]
        if new_pos < theta_space[dim,1] and new_pos > theta_space[dim,0]:
            good_propose = True
    
    theta_left = np.copy(theta_space)
    theta_left[dim,1] = new_pos
    theta_right = np.copy(theta_space)
    theta_right[dim,0] = new_pos
    
    new_M_left= Mondrian_Gaussian_perturbation(theta_left, old_sample[1], stepsize)
    new_M_right = Mondrian_Gaussian_perturbation(theta_right, old_sample[2], stepsize)
    
    return (theta_space, new_M_left, new_M_right)

def comp_log_p_prior(theta_space, table, cut_history):
    """
    This function returns prior probability of a Mondrian process theta_space
    """
    if theta_space[1] == None and theta_space[2] == None:
        return 0
    
    log_prior = 0
    

    # INFORMATIVE PRIORS
    upper_cut = (5., 2.)
    lower_cut = (2., 5.)
    middle_cut = (5., 5.)
    neutral_cut = (2., 2.)
    priors_dict = { '-1':lower_cut, '0':neutral_cut, '1':upper_cut, 
                   '-1 0':lower_cut, '-1 1':middle_cut, '0 1':upper_cut,
                   '-1 0 1': middle_cut, '': neutral_cut
                  }
    
    
    # find the dimension and location of first cut
    root_rec = theta_space[0]
    left_rec = theta_space[1][0]
    
    for _ in range(root_rec.shape[0]):
        if root_rec[_,1] != left_rec[_,1]:
            break
    dim  = _
    beta_pos =  (left_rec[_,1] - left_rec[dim,0])/(root_rec[dim,1] - root_rec[dim, 0])
    
    prior_params = priors_dict[' '.join([str(int(x)) \
                                         for x in sorted(set(table[table.columns[dim]]))])]

    # compute the log likelihood of the first cut
    types_str = [' '.join([str(int(x)) for x in sorted(set(table[table.columns[d]]))]) 
                 for d in range(table.shape[1])]
    
    low_priority, medium_priority, high_priority, very_high_priority = 0, 1, 100, 1000
    priority_dict = {'-1': low_priority , '0': low_priority, '1': low_priority, 
                   '-1 0': medium_priority, '0 1': medium_priority,
                   '-1 0 1': high_priority, '-1 1':very_high_priority
    }    
        
    types = np.array([priority_dict[_] for _ in types_str])
    dists = (root_rec[:,1] - root_rec[:,0])* types    
    lin_dim = np.sum(dists)
    
    # probability of dim
    dim_probs = ((dists/lin_dim) * np.array(cut_history)) 
    dim_probs /= np.sum(dim_probs)
    log_prior  += np.log(dim_probs[dim])
    
    
    # probability of pos
    log_prior += np.log(beta.pdf(beta_pos, prior_params[0], prior_params[1]))
    
    # split the table
    cut_history[dim] = 0
    cut_type = types_str[dim]
    
    if cut_type in {"-1 0 1", '-1 1'}: 
        idx_table_left = table[table.columns[dim]] != 1
        table_left = table.loc[idx_table_left]

        idx_table_right = table[table.columns[dim]] != -1
        table_right = table.loc[idx_table_right]
    
    if cut_type == '-1 0':
        idx_table_left = table[table.columns[dim]] == -1
        table_left = table.loc[idx_table_left]

        idx_table_right = table[table.columns[dim]] == 0
        table_right = table.loc[idx_table_right]
        

    if cut_type == '0 1':
        idx_table_left = table[table.columns[dim]] == 0
        table_left = table.loc[idx_table_left]

        idx_table_right = table[table.columns[dim]] == 1
        table_right = table.loc[idx_table_right]
        
        
        
    return log_prior + comp_log_p_prior(theta_space[1], table_left, list(cut_history)) \
        + comp_log_p_prior(theta_space[2], table_right, list(cut_history))                                                                              

def classify_cells(data, mp_tree, table, cell_type_name2idx):
    Y = np.array([1]*data.shape[0])
    
    if data.shape[0] == 0:
        return Y
    
    if mp_tree[1] == None and mp_tree[2] == None:
        if table.shape[0] > 1:
#            print "more than one clusters, number of data points:", data.shape[0]
            labels = [cell_type_name2idx[table.index[_]] for _ in range(table.shape[0])]
            return np.array(np.random.choice(labels, data.shape[0],replace = True))
        else:
            return Y * cell_type_name2idx[table.index[0]]
    
    
    # find the dimension and location of first cut
    root_rec = mp_tree[0]
    left_rec = mp_tree[1][0]
    
    for _ in range(root_rec.shape[0]):
        if root_rec[_,1] != left_rec[_,1]:
            break
    dim, pos = _, left_rec[_,1]
    
    # find labels that match dim info from table
    idx_table_left = table[table.columns[dim]] != 1
    table_left = table.loc[idx_table_left]

    idx_table_right = table[table.columns[dim]] != -1
    table_right = table.loc[idx_table_right]
    
    # find data INDICIES that go high / low on cut position in dimension dim
    idx_left = data[:,dim]  < pos
    idx_right = data[:,dim]  >= pos
    
    Y[idx_left] = classify_cells(data[idx_left],mp_tree[1],table_left, cell_type_name2idx)
    Y[idx_right] = classify_cells(data[idx_right],mp_tree[2],table_right, cell_type_name2idx)

    return Y

# load AML data and table
##### X: np.array, flow cytometry data, arcsin transformed
##### T: table of expert knowledge

np.random.seed(1234)
PATH = 'auto-gating/reference/mp_ref/'
### LOAD DATA ###
path = PATH + 'data/'
df = pd.read_csv( path + 'data2.csv', sep=',', header = 0, engine='python')
table = pd.read_csv(path + 'data_table2.csv', sep=',', header=0, index_col=0)

### PROCESS: discard ungated events ###
df = df.drop(['id'], axis = 1)
channels = [item[:item.find('(')] for item in df.columns[:-1]]
df.columns = channels + ['cell_type']

table = table.fillna(0)
X = df[channels].values
table_headers = list(table)

data = X
theta_space = np.array([[data[:,d].min(), data[:,d].max()] for d in range(data.shape[1])])

idx2ct = [key for idx, key in enumerate(table.index)]
idx2ct.append('unknown')
cell_type_name2idx = {x:i for i,x in enumerate(table.index)}
cell_type_name2idx['unknown'] = len(table.index)
Y = np.array([cell_type_name2idx[_] for _ in df.cell_type])

from sklearn.utils import shuffle
N = data.shape[0]
#new_df = shuffle(df)[:N]
new_df=df
X_subset = new_df[channels].values
data_subset = np.arcsinh((X_subset-1.)/5.)
Y_subset = np.array([cell_type_name2idx[_] for _ in new_df.cell_type])
N, d = data_subset.shape

temp_headers = list(table)
table.columns = temp_headers
emp_bounds = np.array([(data_subset[:,i].min(), data_subset[:,i].max()) for i in range(d)])

n_mcmc_chain = 5
n_mcmc_sample = 2000
mcmc_gaussin_std = 0.1 # tune step size s.t. acceptance rate ~50%
accepts = [[] for _ in range(n_mcmc_chain)]
rejects = [[] for _ in range(n_mcmc_chain)]
logl_accepted_trace = [[] for _ in range(n_mcmc_chain)]
logl_complete_trace = [[] for _ in range(n_mcmc_chain)]
Y_predict_accepted_trace = [[] for _ in range(n_mcmc_chain)]
accuracy_accepted_trace = [[] for _ in range(n_mcmc_chain)]

for chain in range(n_mcmc_chain):
    
    print("Drawing Chain %d ..." % chain)
    
    sample = draw_informed_Mondrian(emp_bounds, table)
    log_p_sample = comp_log_p_sample(sample, data_subset)
    
    accepts[chain].append(sample)
    logl_accepted_trace[chain].append(log_p_sample)
    logl_complete_trace[chain].append(log_p_sample)
    
    Y_predict = classify_cells(data_subset, sample, table, cell_type_name2idx)
    accuracy = sum(Y_subset == Y_predict)*1.0/ data_subset.shape[0]
    accuracy_accepted_trace[chain].append(accuracy)
    Y_predict_accepted_trace[chain].append(Y_predict)
    
    for idx in range(n_mcmc_sample):
        
        new_sample = Mondrian_Gaussian_perturbation(emp_bounds,sample, mcmc_gaussin_std)
        new_log_p_sample = comp_log_p_sample(new_sample, data_subset) 
        logl_complete_trace[chain].append(log_p_sample)
        if new_log_p_sample <  log_p_sample and \
            np.log(np.random.uniform(low=0, high=1.)) > (new_log_p_sample - log_p_sample):
                rejects[chain].append(new_sample)
        
        else:
            if new_log_p_sample <  log_p_sample:
                print("accepted some bad samples")
                
            sample = new_sample
            log_p_sample = new_log_p_sample
            accepts[chain].append(sample)
            logl_accepted_trace[chain].append(log_p_sample)
        
            Y_predict = classify_cells(data_subset, sample, table, cell_type_name2idx)
            accuracy = sum(Y_subset == Y_predict)*1.0/ data_subset.shape[0]
            accuracy_accepted_trace[chain].append(accuracy)
            Y_predict_accepted_trace[chain].append(Y_predict)
        
        if (idx+1) % 500 == 0:
            print("Iteration %d, cummulative accepted sample size is %d" %(idx+1, len(accepts[chain])))
            
    
    if (chain + 1) % 10 == 0:
        # prediction and visualization
        Y_predict = classify_cells(data_subset, accepts[chain][-1], table, cell_type_name2idx)
        accuracy = sum(Y_subset == Y_predict)*1.0/ data_subset.shape[0]
        print("Chain % d accuracy on subset data: %.3f" % (chain+1,accuracy))         
            
            
print("Total number of accepted samples: %d" %(sum([len(accepts[chain]) for chain in range(n_mcmc_chain)])))

Y_predict = classify_cells(data_subset, accepts[chain][-1], table, cell_type_name2idx)

accuracy = sum(Y_subset == Y_predict)*1.0/ data_subset.shape[0]

