# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 13:18:34 2022

@author: luis.barreiro
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  matplotlib.colors import LogNorm
import seaborn as sns
from sklearn.model_selection import train_test_split as splitter
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import os 
from sklearn.neural_network import MLPClassifier

#%%
import os
os.chdir("C:/Users/luis.barreiro/Documents/GitHub/Projects_STK4155/Project3")
cwd = os.getcwd()
print(cwd)

#%%
np.random.seed(3)        #create same seed for random number every time

trees=pd.read_csv("input_data\input_trees_v02.csv")     #Load tree data

trees.columns

x=trees[['min', 'max', 'avg', 'std', 'ske', 'kur', 'p05', 'p25','p50', 'p75', 'p90', 'c00', 'int_min', 'int_max', 'int_avg', 'int_std','int_ske', 'int_kur', 'int_p05', 'int_p25', 'int_p50', 'int_p75','int_p90']]
y1=trees['CON_DEC']
y2=trees['Specie']

x_train,x_test,y_train,y_test=splitter(x,y1,test_size=0.3)   #Split datasets into training and testing

#Scale the data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

M = 100   #size of each minibatch
m = int(y_train.shape[0]/M) #number of minibatches
epochs = 100000 #number of epochs 

#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler

#from preprocessing import *

network = MLPRegressor(max_iter=1000)

print("Learning rates to search:")
eta_vals=np.logspace(-8, -5, 4) # might not go this low with eta?
print(eta_vals)

print("Regularization params to search:")
alpha_vals = np.logspace(-3, 1, 5)
print(alpha_vals)

hyperparameters_to_search = {
    "hidden_layer_sizes" : [[5, 10, 20, 50],[0, 5, 10, 20, 50]],
    "alpha" : alpha_vals,
    "learning_rate_init" : eta_vals
}

regression = GridSearchCV(
    network, 
    param_grid=hyperparameters_to_search, 
    scoring="r2", 
    n_jobs=-1, # use all cores in parallel
    refit=True,
    cv=10, # number of folds
    verbose=3
    )

print("Do the search (can take some time!)")
search = regression.fit(x, y1)

print("Best parameters from gridsearch:")
print(search.best_params_)

#print("Best CV R2 score from gridsearch:")
#print(search.best_score_)

#print("R2 score on test data:")
#r2 = search.score(x_test, y_test)
#print("R2 = ", r2)

#with open('datasets/cv_ffnn_results2.pickle','wb') as f:
#    pickle.dump(search.cv_results_, f)