# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 15:15:30 2022

@author: luis.barreiro
"""

#%%

# Common imports
from IPython.display import Image 
#from pydot import graph_from_dot_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
import scikitplot as skplt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split as splitter
import os


#%%
np.random.seed(3)        #create same seed for random number every time

#For Ridge regression, set up the hyper-parameters to investigate
lmbd = 1e-6

#Number of bootstraps
n_bootstraps = 75

# Generate dataset with n observations
n = 100
x = np.random.uniform(0,1,n)
y = np.random.uniform(0,1,n)

#Define noise
var = 0.01
noise = np.random.normal(0,var,n)

z = FrankeFunction(x,y) + noise 

x = np.array(x).reshape(n,1)
y = np.array(y).reshape(n,1)
x1 = np.hstack((x,y)).reshape(n,2)

#Split train (80%) and test(20%) data before looping on polynomial degree
x_train, x_test, z_train, z_test = train_test_split(x1, z, test_size=0.2)

z_train = z_train.astype('int')
z_test = z_test.astype('int')
    
#Scaling not needed
#%%
#inizializing before looping:
n_hidden_neurons=5
error = np.zeros(n_hidden_neurons)
bias = np.zeros(n_hidden_neurons)
variance = np.zeros(n_hidden_neurons)

z_pred = np.empty((z_test.shape[0],n_bootstraps))

for n in range(1,n_hidden_neurons):
    for i in range(n_bootstraps):
        x_, z_ = resample(x_train,z_train)
      
        X_train = DesignMatrix(x_[:,0],x_[:,1],degree+1)
        X_test = DesignMatrix(x_test[:,0],x_test[:,1],degree+1)
        
        dnn = MLPRegressor(hidden_layer_sizes=n, activation='relu', solver ='lbfgs',alpha=10, learning_rate_init=1.e-5, max_iter=1000)
        dnn.fit(X_train, z_train)
        print("Test set accuracy Neural Network: {:.2f}".format(dnn.score(X_test,z_test)))
        z_pred = dnn.predict(X_test)
    
    error[n] = np.mean((z_test - z_pred)**2)
    bias[n] = np.mean( (z_test - np.mean(z_pred))**2 )
    variance[n] = np.var(z_pred)


plt.plot(range(1,n_hidden_neurons), error[1:], label = 'Error')
plt.plot(range(1,n_hidden_neurons), bias[1:], label = 'Bias')
plt.plot(range(1,n_hidden_neurons), variance[1:], label = 'Variance')
plt.ylabel('Error')
plt.xlabel('Model complexity: number of neurons in one layer')
plt.title("Variance-Bias tradeoff for MLP")
plt.legend()
#plt.savefig("Results/bias_variance_tradeoff/MLP_bias_var_tradeoff.png",dpi=150)
plt.show()


temp=error-(bias+variance)
