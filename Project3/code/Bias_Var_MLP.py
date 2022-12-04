# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 15:15:30 2022

@author: luis.barreiro
"""

#%%
import os
os.chdir("C:/Users/luis.barreiro/Documents/GitHub/Projects_STK4155/Project3")
cwd = os.getcwd()
print(cwd)

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
from Functions import LassoReg, DesignMatrix, FrankeFunction
from sklearn.utils import resample


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
z = z.reshape(z.shape[0],1)

x = np.array(x).reshape(n,1)
y = np.array(y).reshape(n,1)
x1 = np.hstack((x,y)).reshape(n,2)

#Split train (80%) and test(20%) data before looping on polynomial degree
x_train, x_test, z_train, z_test = train_test_split(x1, z, test_size=0.2)

#z_train = z_train.astype('int')
#z_test = z_test.astype('int')
    
#Scaling not needed
#%%
#inizializing before looping:
n_hidden_neurons=50
error = np.zeros(n_hidden_neurons)
bias = np.zeros(n_hidden_neurons)
variance = np.zeros(n_hidden_neurons)

for n in range(n_hidden_neurons):
    z_pred = np.empty((z_test.shape[0],n_bootstraps))
    for i in range(n_bootstraps):
        x_, z_ = resample(x_train,z_train)
             
        dnn = MLPRegressor(hidden_layer_sizes=n+1, activation='relu', solver ='lbfgs',alpha=10, learning_rate_init=1.e-5, max_iter=1000)
        dnn.fit(x_, z_)
        print("Test set accuracy Neural Network: {:.2f}".format(dnn.score(x_test,z_test)))
        z_pred[:, i] = dnn.predict(x_test)
    
    error[n] = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    bias[n] = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance[n] = np.mean( np.var(z_pred, axis=1, keepdims=True) )


plt.plot(range(1,n_hidden_neurons+1), error, label = 'Error')
plt.plot(range(1,n_hidden_neurons+1), bias, label = 'Bias')
plt.plot(range(1,n_hidden_neurons+1), variance, label = 'Variance')
plt.ylabel('Error')
plt.xlabel('Model complexity: number of neurons in one layer')
plt.xticks(np.arange(0, n_hidden_neurons+1, step=5))  # Set label locations.
plt.title("Variance-Bias tradeoff for MLP")
plt.legend()
plt.savefig("Results/bias_variance_tradeoff/MLP_bias_var_tradeoff.png",dpi=150)
plt.show()

#Check that bias+variance=error
temp=error-(bias+variance)
print(temp)

#%%
n_hidden_neurons=[[5],[5,5],[5,5,5],[5,5,5,5],[5,5,5,5,5]]
error = np.zeros(len(n_hidden_neurons))
bias = np.zeros(len(n_hidden_neurons))
variance = np.zeros(len(n_hidden_neurons))

for n in range(len(n_hidden_neurons)):
    z_pred = np.empty((z_test.shape[0],n_bootstraps))
    for i in range(n_bootstraps):
        x_, z_ = resample(x_train,z_train)
             
        dnn = MLPRegressor(hidden_layer_sizes=n_hidden_neurons[n], activation='relu', solver ='lbfgs',alpha=10, learning_rate_init=1.e-5, max_iter=1000)
        dnn.fit(x_, z_)
        print("Test set accuracy Neural Network: {:.2f}".format(dnn.score(x_test,z_test)))
        z_pred[:,i] = dnn.predict(x_test)
    
    error[n] = np.mean((z_test - z_pred)**2)
    bias[n] = np.mean( (z_test - np.mean(z_pred))**2 )
    variance[n] = np.var(z_pred)


plt.plot(range(1,len(n_hidden_neurons)+1), error, label = 'Error')
plt.plot(range(1,len(n_hidden_neurons)+1), bias, label = 'Bias')
plt.plot(range(1,len(n_hidden_neurons)+1), variance, label = 'Variance')
plt.ylabel('Error')
plt.xlabel('Model complexity: number layers')
plt.title("Variance-Bias tradeoff for MLP")
plt.xticks(np.arange(1, len(n_hidden_neurons)+1, step=1))  # Set label locations.
plt.legend()
plt.savefig("Results/bias_variance_tradeoff/MLP_bias_var_tradeoff_layers.png",dpi=150)
plt.show()


#Check that bias+variance=error
temp=error-(bias+variance)
print(temp)



#%%
dnn = MLPRegressor(hidden_layer_sizes=[5], activation='relu', solver ='lbfgs',alpha=10, learning_rate_init=1.e-5, max_iter=1000)
dnn.fit(x_train, z_train)
#print("Test set accuracy Neural Network: {:.2f}".format(dnn.score(x_test,z_test)))
z_pred = dnn.predict(x_test)
error = np.mean((z_test - z_pred)**2)
print("One layer",error)

dnn = MLPRegressor(hidden_layer_sizes=(5,5), activation='relu', solver ='lbfgs',alpha=10, learning_rate_init=1.e-5, max_iter=1000)
dnn.fit(x_train, z_train)
#print("Test set accuracy Neural Network: {:.2f}".format(dnn.score(x_test,z_test)))
z_pred = dnn.predict(x_test)
error = np.mean((z_test - z_pred)**2)
print("Two layers",error)

dnn = MLPRegressor(hidden_layer_sizes=(5,5,5), activation='relu', solver ='lbfgs',alpha=10, learning_rate_init=1.e-5, max_iter=1000)
dnn.fit(x_train, z_train)
#print("Test set accuracy Neural Network: {:.2f}".format(dnn.score(x_test,z_test)))
z_pred = dnn.predict(x_test)
error = np.mean((z_test - z_pred)**2)
print("Three layers",error)

dnn = MLPRegressor(hidden_layer_sizes=(5,5,5,5), activation='relu', solver ='lbfgs',alpha=10, learning_rate_init=1.e-5, max_iter=1000)
dnn.fit(x_train, z_train)
#print("Test set accuracy Neural Network: {:.2f}".format(dnn.score(x_test,z_test)))
z_pred = dnn.predict(x_test)
error = np.mean((z_test - z_pred)**2)
print("Four layers",error)

dnn = MLPRegressor(hidden_layer_sizes=(5,5,5,5,5), activation='relu', solver ='lbfgs',alpha=10, learning_rate_init=1.e-5, max_iter=1000)
dnn.fit(x_train, z_train)
#print("Test set accuracy Neural Network: {:.2f}".format(dnn.score(x_test,z_test)))
z_pred = dnn.predict(x_test)
error = np.mean((z_test - z_pred)**2)
print("Five layers",error)


# One layer 0.09090334050199224
# Two layers 0.0909074526415627
# Three layers 0.09090573362074586
# Four layers 0.09091162015144265
# Five layers 0.090905414146385