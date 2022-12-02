
#%%

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 15:12:27 2022

@author: luis.barreiro
"""
#%%
import os
os.chdir("C:/Users/luis.barreiro/Documents/GitHub/Projects_STK4155/Project3")
cwd = os.getcwd()
print(cwd)
#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
#import Functions module
from Functions import FrankeFunction, LinReg, DesignMatrix, RidgeReg, LassoReg
from IPython.display import Image 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pydot import graph_from_dot_data
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
import scikitplot as skplt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import os


#%%
np.random.seed(3)  

#Bias-variance analysis of the Franke function

#Bootstrap
n_bootstraps = 75

# Make data set
n = 100
maxdegree = 8

x = np.random.uniform(0,1,n)
y = np.random.uniform(0,1,n)
z = FrankeFunction(x, y)
z = z + np.random.normal(0,0.1,z.shape)

x = np.array(x).reshape(n,1)
y = np.array(y).reshape(n,1)
z = np.array(z).reshape(n,1)

x1 = np.hstack((x,y)).reshape(n,2)

#Split train (80%) and test(20%) data before looping on polynomial degree
x_train, x_test, z_train, z_test = train_test_split(x1, z, test_size=0.2)


#%% OLS

#Initialize before looping:
error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)

#Bootstrap LB
for degree in range(maxdegree):
    X_train = DesignMatrix(x_train[:,0],x_train[:,1],degree+1)
    X_test = DesignMatrix(x_test[:,0],x_test[:,1],degree+1)
    z_pred = np.zeros((z_test.shape[0], n_bootstraps))
    for i in range(n_bootstraps):
        x_, z_ = resample(X_train, z_train)
        z_fit, zpred, beta = LinReg(x_, X_test, z_)
        z_pred[:, i] = zpred.ravel()
  
    polydegree[degree] = degree+1
    error[degree] = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    bias[degree] = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance[degree] = np.mean( np.var(z_pred, axis=1, keepdims=True) )
#    print('Polynomial degree:', degree)
#    print('Error:', error[degree])
#    print('Bias^2:', bias[degree])
#    print('Var:', variance[degree])
#    print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.xticks(np.arange(1, 8, step=1))  # Set label locations.
plt.xlabel('Model complexity')
plt.ylabel('Mean squared error')
plt.legend()
#plt.savefig("Results/bias_variance_tradeoff/OLS_bias_var_tradeoff.png",dpi=150)
plt.show()