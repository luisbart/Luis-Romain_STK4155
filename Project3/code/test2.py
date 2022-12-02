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

#%% Ridge
#For Ridge regression, set up the hyper-parameters to investigate
maxdegree=5
nlambdas = 9
lambdas = np.logspace(-7, 1, nlambdas)

#Initialize before looping:
TestError = np.zeros(maxdegree)
TrainError = np.zeros(maxdegree)
TestR2 = np.zeros(maxdegree)
TrainR2 = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
predictor =[]

error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)

E = np.zeros((maxdegree,9))

#Initialize bootstrap matrice
z_pred = np.empty((z_test.shape[0],n_bootstraps))

# Loop for subplotting Error-Bias-variance
fig = plt.figure(figsize=(15,12))
c=1

for l in range(nlambdas):
    for degree in range(maxdegree):   
        for i in range(n_bootstraps):
            x_, z_ = resample(x_train,z_train)
      
            X_train = DesignMatrix(x_[:,0],x_[:,1],degree+1)
            X_test = DesignMatrix(x_test[:,0],x_test[:,1],degree+1)
            #z_fit, z_pred[:,i], Beta = RidgeReg(X_train, X_test, z_, z_test,lambdas[l])
            Beta = np.linalg.pinv(X_train.T @ X_train + 1*np.identity(X_train.shape[1])) @ X_train.T @ z_
            z_fit = X_train @ Beta
            z_pred = X_test @ Beta
                    
        z_test = np.reshape(z_test, (len(z_test),1))
        predictor=np.append(predictor,Beta)
        polydegree[degree] = degree+1
           
        error[degree] = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
        bias[degree] = np.mean( (z_test - np.mean(z_pred, axis=1,keepdims=True))**2 )
        variance[degree] = np.mean( np.var(z_pred, axis=1,keepdims=True) )
        
    E[:,l] = error
    

    plt.subplot(3,3,c)
    plt.plot(range(1,maxdegree+1), error, label = 'Error')
    plt.plot(range(1,maxdegree+1), bias, label = 'Bias')
    plt.plot(range(1,maxdegree+1), variance, label = 'Variance')
    plt.title('lambda = %.0e' %lambdas[l])
    c = c+1

fig.text(0.5, 0.08, 'Model complexity', ha='center')
fig.text(0.07, 0.5, 'Error', va='center', rotation='vertical')
fig.suptitle("Variance-Bias tradeoff for different lambda (Ridge)", fontsize=18, y=0.95)
plt.legend()
plt.savefig("Results/bias_variance_tradeoff/Ridge_bias_var_tradeoff_LAMBDAS.png",dpi=150)
plt.show()


#%% Lasso
maxdegree=20
#Initialize before looping:
TestError = np.zeros(maxdegree)
TrainError = np.zeros(maxdegree)
TestR2 = np.zeros(maxdegree)
TrainR2 = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)

error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)

E = np.zeros((maxdegree,9))

#Initialize bootstrap matrice
z_pred = np.empty((z_test.shape[0],n_bootstraps))

# Loop for subplotting Error-Bias-variance
fig = plt.figure(figsize=(15,12))
c=1

for l in range(nlambdas):
    for degree in range(maxdegree):   
        for i in range(n_bootstraps):
            x_, z_ = resample(x_train,z_train)
      
            X_train = DesignMatrix(x_[:,0],x_[:,1],degree+1)
            X_test = DesignMatrix(x_test[:,0],x_test[:,1],degree+1)
            z_fit, z_pred[:,i] = LassoReg(X_train, X_test, z_, z_test,lambdas[l])
                    
        z_test = np.reshape(z_test, (len(z_test),1))
        polydegree[degree] = degree+1
           
        error[degree] = np.mean( np.mean((z_test - z_pred)**2, keepdims=True) )
        bias[degree] = np.mean( (z_test - np.mean(z_pred, keepdims=True))**2 )
        variance[degree] = np.mean( np.var(z_pred, keepdims=True) )
        
    E[:,l] = error

    plt.subplot(3,3,c)
    plt.plot(range(1,maxdegree+1), error, label = 'Error')
    plt.plot(range(1,maxdegree+1), bias, label = 'Bias')
    plt.plot(range(1,maxdegree+1), variance, label = 'Variance')
    plt.title('lambda = %.0e' %lambdas[l])
    c = c+1

fig.text(0.5, 0.08, 'Model complexity', ha='center')
fig.text(0.07, 0.5, 'Error', va='center', rotation='vertical')
fig.suptitle("Variance-Bias tradeoff for different lambda (Lasso)", fontsize=18, y=0.95)
plt.legend()
plt.savefig("Results/bias_variance_tradeoff/Lasso_bias_var_tradeoff_LAMBDAS.png",dpi=150)
plt.show()