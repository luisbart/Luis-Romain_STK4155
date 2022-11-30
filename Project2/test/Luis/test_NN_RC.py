# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 21:02:31 2022

@author: luis.barreiro
"""
import pandas as pd
import numpy as np
from numpy import vectorize
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

import sys
sys.path.append("class/")
from NN_als import NeuralNetwork
from NN_LB import NeuralNetwork as NeuralNetwork2
#import projectfunctions as pf

from Functions import FrankeFunction, DesignMatrix, MSE, R2, Plot3D, create_mini_batches

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function

sns.set()
sns.set_style("whitegrid")
sns.set_palette("husl")


#testing NN on Franke's Function
n = 100
maxdegree = 4
noise=0.1

x = np.arange(0, 1, 0.01)
y = np.arange(0, 1, 0.01)
z = FrankeFunction(x, y)
#z = 1 + x + y + x*y + x**2 + y**2

# Add random distributed noise
var = 0.1
z = z + np.random.normal(0,var,z.shape)



# x = np.array(x)
# y = np.array(y)



#x_grid, y_grid = np.meshgrid(x, y)

#flatten x and y
#x = x_grid.flatten()
#y = y_grid.flatten()

x1 = np.hstack((x,y)).reshape(x.shape[0],2)

#compute z and flatten it
#z_grid = FrankeFunction(x_grid, y_grid)
#z = z_grid.flatten() + np.random.normal(0,noise,len(x))

#X = DesignMatrix(x_grid,y_grid, maxdegree)

#X = np.array([x,y]).transpose()


X_train, X_test, z_train, z_test = train_test_split(x1, z, test_size=0.2)





#%%
from NeuralNetwork_regression_sigmoid import NeuralNetwork as NeuralNetworkRC

z_train=z_train.reshape(z_train.shape[0],1)
z_test=z_test.reshape(z_test.shape[0],1)

input_neurons = X_train.shape[1]
architecture = [input_neurons, 4, 1]
epochs = 10
batch_size = 20
eta = 0.1
lmbd = 0#.0001
n_layers = len(architecture)



n_hidden_neurons=[50]
n_categories=10
epochs=100
batch_size=20
eta=0.1
lmbd=0


dnn = NeuralNetworkRC(X_train, z_train, n_hidden_neurons=n_hidden_neurons, n_categories=n_categories, epochs=epochs, batch_size=batch_size, eta=eta, lmbd=lmbd)
#dnn = NeuralNetworkRC(X_train, z_train)
dnn.train()
test_predict = dnn.predict(X_train)