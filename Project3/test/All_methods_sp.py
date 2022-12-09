# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 21:36:12 2022

@author: luis.barreiro
"""

#%%
# Common imports
from IPython.display import Image 
from pydot import graph_from_dot_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from sklearn.neural_network import MLPClassifier
import os

#%%
os.chdir("C:/Users/luis.barreiro/Documents/GitHub/Projects_STK4155/Project3")
cwd = os.getcwd()
print(cwd)

#%%
np.random.seed(3)        #create same seed for random number every time

trees=pd.read_csv("input_data\input_trees_v05.csv")     #Load tree data

trees.columns

x=trees[['min', 'max', 'avg', 'std', 'ske', 'kur', 'p05', 'p25','p50', 'p75', 'p90', 'c00', 'int_min', 'int_max', 'int_avg', 'int_std','int_ske', 'int_kur', 'int_p05', 'int_p25', 'int_p50', 'int_p75','int_p90']]
y1=trees['CON_DEC']
y2=trees['Specie']

x_train,x_test,y_train,y_test=splitter(x,y2,test_size=0.3)   #Split datasets into training and testing

#Scale the data
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

#%%

#define methods
# Neural Network with 20 hidden layers, eta=1.e-5, lmbd=10
dnn = MLPClassifier(hidden_layer_sizes=20, activation='relu', solver ='lbfgs',alpha=10, learning_rate_init=1.e-5, max_iter=1000)
dnn.fit(x_train_scaled, y_train)
print("Test set accuracy Neural Network with scaled data: {:.2f}".format(dnn.score(x_test_scaled,y_test)))
# Logistic Regression
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(x_train_scaled, y_train)
print("Test set accuracy Logistic Regression with scaled data: {:.2f}".format(logreg.score(x_test_scaled,y_test)))
# Decision Trees
deep_tree_clf = DecisionTreeClassifier(max_depth=None)
deep_tree_clf.fit(x_train_scaled, y_train)
print("Test set accuracy with Decision Trees and scaled data: {:.2f}".format(deep_tree_clf.score(x_test_scaled,y_test)))
# Support Vector Machine
svm = SVC(gamma='auto', C=100)
svm.fit(x_train_scaled, y_train)
print("Test set accuracy SVM with scaled data: {:.2f}".format(logreg.score(x_test_scaled,y_test)))
# Random forests
#Instantiate the model with 500 trees and entropy as splitting criteria
Random_Forest_model = RandomForestClassifier(n_estimators=500,criterion="entropy")
Random_Forest_model.fit(x_train_scaled, y_train)
print("Test set accuracy with Random Forests and scaled data: {:.2f}".format(Random_Forest_model.score(x_test_scaled,y_test)))


#%%
#Confusion matrix with Neural Network
y_pred = dnn.predict(x_test_scaled)
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
plt.show()
y_probas = dnn.predict_proba(x_test_scaled)
skplt.metrics.plot_roc(y_test, y_probas3)
plt.show()
skplt.metrics.plot_cumulative_gain(y_test, y_probas3)
plt.show()

#Confusion matrix with Random Forest
y_pred1 = Random_Forest_model.predict(x_test_scaled)
skplt.metrics.plot_confusion_matrix(y_test, y_pred1, normalize=True)
plt.show()
y_probas1 = Random_Forest_model.predict_proba(x_test_scaled)
skplt.metrics.plot_roc(y_test, y_probas1)
plt.show()
skplt.metrics.plot_cumulative_gain(y_test, y_probas1)
plt.show()

#Confusion matrix with SVM
y_pred2 = svm.predict(x_test_scaled)
skplt.metrics.plot_confusion_matrix(y_test, y_pred2, normalize=True)
plt.show()
y_probas2 = svm.predict_proba(x_test_scaled)
skplt.metrics.plot_roc(y_test, y_probas2)
plt.show()
skplt.metrics.plot_cumulative_gain(y_test, y_probas2)
plt.show()

#Confusion matrix with Logistic
y_pred3 = logreg.predict(x_test_scaled)
skplt.metrics.plot_confusion_matrix(y_test, y_pred3, normalize=True)
plt.show()
y_probas3 = logreg.predict_proba(x_test_scaled)
skplt.metrics.plot_roc(y_test, y_probas3)
plt.show()
skplt.metrics.plot_cumulative_gain(y_test, y_probas3)
plt.show()























