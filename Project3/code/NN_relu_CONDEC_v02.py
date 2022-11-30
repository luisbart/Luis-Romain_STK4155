# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:01:17 2022

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
from sklearn.datasets import load_breast_cancer
import pickle
import os 
from sklearn.neural_network import MLPRegressor, MLPClassifier

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

#Scaling
x_train_mean = np.mean(x_train)
x_train = x_train - x_train_mean
x_test = x_test - x_train_mean

M = 100   #size of each minibatch
m = int(y_train.shape[0]/M) #number of minibatches
epochs = 100000 #number of epochs 


#%%
# Classify conifer/deciduos  

eta_vals = np.logspace(-8, -5, 4)
lmbd_vals = np.logspace(-3, 1, 5)
n_hidden_neurons = 20
epochs = 100 #number of epochs 

DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = MLPClassifier(hidden_layer_sizes=(n_hidden_neurons), activation='relu', solver ='lbfgs',
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs)
        dnn.fit(x_train, y_train)
        
        DNN_scikit[i][j] = dnn
        
        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("Accuracy score on test set: ", dnn.score(x_test, y_test))
        print()
        

#%%
# visual representation of grid search
# uses seaborn heatmap, could probably do this in matplotlib

sns.set()

train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        dnn = DNN_scikit[i][j]
        
        train_pred = dnn.predict(x_train) 
        test_pred = dnn.predict(x_test)

        train_accuracy[i][j] = accuracy_score(y_train, train_pred)
        test_accuracy[i][j] = accuracy_score(y_test, test_pred)

        
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis", xticklabels=lmbd_vals, yticklabels=eta_vals)
ax.set_title("Training Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.savefig(f"Results/NN/NN_TrainingAccuracy_relu_CON_DEC_.png", dpi=150)
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis", xticklabels=lmbd_vals, yticklabels=eta_vals)
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.savefig(f"Results/NN/NN_TestAccuracy_relu_CON_DEC_.png", dpi=150)
plt.show()

        
#%%
#Choose the right eta and lambda
eta_val = np.logspace(-5, -5, 1)
lmbd_val = np.logspace(10, 10, 1)


for i in range(len(eta_val)):
    for j in range(len(lmbd_val)):
        dnn = DNN_scikit[i][j]
        
        train_pred2 = dnn.predict(x_train) 
        test_pred2 = dnn.predict(x_test)

        train_accuracy[i][j] = accuracy_score(y_train, train_pred2)
        test_accuracy[i][j] = accuracy_score(y_test, test_pred2)
        
        
# Calculate the confusion matrix

conf_matrix = confusion_matrix(y_test, test_pred2)

# Print the confusion matrix using Matplotlib

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.savefig(f"Results/NN/NN_Conf_Matrix_relu_CON_DEC_.png", dpi=150)
plt.show()        
        
TP=conf_matrix[1,1]
TN=conf_matrix[0,0]
FP=conf_matrix[0,1]
FN=conf_matrix[1,0]

Accuracy=(TP+TN)/(TP+TN+FP+FN)
Recall=TP/(TP+FN)
Precision=TP*(TP+FP)
F1_score=2*((Precision*Recall)/(Precision+Recall))    

print("Accuracy:",Accuracy)        
print("F1_score", F1_score)        
        
        
#%%        
conf_matrix = confusion_matrix(y_train, train_pred2)

# Print the confusion matrix using Matplotlib

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)


y_test['CON_DEC'].value_counts()
        
temp=y_test['CON_DEC']        

for col in y_test.columns:
    print(col)
        