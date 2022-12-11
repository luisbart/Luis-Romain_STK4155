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
from sklearn.preprocessing import StandardScaler
import pickle
import os 
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

#%%
import os
os.chdir("C:/Users/luis.barreiro/Documents/GitHub/Projects_STK4155/Project3")
cwd = os.getcwd()
print(cwd)

#%%
np.random.seed(3)        #create same seed for random number every time

trees=pd.read_csv("input_data\input_trees_v04.csv")     #Load tree data

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


#%% Test which number of hidden neurons works best

#Number of bootstraps
n_bootstraps = 50

y_test = np.array(y_test).reshape(y_test.shape[0],1)


eta_val = 1e-6
lmbd_val = 1e-2
n_hidden_neurons = 20
epochs = 1000 #number of epochs 

error = np.zeros(n_hidden_neurons)

for n in range(n_hidden_neurons):
    y_pred = np.empty((y_test.shape[0],n_bootstraps))
    for i in range(n_bootstraps):
        x_, y_ = resample(x_train,y_train)
 

        dnn = MLPClassifier(hidden_layer_sizes=n+1, activation='relu', solver ='lbfgs',alpha=lmbd_val, learning_rate_init=eta_val, max_iter=epochs)
        dnn.fit(x_, y_)
        print("Test set accuracy Neural Network: {:.2f}".format(dnn.score(x_test,y_test)))
        y_pred[:, i] = dnn.predict(x_test)
    
    error[n] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )

plt.plot(range(1,n_hidden_neurons+1), error)
plt.ylabel('MSE')
plt.xlabel('Model complexity: number of neurons in one layer')
plt.xticks(np.arange(0, n_hidden_neurons+1, step=5))  # Set label locations.
plt.legend()
plt.savefig("Results/MLP_n_neurons.png",dpi=150)
plt.show()

#%% Test which number of hidden layers works best

n_bootstraps = 50

y_test = np.array(y_test).reshape(y_test.shape[0],1)


eta_val = 1e-6
lmbd_val = 1e-2
n_hidden_neurons=[[1],[1,1],[1,1,1],[1,1,1,1],[1,1,1,1,1]]
epochs = 1000 #number of epochs 

error = np.zeros(len(n_hidden_neurons))

for n in range(len(n_hidden_neurons)):
    y_pred = np.empty((y_test.shape[0],n_bootstraps))
    for i in range(n_bootstraps):
        x_, y_ = resample(x_train,y_train)
             
        dnn = MLPClassifier(hidden_layer_sizes=n_hidden_neurons[n], activation='relu', solver ='lbfgs',alpha=lmbd_val, learning_rate_init=eta_val, max_iter=epochs)
        dnn.fit(x_, y_)
        print("Test set accuracy Neural Network: {:.2f}".format(dnn.score(x_test,y_test)))
        y_pred[:,i] = dnn.predict(x_test)
    
    error[n] = np.mean((y_test - y_pred)**2)


plt.plot(range(1,len(n_hidden_neurons)+1), error)
plt.ylabel('MSE')
plt.xlabel('Model complexity: number of layers')
plt.xticks(np.arange(1, len(n_hidden_neurons)+1, step=1))  # Set label locations.
plt.legend()
plt.savefig("Results/MLP_n_layers.png",dpi=150)
plt.show()

#%%
# Classify conifer/deciduos  

eta_vals = np.logspace(-8, -5, 4)
lmbd_vals = np.logspace(-3, 1, 5)
n_hidden_neurons = 1
epochs = 1000 #number of epochs 

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
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis", xticklabels=lmbd_vals, yticklabels=eta_vals)
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.savefig(f"Results/NN_TestAccuracy_relu_CON_DEC_.png", dpi=150)
plt.show()

    
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis", xticklabels=lmbd_vals, yticklabels=eta_vals)
ax.set_title("Train Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.savefig(f"Results/NN_TrainAccuracy_relu_CON_DEC_.png", dpi=150)
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
plt.savefig(f"Results/NN_Conf_Matrix_relu_CON_DEC_.png", dpi=150)
plt.show()        
        
      
        
        
        