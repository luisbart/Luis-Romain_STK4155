# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:01:17 2022

@author: luis.barreiro
"""
'''This program performs hyperparameter tuning for the tree species classifiers 
Author: L Barreiro'''
#%%
#Set working directory
import os
os.chdir("C:/Users/luis.barreiro/Documents/GitHub/Projects_STK4155/Project3")
cwd = os.getcwd()
print(cwd)

#Import some libraries and modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as splitter
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#%%
#Import tree data
np.random.seed(3)        #create same seed for random number every time

trees=pd.read_csv("input_data\input_trees.csv")     #Load tree data

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
n_bootstraps = 20

y_test = np.array(y_test).reshape(y_test.shape[0],1)


eta_val = 1e-7
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

sns.set(font_scale=1.5)
plt.plot(range(1,n_hidden_neurons+1), error)
plt.ylabel('MSE', fontsize=15)
plt.xlabel('Number of neurons in one layer', fontsize=15)
plt.xticks(np.arange(0, n_hidden_neurons+1, step=5), fontsize=15)  # Set label locations.
plt.tight_layout()
plt.savefig("Results/MLP_n_neurons.png",dpi=150)
plt.show()

#%% Test which number of hidden layers works best

n_bootstraps = 50

y_test = np.array(y_test).reshape(y_test.shape[0],1)


eta_val = 1e-7
lmbd_val = 1e-2
n_hidden_neurons=[[1],[1,1],[1,1,1],[1,1,1,1],[1,1,1,1,1]]
epochs = 10000 #number of epochs 

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
plt.xlabel('Number of layers')
plt.xticks(np.arange(1, len(n_hidden_neurons)+1, step=1))  # Set label locations.
plt.tight_layout()
#plt.savefig("Results/MLP_n_layers.png",dpi=150)
plt.show()

#%%
# Test hyperparameters on NN

eta_vals = np.logspace(-8, -5, 4)
lmbd_vals = np.logspace(-3, 1, 5)
n_hidden_neurons = 1
epochs = 10000 #number of epochs 

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
# visual representation of grid search for NN hyperparameters

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

        
plt.rcParams.update({'font.size': 20})
      
fig, ax = plt.subplots(figsize = (10, 10))
sns.set(font_scale=2)
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy",fontsize = 20)
ax.set_ylabel("$\eta$",fontsize = 20)
ax.set_xlabel("$\lambda$",fontsize = 20)
ax.set_xticklabels(lmbd_vals,fontsize = 20)
ax.set_yticklabels(eta_vals, fontsize = 20)
plt.savefig(f"Results/NN_TestAccuracy_relu_CON_DEC.png", dpi=150)
plt.show()

    
fig, ax = plt.subplots(figsize = (10, 10))
sns.set(font_scale=2)
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Train Accuracy",fontsize = 20)
ax.set_ylabel("$\eta$",fontsize = 20)
ax.set_xlabel("$\lambda$",fontsize = 20)
ax.set_xticklabels(lmbd_vals,fontsize = 20)
ax.set_yticklabels(eta_vals, fontsize = 20)
plt.savefig(f"Results/NN_TrainAccuracy_relu_CON_DEC_.png", dpi=150)
plt.show()


#%%
#Plot confusion matrix for NN withse the right eta and lambda

dnn = MLPClassifier(hidden_layer_sizes=(n_hidden_neurons), activation='relu', solver ='lbfgs',
                    alpha=1e-2, learning_rate_init=1e-7, max_iter=epochs)

dnn.fit(x_train, y_train)

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
        
      
#%%
# hyperparameter tuning for logistic regression

#Logistic regression
etas = np.logspace(-7, -7, 1)
lambdas = np.logspace(-3, 1, 5)

#Initialize error to store
logreg_scikit_train = np.zeros(shape=(lambdas.shape[0],etas.shape[0]))
logreg_scikit_test = np.zeros(shape=(lambdas.shape[0],etas.shape[0]))

#Looping through regularization and learning rates
i=0
for lmbd in lambdas:
    j=0  
    for eta in etas: 
        
        logreg = LogisticRegression(solver='lbfgs',max_iter=10000,C = 1/lmbd)
        logreg.fit(x_train, y_train)
        y_fit = logreg.predict(x_train)
        y_pred = logreg.predict(x_test)
        
        logreg_scikit_train[i,j] = accuracy_score(y_train,y_fit)
        logreg_scikit_test[i,j] = accuracy_score(y_test,y_pred)
        
        j+=1
    i+=1

#Heatmap for gridsearch 
x_axis = etas # labels for x-axis
y_axis = lambdas # labels for y-axis 

heat3 = sns.heatmap(logreg_scikit_train,vmin=0.009,vmax=1,annot=True, xticklabels='', yticklabels=y_axis, cmap="viridis",linewidths =0.5) 
heat3.set(xlabel='', ylabel ='regularization', title = f"Accuracy score training set Scikit")
plt.savefig(f"Results/heatmap_Logistic_regression_train_scikit.png", dpi=150)
plt.show()


heat4 = sns.heatmap(logreg_scikit_test,vmin=0.009,vmax=1,annot=True, xticklabels='', yticklabels=y_axis, cmap="viridis",linewidths =0.5)
heat4.set(xlabel='', ylabel ='regularization', title = f"Accuracy score test set Scikit")
plt.savefig(f"Results/heatmap_Logistic_regression_test_scikit.png", dpi=150)
plt.show()


#%%
#Hyperparameter tuning for Decision tree / random forest

#find best n_estimators and max_depth

scores =[]
for k in range(1, 101):
    rfc = RandomForestClassifier(n_estimators=k)
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    scores.append(accuracy_score(y_test, y_pred))

# plot the relationship between k and testing accuracy
sns.set(font_scale=1)
plt.plot(range(1, 101), scores)
plt.xlabel('Value of n_estimators for Random Forest Classifier')
plt.ylabel('Accuracy score')
plt.savefig(f"Results/RF_test_n_estimators_scikit.png", dpi=150)


scores_train =[]
scores_test = []
for d in range(1, 51):
    rfc = RandomForestClassifier(max_depth=d)
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    y_fit = rfc.predict(x_train)
    scores_test.append(accuracy_score(y_test, y_pred))
    scores_train.append(accuracy_score(y_train, y_fit))

# plot the relationship between d and testing accuracy
sns.set(font_scale=1)
plt.plot(range(1, 51), scores_test)
plt.plot(range(1, 51), scores_train)
plt.xlabel('Value of max_depth for Random Forest Classifier')
plt.ylabel('Accuracy score')
plt.legend(["test score", "train score"])
plt.savefig(f"Results/RF_test_max_depth_scikit.png", dpi=150)


#%%
#Hyperparameter tuning for DT
# max_depth and criterion for DecisionTreeClassifier

scores_train =[]
scores_test = []
for d in range(1, 51):
    dt = DecisionTreeClassifier(max_depth=d, criterion="entropy")
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    y_fit = dt.predict(x_train)
    scores_test.append(accuracy_score(y_test, y_pred))
    scores_train.append(accuracy_score(y_train, y_fit))

# plot the relationship between d and testing accuracy
sns.set(font_scale=1)
plt.plot(range(1, 51), scores_test)
plt.plot(range(1, 51), scores_train)
plt.xlabel('Value of max_depth for Random Forest Classifier')
plt.ylabel('Accuracy score')
plt.legend(["test score", "train score"])
plt.savefig(f"Results/RF_test_max_depth_scikit.png", dpi=150)

#%%
#Hyperparameter tuning for SVM

# penalty parameter C in SVM. The strength of the regularization is inversely proportional to C. 
#The penalty is a squared L2 penalty

C_vals=np.arange(0.1,50,0.1)

scores =[]
for c in C_vals:
    svm = SVC(gamma='scale', C=c)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    scores.append(accuracy_score(y_test, y_pred))

# plot the relationship between C and testing accuracy
sns.set(font_scale=1)
plt.plot(C_vals, scores)
plt.xlabel('Value of Regularization parameter for SVM')
plt.ylabel('Accuracy score')
plt.savefig(f"Results/SVM_test_C_scikit.png", dpi=150)

#%%

gammas = np.logspace(-10, -1, 10)
scores_train =[]
scores_test = []
for g in gammas:
    svm = SVC(gamma=g, C=1)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    y_fit = svm.predict(x_train)
    scores_test.append(accuracy_score(y_test, y_pred))
    scores_train.append(accuracy_score(y_train, y_fit))

 # plot the relationship between d and testing accuracy
sns.set(font_scale=1)
plt.plot(np.log10(gammas), scores_test, label = 'test score')
plt.plot(np.log10(gammas), scores_train, label = 'train score')
plt.xlabel('Kernel coefficient log10(gamma) for SVM')
plt.ylabel('Accuracy score')
plt.legend(["test score", "train score"])
plt.savefig(f"Results/SVM_test_gamma_scikit.png", dpi=150)