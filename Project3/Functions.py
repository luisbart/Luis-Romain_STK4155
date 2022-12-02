#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def FrankeFunction(x,y):
    '''#Definition of the Franke Function'''
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def ScaleData(x_train, x_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    scaler.fit(y_test)
    y_test_scaled = scaler.transform(y_test)
    scaler.fit(y_train)
    y_train_scaled = scaler.transform(y_train)
    
    return x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled

def DesignMatrix(x, y, n ):
    '''This function returns the design matrix of a bi-variate polynomial function'''
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)		# Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X
        
def LinReg(X_train, X_test, y_train):
    '''Performs OLS regression'''
    OLSbeta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
    ytildeTrain = X_train @ OLSbeta
    ytildeTest = X_test @ OLSbeta
    return ytildeTrain, ytildeTest, OLSbeta

def RidgeReg(X_train, X_test, y_train, y_test,lmb):
    '''Performs Ridge regression'''
    Ridgebeta = np.linalg.pinv(X_train.T @ X_train + lmb*np.identity(X_train.shape[1])) @ X_train.T @ y_train
    ytildeTrain = X_train @ Ridgebeta
    ytildeTest = X_test @ Ridgebeta
    return ytildeTrain, ytildeTest, Ridgebeta

def LassoReg(X_train, X_test, y_train, y_test,lmb):
    '''Performs Lasso regression'''
    modelLasso = Lasso(lmb,fit_intercept=False)
    modelLasso.fit(X_train,y_train)
    ytildeTrain = modelLasso.predict(X_train)
    ytildeTest = modelLasso.predict(X_test)
    return ytildeTrain, ytildeTest

def Beta_std(var,X_train,Beta,p):
    '''Computes standard deviation of optimal parameters of Beta in OLS'''
    Beta_var = var*np.linalg.pinv(X_train.T @ X_train)
    err = []
    for p_ in range(p):
        err = np.append(err,Beta_var[p_,p_] ** 0.5)
    return err

def Plot3D(x,y,z):
    '''Makes plot in 3D of terrain surface'''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    x, y = np.meshgrid(x,y)
    
    # Plot the surface 
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_zticks([0, 500, 1000, 1500])
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

def TerrainOLS_CV(maxdegree,k, kfold, x, y, z_scaled):   
    '''Performs CV for OLS in the terrain part'''
    polydegree = np.zeros(maxdegree) 
    
    # Perform the cross-validation to estimate MSE
    scores_KFold_Train = np.zeros((maxdegree, k))
    scores_KFold_Test = np.zeros((maxdegree, k))

    i = 0
    for degree in range(maxdegree):
        polydegree[degree] = degree+1
        X = DesignMatrix(x,y,degree+1)
        j = 0
        for train_inds, test_inds in kfold.split(x):
            X_train = X[train_inds]
            z_train = z_scaled[train_inds]

            X_test = X[test_inds]
            z_test = z_scaled[test_inds]
      
            z_fit, z_pred, beta = LinReg(X_train, X_test, z_train)

            scores_KFold_Train[i,j] = MSE(z_train, z_fit)
            scores_KFold_Test[i,j] = MSE(z_test, z_pred)

            j += 1
        i += 1
        
    estimated_mse_KFold_train = np.mean(scores_KFold_Train, axis = 1)
    estimated_mse_KFold_test = np.mean(scores_KFold_Test, axis = 1)

    plt.figure()
    plt.plot(polydegree, estimated_mse_KFold_train, label = 'KFold train')
    plt.plot(polydegree, estimated_mse_KFold_test, label = 'KFold test')
    plt.xlabel('Complexity')
    plt.ylabel('mse')
    plt.xticks(np.arange(1, maxdegree+1, step=1))  # Set label locations.
    plt.legend()

