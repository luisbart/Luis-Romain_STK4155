# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 20:54:30 2022

@author: luis.barreiro
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
#from sklearn.preprocessing import StandardScaler
from Functions import FrankeFunction, R2, MSE, DesignMatrix, LinReg, RidgeReg, LassoReg, ScaleData
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy.random import normal, uniform
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import seaborn as sb


#%%
# Load the terrain
terrain = imread('data/SRTM_data_Norway_2.tif')

# Show the terrain
plt.figure()
plt.title('Terrain over Norway')
plt.imshow(terrain, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("plots/Terrain/Map_v01.png",dpi=150)
plt.show()
#%%
n = 1000
maxdegree = 5 # polynomial order
terrain = terrain[:n,:n]
# Creates mesh of image pixels
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])
x_mesh, y_mesh = np.meshgrid(x,y)

x_mesh = np.ravel(x_mesh)
y_mesh = np.ravel(y_mesh)

z = terrain.ravel()

# Show the terrain
plt.figure()
plt.title('Terrain over Norway')
plt.imshow(terrain, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("plots/Terrain/Map_v02.png",dpi=150)
plt.show()
#%% Cross validation in OLS

#Scale the data
x = x.reshape(n,1)
y = y.reshape(n,1)

scaler = StandardScaler()

scaler.fit(x)
x_scaled = scaler.transform(x)

scaler.fit(y)
y_scaled = scaler.transform(y)

z=terrain
scaler.fit(z)
z_scaled = scaler.transform(z)



MSE_test = np.zeros(maxdegree)
MSE_train = np.zeros(maxdegree)
k=10

# Decide degree on polynomial to fit
poly = PolynomialFeatures(degree = 6)

# Initialize a KFold instance
k = 10
kfold = KFold(n_splits = k)

# Perform the cross-validation to estimate MSE
scores_KFold_Train = np.zeros((maxdegree, k))
scores_KFold_Test = np.zeros((maxdegree, k))

#
polydegree = np.zeros(maxdegree)

i = 0
for degree in range(maxdegree):
    polydegree[degree] = degree+1
    X = DesignMatrix(x_scaled,y_scaled,degree+1)
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
plt.xticks(np.arange(1, 6, step=1))  # Set label locations.
plt.legend()
plt.title('K-fold Cross Validation, k = 5, OLS')
#plt.savefig("plots/Terrain/CV_OLS.png",dpi=150)
plt.show()


#%%
# CV in Ridge

#set up the hyper-parameters to investigate
nlambdas = 9
lambdas = np.logspace(-1, 7, nlambdas)


# Plot all in the same figure as subplots

#Initialize before looping:
polydegree = np.zeros(maxdegree)
error_Kfold = np.zeros((maxdegree,k))
estimated_mse_Kfold = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)

E = np.zeros((maxdegree,9))

# Create a matplotlib figure
fig, ax = plt.subplots()

for l in range(nlambdas):   
    i=0
    for degree in range(maxdegree): 
        j=0
        for train_inds, test_inds in kfold.split(x):
            
            X = DesignMatrix(x_scaled,y_scaled,degree+1)
            
            X_train = X[train_inds]
            z_train = z_scaled[train_inds]   
            X_test = X[test_inds]
            z_test = z_scaled[test_inds]
                 
            z_fit, z_pred, Beta = RidgeReg(X_train, X_test, z_train, z_test,lambdas[l])
            
            error_Kfold[i,j] = MSE(z_test,z_pred)
            
            j+=1
            
        estimated_mse_Kfold[degree] = np.mean(error_Kfold[i,:])
        polydegree[degree] = degree+1
                
        i+=1
    
    E[:,l] = estimated_mse_Kfold
    ax.plot(polydegree, estimated_mse_Kfold, label='%.0e' %lambdas[l])

plt.xlabel('Model complexity')    
plt.xticks(np.arange(1, len(polydegree)+1, step=1))  # Set label locations.
plt.ylabel('log10(MSE)')
plt.title('MSE Ridge regression for different lambdas (kfold=10)')

# Add a legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], title='lambda', loc='center right', bbox_to_anchor=(1.27, 0.5))

#Save figure
plt.savefig("plots/Terrain/CV_Ridge.png",dpi=150, bbox_inches='tight')
plt.show()

#Create a heatmap with the error per nlambdas and polynomial degree
heatmap = sb.heatmap(E,annot=True, annot_kws={"size":7}, cmap="coolwarm", xticklabels=lambdas, yticklabels=range(1,maxdegree+1), cbar_kws={'label': 'Mean squared error'})
heatmap.invert_yaxis()
heatmap.set_ylabel("Complexity")
heatmap.set_xlabel("lambda")
heatmap.set_title("MSE heatmap, Cross Validation, kfold = {:}".format(k))
plt.tight_layout()
plt.savefig("plots/Terrain/CV_Ridge_heatmap.png",dpi=150)
plt.show()


#%%
#CV in Lasso

#set up the hyper-parameters to investigate
nlambdas = 9
lambdas = np.logspace(-3, 5, nlambdas)

#Initialize before looping:
polydegree = np.zeros(maxdegree)
error_Kfold = np.zeros((maxdegree,k))
estimated_mse_Kfold = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)

E = np.zeros((maxdegree,9))

# Create a matplotlib figure
fig, ax = plt.subplots()

for l in range(nlambdas):   
    i=0
    for degree in range(maxdegree): 
        j=0
        for train_inds, test_inds in kfold.split(x):
            
            X = DesignMatrix(x_scaled,y_scaled,degree+1)
            
            X_train = X[train_inds]
            z_train = z_scaled[train_inds]   
            X_test = X[test_inds]
            z_test = z_scaled[test_inds]
            
            z_fit, z_pred = LassoReg(X_train, X_test, z_train, z_test,lambdas[l])
            
            error_Kfold[i,j] = MSE(z_test,z_pred)
            
            j+=1
            
        estimated_mse_Kfold[degree] = np.mean(error_Kfold[i,:])
        polydegree[degree] = degree+1
                
        i+=1

    E[:,l] = estimated_mse_Kfold    
    ax.plot(polydegree, estimated_mse_Kfold, label='%.0e' %lambdas[l])

plt.xlabel('Model complexity')    
plt.xticks(np.arange(1, len(polydegree)+1, step=1))  # Set label locations.
plt.ylabel('MSE')
plt.title('MSE Lasso regression for different lambdas (Kfold=10)')

# Add a legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], title='lambda', loc='center right', bbox_to_anchor=(1.27, 0.5))

#Save figure
plt.savefig("plots/Terrain/CV_Lasso.png",dpi=150, bbox_inches='tight')
plt.show()

#%%
#Create a heatmap with the error per nlambdas and polynomial degree

heatmap = sb.heatmap(E,annot=True, annot_kws={"size":7}, cmap="coolwarm", xticklabels=lambdas, yticklabels=range(1,maxdegree+1), cbar_kws={'label': 'Mean squared error'})
heatmap.invert_yaxis()
heatmap.set_ylabel("Complexity")
heatmap.set_xlabel("lambda")
heatmap.set_title("MSE heatmap, Cross Validation, kfold = {:}".format(k))
plt.tight_layout()
plt.savefig("plots/Terrain/CV_Lasso_heatmap.png",dpi=150)
plt.show()