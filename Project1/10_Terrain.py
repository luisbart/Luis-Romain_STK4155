'''This program performs Ordinary least square, Ridge and Lasso regression on a terrain dataset
and cross-validation as resampling technique to evaluate which model fits the data best.
Author: R Corseri & L Barreiro'''

#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from Functions import MSE, DesignMatrix, RidgeReg, LassoReg, Plot3D, TerrainOLS_CV
from imageio import imread
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import seaborn as sb


#%%
# Load the terrain and show map
terrain = imread('data/SRTM_data_Norway_1.tif')

# Show the terrain
plt.figure()
plt.title('Terrain over Norway')
plt.imshow(terrain, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
#plt.savefig("plots/Terrain/Map_v01.png",dpi=150)
plt.show()
#%% Show selected area
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
plt.colorbar()
#plt.savefig("plots/Terrain/Map_v02.png",dpi=150)
plt.show()


#Plot 3D Original 
Plot3D(x,y,terrain)    
plt.title('Original terrain')
#plt.savefig("plots/Terrain/Map_3D.png", dpi=150)
plt.show()

#%% Cross validation in OLS up to degree 5

#Scale the data
x1 = x.reshape(n,1)
y1 = y.reshape(n,1)

scaler = StandardScaler()

scaler.fit(terrain)
z_scaled = scaler.transform(terrain)

# Initialize a KFold instance
k=10
kfold = KFold(n_splits = k)

   
       
TerrainOLS_CV(maxdegree,k, kfold, x1, y1, z_scaled)
plt.title('K-fold Cross Validation, k = 10, OLS')
#plt.savefig("plots/Terrain/CV_OLS.png",dpi=150)
plt.show()

#And now to degree 10
maxdegree=10
TerrainOLS_CV(maxdegree,k, kfold, x1, y1, z_scaled)
plt.title('K-fold Cross Validation, k = 10, OLS')
#plt.savefig("plots/Terrain/CV_OLS_pol1to10.png",dpi=150)
plt.show()

#%% Make some more OLS plots
#For complexity=3
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])


deg=3
X1 = DesignMatrix(x,y,deg)
OLSbeta1 = np.linalg.pinv(X1.T @ X1) @ X1.T @ terrain
ytilde1 = X1 @ OLSbeta1

#Plot 3D OLS
Plot3D(x,y,ytilde1)    
plt.title('OLS, pol=3')
#plt.savefig("plots/Terrain/Map_3d_OLS_pol3.png", dpi=150)
plt.show()

#%% RIDGE

#set up the hyper-parameters to investigate
nlambdas = 9
lambdas = np.logspace(-4, 4, nlambdas)

#Initialize before looping:
polydegree = np.zeros(maxdegree)
error_Kfold_train = np.zeros((maxdegree,k))
error_Kfold_test = np.zeros((maxdegree,k))
estimated_mse_Kfold_train = np.zeros(maxdegree)
estimated_mse_Kfold_test = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)

Etest = np.zeros((maxdegree,9))
Etrain = np.zeros((maxdegree,9))

# Create a matplotlib figure
fig, ax = plt.subplots()

for l in range(nlambdas):   
    i=0
    for degree in range(maxdegree): 
        j=0
        for train_inds, test_inds in kfold.split(x):
            
            X = DesignMatrix(x,y,degree+1)
            
            X_train = X[train_inds]
            z_train = z_scaled[train_inds]   
            X_test = X[test_inds]
            z_test = z_scaled[test_inds]
                 
            z_fit, z_pred, Beta = RidgeReg(X_train, X_test, z_train, z_test,lambdas[l])
            
            error_Kfold_test[i,j] = MSE(z_test,z_pred)
            error_Kfold_train[i,j] = MSE(z_train,z_fit)
            
            j+=1
        
        estimated_mse_Kfold_test[degree] = np.mean(error_Kfold_test[i,:])
        estimated_mse_Kfold_train[degree] = np.mean(error_Kfold_train[i,:])
        polydegree[degree] = degree+1
                
        i+=1
    
    Etest[:,l] = estimated_mse_Kfold_test
    Etrain[:,l] = estimated_mse_Kfold_train
    ax.plot(polydegree, estimated_mse_Kfold_test, label='%.0e' %lambdas[l])

plt.xlabel('Model complexity')    
plt.xticks(np.arange(1, len(polydegree)+1, step=1))  # Set label locations.
plt.ylabel('MSE')
plt.title('MSE Ridge regression for different lambdas (kfold=10)')

# Add a legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], title='lambda', loc='center right', bbox_to_anchor=(1.27, 0.5))

#Save figure
#plt.savefig("plots/Terrain/CV_Ridge_lambda_pol1to10.png",dpi=150, bbox_inches='tight')
plt.show()

#Compare train and test performance
plt.figure()
plt.plot(polydegree, Etrain[:,2], label = 'KFold train')
plt.plot(polydegree, Etest[:,2], label = 'KFold test')
plt.xlabel('Complexity')
plt.ylabel('mse')
plt.xticks(np.arange(1, maxdegree+1, step=1))  # Set label locations.
plt.legend()
plt.title('K-fold Cross Validation, k=10, Ridge, lambda=10')
#plt.savefig("plots/Terrain/CV_Ridge_pol1to10.png",dpi=150)
plt.show()

#Create a heatmap with the error per nlambdas and polynomial degree
heatmap = sb.heatmap(Etest,annot=True, annot_kws={"size":7}, cmap="coolwarm", xticklabels=lambdas, yticklabels=range(1,maxdegree+1), cbar_kws={'label': 'Mean squared error'})
heatmap.invert_yaxis()
heatmap.set_ylabel("Complexity")
heatmap.set_xlabel("lambda")
heatmap.set_title("MSE heatmap, Cross Validation, kfold = {:}".format(k))
plt.tight_layout()
#plt.savefig("plots/Terrain/CV_Ridge_heatmap_pol1to10.png",dpi=150)
plt.show()

#%%
#Make some more Ridge plots
#For complexity=1, lambda=10^3 (MSE_train=0.82)
deg=5
lmb=10

X1 = DesignMatrix(x,y,deg)
Ridgebeta1 = np.linalg.pinv(X1.T @ X1 + lmb*np.identity(X1.shape[1])) @ X1.T @ terrain
ytilde2 = X1 @ Ridgebeta1

#Plot 3D Ridge
Plot3D(x,y,ytilde2)    
plt.title('Ridge, pol=4, lmb=10')
#plt.savefig("plots/Terrain/Map_3d_Ridge_pol4_lmb10.png", dpi=150)
plt.show()

#Compare train and test performance
plt.figure()
plt.plot(polydegree, Etrain[:,2], label = 'KFold train')
plt.plot(polydegree, Etest[:,2], label = 'KFold test')
plt.xlabel('Complexity')
plt.ylabel('mse')
plt.xticks(np.arange(1, maxdegree+1, step=1))  # Set label locations.
plt.legend()
plt.title('K-fold Cross Validation, k = 10, Ridge, lambda=10')
#plt.savefig("plots/Terrain/CV_Ridge.png",dpi=150)
plt.show()



#%%
#CV in Lasso: Many ConvergenceWarning, tested also by increasing number of iterations quite a lot, but we still get warnings.
#In the lab session we were told that as long as we get meaningful results, we could ignore this warning

#set up the hyper-parameters to investigate
lambdas = np.logspace(-6, 2, nlambdas)

#Initialize before looping:
polydegree = np.zeros(maxdegree)
error_Kfold_test = np.zeros((maxdegree,k))
error_Kfold_train = np.zeros((maxdegree,k))
estimated_mse_Kfold_train = np.zeros(maxdegree)
estimated_mse_Kfold_test = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)

Etrain = np.zeros((maxdegree,9))
Etest = np.zeros((maxdegree,9))

# Create a matplotlib figure
fig, ax = plt.subplots()

for l in range(nlambdas):   
    i=0
    for degree in range(maxdegree): 
        j=0
        for train_inds, test_inds in kfold.split(x):
            
            X = DesignMatrix(x,y,degree+1)
            
            X_train = X[train_inds]
            z_train = z_scaled[train_inds]   
            X_test = X[test_inds]
            z_test = z_scaled[test_inds]
            
            z_fit, z_pred = LassoReg(X_train, X_test, z_train, z_test,lambdas[l])
            
            error_Kfold_test[i,j] = MSE(z_test,z_pred)
            error_Kfold_train[i,j] = MSE(z_train,z_fit)
            
            j+=1
            
        estimated_mse_Kfold_test[degree] = np.mean(error_Kfold_test[i,:])
        estimated_mse_Kfold_train[degree] = np.mean(error_Kfold_train[i,:])
        polydegree[degree] = degree+1
                
        i+=1

    Etest[:,l] = estimated_mse_Kfold_test
    Etrain[:,l] = estimated_mse_Kfold_train
    ax.plot(polydegree, estimated_mse_Kfold_test, label='%.0e' %lambdas[l])

plt.xlabel('Model complexity')    
plt.xticks(np.arange(1, len(polydegree)+1, step=1))  # Set label locations.
plt.ylabel('MSE')
plt.title('MSE Lasso regression for different lambdas (Kfold=10), pol1to10')

# Add a legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], title='lambda', loc='center right', bbox_to_anchor=(1.27, 0.5))

#Save figure
#plt.savefig("plots/Terrain/CV_Lasso_lambda.png",dpi=150, bbox_inches='tight')
plt.show()

#Compare train and test performance
plt.figure()
plt.plot(polydegree, Etrain[:,4], label = 'KFold train')
plt.plot(polydegree, Etest[:,4], label = 'KFold test')
plt.xlabel('Complexity')
plt.ylabel('mse')
plt.xticks(np.arange(1, maxdegree+1, step=1))  # Set label locations.
plt.legend()
plt.title('K-fold Cross Validation, k = 10, Lasso, lambda=0.01')
#plt.savefig("plots/Terrain/CV_Lasso_pol1to10.png",dpi=150)
plt.show()

#%%
#Create a heatmap with the error per nlambdas and polynomial degree

heatmap = sb.heatmap(Etest,annot=True, annot_kws={"size":7}, cmap="coolwarm", xticklabels=lambdas, yticklabels=range(1,maxdegree+1), cbar_kws={'label': 'Mean squared error'})
heatmap.invert_yaxis()
heatmap.set_ylabel("Complexity")
heatmap.set_xlabel("lambda")
heatmap.set_title("MSE heatmap, Cross Validation, kfold = {:}".format(k))
plt.tight_layout()
#plt.savefig("plots/Terrain/CV_Lasso_heatmap_pol1to10.png",dpi=150)
plt.show()

#%%
#Make some more Lasso plots
#For complexity=4, lambda=10^-1 (MSE_train=0.82)
deg=4
lmb=0.01

X1 = DesignMatrix(x,y,deg)
modelLasso = Lasso(lmb,fit_intercept=False)
modelLasso.fit(X1,terrain)
ytilde3 = modelLasso.predict(X1)

#Plot 3D Ridge
Plot3D(x,y,ytilde3)    
plt.title('Lasso, pol=4, lmd=0.01')
#plt.savefig("plots/Terrain/Map_3d_Lasso_pol4_lmb0_01.png", dpi=150)
plt.show()
