
#%%
#import Functions module
import sys 
import os
sys.path.append(os.path.abspath("C:/Users/luis.barreiro/Documents/GitHub/FYS_STK4155/Project1/code"))
import Functions as Func

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils import resample
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#%%
#Bias-variance analysis of the Franke function

#Bootstrap
n_bootstraps = 75

# Make data set. (take from previous exercise)
np.random.seed(2003)
n = 75
maxdegree = 5

x = np.random.uniform(0,1,n)
y = np.random.uniform(0,1,n)
z = Func.FrankeFunction(x, y)
z = z + np.random.normal(0,0.1,z.shape)

x = np.array(x).reshape(n,1)
y = np.array(y).reshape(n,1)
z = np.array(z).reshape(n,1)

x1 = np.hstack((x,y)).reshape(n,2)


error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
x_train, x_test, z_train, z_test = train_test_split(x1, z, test_size=0.2)


for degree in range(maxdegree):
    X_train = Func.DesignMatrix(x_train[:,0],x_train[:,1],degree+1)
    X_test = Func.DesignMatrix(x_test[:,0],x_test[:,1],degree+1)
    z_pred = np.zeros((z_test.shape[0], n_bootstraps))
    for i in range(n_bootstraps):
        x_, z_ = resample(X_train, z_train)
        z_fit, zpred, beta = Func.OLS(x_, X_test, z_)
        z_pred[:, i] = zpred.ravel()
  
    polydegree[degree] = degree+1
    error[degree] = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    bias[degree] = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance[degree] = np.mean( np.var(z_pred, axis=1, keepdims=True) )
    print('Polynomial degree:', degree)
    print('Error:', error[degree])
    print('Bias^2:', bias[degree])
    print('Var:', variance[degree])
    print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.xticks(np.arange(1, 6, step=1))  # Set label locations.
plt.xlabel('Model complexity')
plt.legend()
plt.savefig("C:/Users/luis.barreiro/Documents/GitHub/FYS_STK4155/Project1/Plots/Error_bias_var.png",dpi=150)
plt.show()
