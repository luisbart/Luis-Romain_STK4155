import numpy as np
from Functions import FrankeFunction, DesignMatrix, LinReg, MSE, R2
from math import exp, sqrt
from random import random, seed
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

#%%
#OLS on the Franke function
#Create data
#np.random.seed(2003)
n = 100
maxdegree = 2

x = np.random.rand(100,1)
y = 2.0+5*x*x+0.1*np.random.randn(100,1)

# x = np.array(x).reshape(n,1)
# y = np.array(y).reshape(n,1)

# x1 = np.hstack((x,y)).reshape(n,2)

# Split the data in test (80%) and training dataset (20%) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

X_train = DesignMatrix(x_train,x_train,maxdegree)
X_test = DesignMatrix(x_test,x_test,maxdegree)

#%%
#OLS with matrix inversion

y_fit, y_pred, betas = LinReg(X_train, X_test, y_train)

# print("Beta from matrice inversion")
# print(betas)

print("Training error OLS")
print("MSE =",MSE(y_train,y_fit))
print("R2 =",R2(y_train,y_fit))
    
print("Testing error OLS")
print("MSE =",MSE(y_test,y_pred))  
print("R2  =",R2(y_test,y_pred))

#%% OLS with scikit learn
model = Pipeline([('poly', PolynomialFeatures(degree=maxdegree)),('linear',\
              LinearRegression(fit_intercept=False))])
model = model.fit(y_train,y_train) 
Beta = model.named_steps['linear'].coef_


z_fit = model.predict(x_train)
z_pred = model.predict(x_test) 



print("Training error OLS skl")
print("MSE =",MSE(y_train,y_fit))
print("R2 =",R2(y_train,y_fit))
    
print("Testing error OLS skl")
print("MSE =",MSE(y_test,y_pred))  
print("R2  =",R2(y_test,y_pred))

#%% Gradient descent

#Fixed eta
beta = np.random.randn(X_train.shape[1],1)
eta=0.01
eps = [1]
i=0



while(eps[-1] >= 10**(-4)) :
    d = y_train.shape[0] 
    y_train = np.reshape(y_train,(d,1))
    gradient = (2.0/d)*X_train.T @ (X_train @ beta - y_train)
    eps = np.append(eps,np.linalg.norm(gradient))
    beta -= eta*gradient
    i+=1
    
print('number of iterations: ', i)

plt.plot(eps)


y_pred = X_test @ beta
y_fit = X_train @ beta


print("Beta with Gradient Descent")
print(beta)

print("Training error")
print("MSE =",MSE(y_train,y_fit))
print("R2 =",R2(y_train,y_fit))
    
print("Testing error")
print("MSE =",MSE(y_test,y_pred))
print("R2  =",R2(y_test,y_pred))
