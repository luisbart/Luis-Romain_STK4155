import numpy as np
from Functions import FrankeFunction, DesignMatrix, LinReg, MSE, R2, create_mini_batches, DesignMatrix2
from math import exp, sqrt
from random import random, seed
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

#%%
#Create data
#np.random.seed(2003)
n = 100
maxdegree = 2

x = np.random.rand(100)
y = 2.0+5*x*x + 0.6 * np.random.randn(100)

plt.scatter(x,y)

# Split the data in test (80%) and training dataset (20%) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

X_train = DesignMatrix2(x_train,maxdegree)
X_test = DesignMatrix2(x_test,maxdegree)

#%%
#OLS with matrix inversion
   
y_fit, y_pred, betas = LinReg(X_train, X_test, y_train)

print("Beta from matrice inversion")
print(betas)
print("Training error")
print("MSE =",MSE(y_train,y_fit))
print("R2 =",R2(y_train,y_fit))  
print("Testing error") 
print("MSE =",mean_squared_error(y_train,y_fit))
print("R2 =",r2_score(y_train,y_fit))


#%% OLS with scikit learn
model = LinearRegression(fit_intercept=False)        
model = model.fit(X_train,y_train) 
Beta = model.coef_
      
y_fit = model.predict(X_train)
y_pred = model.predict(X_test) 
    
print('Betas for Scikit:\n',Beta)

print("Training error")
print("MSE =",MSE(y_train,y_fit))
print("R2 =",R2(y_train,y_fit))  

print("Testing error") 
print("MSE =",MSE(y_test,y_pred))
print("R2  =",R2(y_test,y_pred))
#%% Gradient descent
beta = np.random.randn(X_train.shape[1])
eta = 0.01
eps = [1]
i=0

while(eps[-1] >= 10**(-4)) :
    d = y_train.shape[0] 
    #y_train = np.reshape(y_train,(d,1))
    gradient = (2.0/d)*X_train.T @ (X_train @ beta - y_train)
    eps = np.append(eps,np.linalg.norm(gradient))
    beta -= eta*gradient
    i+=1
print('number of iterations: ', i)

y_pred = X_test @ beta
y_fit = X_train @ beta

print("\nBeta with Gradient Descent")
print(beta)

print("Training error")
print("MSE =",MSE(y_train,y_fit))
print("R2 =",R2(y_train,y_fit))
    
print("Testing error")
print("MSE =",MSE(y_test,y_pred))
print("R2  =",R2(y_test,y_pred))

#%% SGD
j = 0
eps = []
M = 20   #size of each minibatch
m = int(y.shape[0]/M) #number of minibatches
n_epochs = 50000 #number of epochs

y_train = y_train.reshape(y_train.shape[0],1)
beta = np.random.randn(X_train.shape[1],1)

for epoch in range(1,n_epochs+1):
    mini_batches = create_mini_batches(X_train,y_train,M)   
    for minibatch in mini_batches:
        X_mini, z_mini = minibatch
        gradient = (2.0/M)*X_mini.T @ (X_mini @ beta - z_mini)
        beta -= eta*gradient
        if (np.linalg.norm(gradient)!= 0):
            eps = np.append(eps, np.linalg.norm(gradient))
    j+=1


print("Beta with SGD")
print(beta.T)
print("Training error")
print("MSE =",MSE(y_train,X_train @ beta))
print("R2 =",R2(y_train,X_train @ beta))
print("Test error")
print("MSE =",mean_squared_error(y_test,X_test @ beta))
print("R2 =",r2_score(y_test,X_test @ beta))
    

mean_squared_error, r2_score

