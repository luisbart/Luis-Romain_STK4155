import numpy as np
from Functions import FrankeFunction, DesignMatrix, LinReg, MSE, R2, create_mini_batches, DesignMatrix2
from math import exp, sqrt
from random import random, seed
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline, Pipeline

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
#OLS with Ridge
nlambdas = 100
lambdas = np.logspace(-4, 4, nlambdas)   

MSERidgePredict = np.zeros(nlambdas)
MSELassoPredict = np.zeros(nlambdas)
R2RidgePredict = np.zeros(nlambdas)
R2LassoPredict = np.zeros(nlambdas)
MSERidgeTilde = np.zeros(nlambdas)
MSELassoTilde = np.zeros(nlambdas)
R2RidgeTilde = np.zeros(nlambdas)
R2LassoTilde = np.zeros(nlambdas)


for i in range(nlambdas):
    lmb = lambdas[i]
    # Make the fit using Ridge and Lasso
    RegRidge = linear_model.Ridge(lmb,fit_intercept=False)
    RegRidge.fit(X_train,y_train)
    RegLasso = linear_model.Lasso(lmb,fit_intercept=False)
    RegLasso.fit(X_train,y_train)
    # and then make the prediction
    ypredictRidge = RegRidge.predict(X_test)
    ypredictLasso = RegLasso.predict(X_test)
    ytildeRidge = RegRidge.predict(X_train)
    ytildeLasso = RegLasso.predict(X_train)
    # Compute the MSE and print it
    MSERidgePredict[i] = MSE(y_test,ypredictRidge)
    MSELassoPredict[i] =MSE(y_test,ypredictLasso)
    R2RidgePredict[i] = R2(y_test,ypredictRidge)
    R2LassoPredict[i] = R2(y_test,ypredictLasso)
    
    MSERidgeTilde[i] = MSE(y_train,ytildeRidge)
    MSELassoTilde[i] = MSE(y_train,ytildeLasso)
    R2RidgeTilde[i] = R2(y_train,ytildeRidge)
    R2LassoTilde[i] = R2(y_train,ytildeLasso)
    
    # print(lmb,RegRidge.coef_)
    # print(lmb,RegLasso.coef_)

# Now plot the results
plt.figure()
plt.plot(np.log10(lambdas), MSERidgePredict, 'r--', label = 'MSE Ridge Test')
plt.plot(np.log10(lambdas), MSERidgeTilde, 'r', label = 'MSE Ridge Train')
plt.plot(np.log10(lambdas), MSELassoPredict, 'b--', label = 'MSE Lasso Test')
plt.plot(np.log10(lambdas), MSELassoTilde, 'b', label = 'MSE Lasso Train')
plt.xlabel('log10(lambda)')
plt.ylabel('MSE')
plt.legend()
#plt.savefig("plots/Ridge_Lasso/MSE_vs_lambda_RIDGE_LASSO.png", dpi=150)
plt.show()

plt.figure()
plt.plot(np.log10(lambdas), R2RidgePredict, 'r--', label = 'R2 Ridge Test')
plt.plot(np.log10(lambdas), R2RidgeTilde, 'r', label = 'R2 Ridge Train')
plt.plot(np.log10(lambdas), R2LassoPredict, 'b--', label = 'R2 Lasso Test')
plt.plot(np.log10(lambdas), R2LassoTilde, 'b', label = 'R2 Lasso Train')
plt.xlabel('log10(lambda)')
plt.ylabel('R2')
plt.legend()
#plt.savefig("plots/Ridge_Lasso/R2_vs_lambda_RIDGE_LASSO.png", dpi=150)
plt.show()

#%% Gradient descent with Ridge
# Decide which values of lambda to use
nlambdas = 100
lambdas = np.logspace(-4, 4, nlambdas)

beta = np.random.randn(X_train.shape[1])
eta = 0.01
eps = [1]
i=0

def CostRidge(X_train, y_train, beta):
    return 2.0/n*X_train.T @ (X_train @ (beta)-y_train)+2*lambda*beta


for l in range(nlambdas):
    lmb = lambdas[l]
    while(eps[-1] >= 10**(-4)) :
        d = y_train.shape[0] 
        #y_train = np.reshape(y_train,(d,1))
        gradient = (2.0/d)*X_train.T @ (X_train @ beta - y_train)
        eps = np.append(eps,np.linalg.norm(gradient))
        beta -= eta*gradient
        i+=1
print('number of iterations for lmb=', lmb,' : ', i)

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

