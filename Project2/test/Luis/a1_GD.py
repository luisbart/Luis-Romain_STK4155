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
maxdegree = 8

x = np.random.rand(100)
y = 2.0+5*x*x + 0.5 * np.random.randn(100)

plt.scatter(x,y)



#  The design matrix as function of a given polynomial
def DesignMatrix2(x, maxdegree):
    X= np.ones((len(x),maxdegree+1)) 
    for i in range(1,maxdegree+1):
        X[:,i] = x**i
    return X


# Split the data in test (80%) and training dataset (20%) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#X_train = DesignMatrix(x_train,x_train,maxdegree)
#X_test = DesignMatrix(x_test,x_test,maxdegree)

#%%
#OLS with matrix inversion
TestError = np.zeros(maxdegree)
TrainError = np.zeros(maxdegree)
TestR2 = np.zeros(maxdegree)
TrainR2 = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
predictor = []

for degree in range(maxdegree):
    X_train = DesignMatrix2(x_train,degree+1)
    X_test = DesignMatrix2(x_test,degree+1)       
    y_fit, y_pred, betas = LinReg(X_train, X_test, y_train)
    predictor=np.append(predictor,betas)
    
    polydegree[degree] = degree+1    
    TestError[degree] = MSE(y_test, y_pred)
    TrainError[degree] = MSE(y_train, y_fit)
    TestR2[degree] = R2(y_test,y_pred)
    TrainR2[degree] = R2(y_train,y_fit)    

    
#Plots 
#MSE   
plt.plot(polydegree, TestError, label='Test sample')
plt.plot(polydegree, TrainError, label='Train sample')
plt.xlabel('Model complexity (degree)')
plt.ylabel('Mean Square Error')
plt.xticks(np.arange(1, maxdegree+1, step=1))  # Set label locations.
plt.legend()
plt.title('own OLS')
#plt.savefig("plots/OLS/MSE_vs_complexity.png", dpi=150)
plt.show()

#R2 score
plt.plot(polydegree, TestR2, label='Test sample')
plt.plot(polydegree, TrainR2, label='Train sample')
plt.xlabel('Model complexity')
plt.ylabel('R2 score')
plt.xticks(np.arange(1, maxdegree+1, step=1))  # Set label locations.
plt.legend()
plt.title('own OLS')
#plt.savefig("plots/OLS/R2_vs_complexity.png", dpi=150)
plt.show()

#%% OLS with scikit learn

TestError = np.zeros(maxdegree)
TrainError = np.zeros(maxdegree)
TestR2 = np.zeros(maxdegree)
TrainR2 = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
predictor = []

for degree in range(maxdegree):
    model = Pipeline([('poly', PolynomialFeatures(degree=maxdegree)),('linear',\
                  LinearRegression(fit_intercept=False))])
        
    X_train = DesignMatrix2(x_train,degree+1)
    X_test = DesignMatrix2(x_test,degree+1)      
    model = model.fit(X_train,y_train) 
    Beta = model.named_steps['linear'].coef_
    predictor=np.append(predictor,betas)
      
    y_fit = model.predict(X_train)
    y_pred = model.predict(X_test) 
    
    polydegree[degree] = degree+1
    TestError[degree] = MSE(y_test, y_pred)
    TrainError[degree] = MSE(y_train, y_fit)
    TestR2[degree] = R2(y_test,y_pred)
    TrainR2[degree] = R2(y_train,y_fit)  
    
    

#Plots 
#MSE   
plt.plot(polydegree, TestError, label='Test sample')
plt.plot(polydegree, TrainError, label='Train sample')
plt.xlabel('Model complexity (degree)')
plt.ylabel('Mean Square Error')
plt.xticks(np.arange(1, maxdegree+1, step=1))  # Set label locations.
plt.legend()
plt.title('Scikit')
#plt.savefig("plots/OLS/MSE_vs_complexity.png", dpi=150)
plt.show()

#R2 score
plt.plot(polydegree, TestR2, label='Test sample')
plt.plot(polydegree, TrainR2, label='Train sample')
plt.xlabel('Model complexity')
plt.ylabel('R2 score')
plt.xticks(np.arange(1, maxdegree+1, step=1))  # Set label locations.
plt.legend()
plt.title('Scikit')
#plt.savefig("plots/OLS/R2_vs_complexity.png", dpi=150)
plt.show()

#%% Gradient descent



# y_train = y_train.reshape(x_train.shape[0],1)
# y_test = y_test.reshape(x_test.shape[0],1)

TestError = np.zeros(maxdegree)
TrainError = np.zeros(maxdegree)
TestR2 = np.zeros(maxdegree)
TrainR2 = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
predictor = []

for degree in range(maxdegree):
    
    #Fixed eta
    eta=0.01
    eps = [1]
    i=0
    
    
    X_train = DesignMatrix2(x_train,degree+1)
    X_test = DesignMatrix2(x_test,degree+1) 
    
    beta = np.random.randn(X_train.shape[1],1)
        
    while(eps[-1] >= 10**(-4)) :
        d = y_train.shape[0] 
        y_train = np.reshape(y_train,(d,1))
        gradient = (2.0/d)*X_train.T @ (X_train @ beta - y_train)
        eps = np.append(eps,np.linalg.norm(gradient))
        beta -= eta*gradient
        i+=1
    
    y_pred = X_test @ beta
    y_fit = X_train @ beta

    polydegree[degree] = degree+1
    TestError[degree] = MSE(y_test, y_pred)
    TrainError[degree] = MSE(y_train, y_fit)
    TestR2[degree] = R2(y_test,y_pred)
    TrainR2[degree] = R2(y_train,y_fit) 
    
    print('number of iterations for ', degree+1, 'degrees: ', i)

#Plots 
#MSE   
plt.plot(polydegree, TestError, label='Test sample')
plt.plot(polydegree, TrainError, label='Train sample')
plt.xlabel('Model complexity (degree)')
plt.ylabel('Mean Square Error')
plt.xticks(np.arange(1, maxdegree+1, step=1))  # Set label locations.
plt.legend()
plt.title('Scikit')
plt.savefig("GD_MSE_vs_complexity.png", dpi=150)
plt.show()

#R2 score
plt.plot(polydegree, TestR2, label='Test sample')
plt.plot(polydegree, TrainR2, label='Train sample')
plt.xlabel('Model complexity')
plt.ylabel('R2 score')
plt.xticks(np.arange(1, maxdegree+1, step=1))  # Set label locations.
plt.legend()
plt.title('Scikit')
plt.savefig("GD_R2_vs_complexity.png", dpi=150)
plt.show()    






# print("Beta with Gradient Descent")
# print(beta)

# print("Training error")
# print("MSE =",MSE(y_train,y_fit))
# print("R2 =",R2(y_train,y_fit))
    
# print("Testing error")
# print("MSE =",mean_squared_error(y_test,y_pred))
# print("R2  =",r2_score(y_test,y_pred))


#%%
print("Testing error")
print("MSE =",MSE(y_test,y_pred))
print("R2  =",R2(y_test,y_pred))




