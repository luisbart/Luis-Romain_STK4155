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
from autograd import grad

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

#%% Add RMSprop to Gradient descent with momentum
#?




#%% Add RMSprop to SGD with momentum

def CostOLS(y,X,theta):
    return np.sum((y-X @ theta)**2)

training_gradient = grad(CostOLS,2)

y_train = y_train.reshape(y_train.shape[0],1)

# Define parameters for Stochastic Gradient Descent
j = 0
eps = []
err = []
M = 20   #size of each minibatch
m = int(y.shape[0]/M) #number of minibatches
n_epochs = 50000 #number of epochs
eta = 0.01

# Guess for unknown parameters beta
beta = np.random.randn(X_train.shape[1],1)
# Value for parameter rho
rho = 0.99
# Including AdaGrad parameter to avoid possible division by zero
delta = 10**-7

# improve with momentum gradient descent
change = 0.0
delta_momentum = 0.3


for epoch in range(1,n_epochs+1):
    mini_batches = create_mini_batches(X_train,y_train,M)   
    Giter = np.zeros(shape=(X_train.shape[1], X_train.shape[1]))
    for minibatch in mini_batches:
        X_mini, z_mini = minibatch
        gradient = (2.0/M)*X_mini.T @ (X_mini @ beta - z_mini)

        	# Previous value for the outer product of gradients
        Previous = Giter       
	    # Accumulated gradient
        Giter +=gradient @ gradient.T
        	# Scaling with rho the new and the previous results
        Gnew = (rho*Previous+(1-rho)*Giter)
        	# Taking the diagonal only and inverting
        Ginverse = np.c_[eta/(delta+np.sqrt(np.diagonal(Gnew)))]
        # compute update
        new_change = np.multiply(Ginverse,gradient) + delta_momentum*change        
        beta -= new_change
        change = new_change
   
    err = np.append(err,MSE(y_train,X_train @ beta))


plt.yscale('log')
plt.plot(err)
 
print("Beta with SGD with momentum")
print(beta.T)
print("Training error")
print("MSE =",MSE(y_train,X_train @ beta))
print("R2 =",R2(y_train,X_train @ beta))
print("Test error")
print("MSE =",mean_squared_error(y_test,X_test @ beta))
print("R2 =",r2_score(y_test,X_test @ beta))


















