# -*- coding: utf-8 -*-

#%%
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from Functions import FrankeFunction

#%%
# Plot Franke function without noise

#Create data
x = np.arange(0, 1, 0.01)
y = np.arange(0, 1, 0.01)

#Apply Franke function
fig = plt.figure()
ax = fig.gca(projection='3d')

x, y = np.meshgrid(x,y)

noise = (np.random.normal(0,0.1,len(x)*len(y))).reshape(len(x),len(y))
z= FrankeFunction(x, y)
znoise = z+noise

# Plot the surface without noise
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("plots/FrankeFunction/FrankeFunction.png", dpi=150)
plt.show()




#%%
# Plot the surface with noise
fig = plt.figure()
ax = fig.gca(projection='3d')

surf2 = ax.plot_surface(x, y, znoise, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("plots/FrankeFunction/FrankeFunction_noise.png", dpi=150)
plt.show()

