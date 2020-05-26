# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:10:29 2019

@author: belen
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[: , 1:2].values
y = dataset.iloc[: , 2].values

#Fitting linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X , y)


#Fitting polynomial linear regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly , y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#plot linear regression
plt.scatter(X, y, color ='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (linear regression)')
plt.label('Positionlevel')
plt.ylabel('salary')
plt.show()


#visualize polynomial regresssion
plt.scatter(X, y, color ='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth or Bluff (polynomial regression)')
plt.label('Positionlevel')
plt.ylabel('salary')
plt.show()

#visualising at 0.1 resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color ='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth or Bluff (polynomial regression)')
plt.label('Positionlevel')
plt.ylabel('salary')
plt.show()
 
#predict using the linear regression
lin_reg.predict([[6.5]])

#predict using the polynomial regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))