# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 22:22:37 2019

@author: belen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data set
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[: , 1].values

#splitting training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3 , random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

#visualising the training set result
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('salary vs Experience(training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()
        
#visualising the test set result
plt.scatter(X_test, y_test, color='green')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('salary vs Experience(test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()