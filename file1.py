# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 21:55:48 2019

@author: belen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,3].values

#managing missing data using the mean strategy

from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = 'np.nan' , strategy = 'mean', fill_value = 'None', verbose = 0)
missingvalues = missingvalues.fit(X[:,1:3])
X[:,1:3] = missingvalues.transform(X[:,1:3])