# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 14:53:37 2018

@author: Monia
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset

dataset = pd.read_csv('Data.csv')

# Matrix of features (Matrix of independent variables) (three first columns) 
X=dataset.iloc[:, :-1].values

# Dependent variable vector
Y=dataset.iloc[:,3].values

# Missing data (1 - delete rows with missing data or 2 - replace missing data with average value from the whole column)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
# Replace missing data with the mean of the column
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data (replace text (categories) with number to allow math)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# Dummy encoding, because there is no numerical difference between countries (0,1,2) in equations make difference, but we dont want this. 
# Dummy encoding replace one number with the matrix with 0s and 1 in matching column
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into the trainig set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Standarisation and Normalization 
#Putting variables in the same range and the same scale (greater difference in salary than in age will make age variable unseen and not important)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# No need to fit to test set - is fitted to train already
X_test = sc_X.transform(X_test)

# No feature scaling for dependent variable here because it is categorical

