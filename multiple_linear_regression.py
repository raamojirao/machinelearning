# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("/Users/raenug001c/Documents/Machine Learning/Multiple Linear Regression/50_Startups.csv")
# Creating independent variable vector
x = dataset.iloc[:,:-1].values # this means take all rows and take all columns except the last column
# Creating dependent variable vector
y = dataset.iloc[:,4].values # this means take all rows and take the last column

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:,3] = labelencoder_X.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

# Avoiding the dummy variable trap
x = x[:, 1:]

# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)# 20% test size

# Fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backwward Elimination
import statsmodels.formula.api as sm
# In multiple regression the formula is y = b0 + b1x1 +...
# This library does not include constant b0 so we should add constant (ones)
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1) # (50,1) is the arguement. Where 50 is rows and 1 is column

#OLS class is ordinary least squares
x_opt = x[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()
# Removing index 4 as it has the highest P Value
x_opt = x[:, [0,1,2,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()
# Removing index 1 as it has the highest P Value
x_opt = x[:, [0,2,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()
# Removing index 2 as it has the highest P Value
x_opt = x[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()
# Removing index 5 as it has the highest P Value
x_opt = x[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()