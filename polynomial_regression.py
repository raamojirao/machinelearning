# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("/Users/raenug001c/Documents/Machine Learning/Polynomial Linear Regression/Position_Salaries.csv")
# Creating independent variable vector
x = dataset.iloc[:,1:2].values # this means take all rows and take all columns except the last column. Converting vector to matrix so put 1:2. x is good to be in matrix and y is good to be in vector
# Creating dependent variable vector
y = dataset.iloc[:,2].values # this means take all rows and take the last column

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# Fitting the Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(x,y, color = 'red')
plt.plot(x,lin_reg.predict(x), color = 'blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(x,y, color = 'red')
plt.plot(x,lin_reg2.predict(X_poly), color = 'blue')
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(6.5))