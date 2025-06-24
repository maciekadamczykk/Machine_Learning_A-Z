import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('Position_Salaries.csv')

X = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values


lin_reg = LinearRegression()
lin_reg.fit(X,y)

pol_reg = PolynomialFeatures(degree=4)
X_poly = pol_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

plt.title("Polynomial Regression")
plt.scatter(X,y, color="red")
plt.plot(X, lin_reg_2.predict(X_poly), color="blue")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

print(lin_reg_2.predict(pol_reg.fit_transform([[6.5]])))