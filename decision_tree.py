import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('Position_Salaries.csv')

X = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=42)

regressor.fit(X,y) 

print(regressor.predict([[6.5]]))