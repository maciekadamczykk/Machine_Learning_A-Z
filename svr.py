import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Position_Salaries.csv')

X = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values
print(X)
print(y)
y = y.reshape(len(y),1)
print(y)

sc = StandardScaler()
sc2 = StandardScaler()
X = sc.fit_transform(X)
y = sc2.fit_transform(y)

print(X)
print(y)