import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("50_Startups.csv")

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

un = df["State"].unique()

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-1])], remainder="passthrough")
x = np.array(ct.fit_transform(x))

print(x)
print(y)