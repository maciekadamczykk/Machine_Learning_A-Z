import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 


df = pd.read_csv("Data.csv")


x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,1:])
x[:,1:] = imputer.transform(x[:,1:])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],remainder="passthrough")
x = np.array(ct.fit_transform(x))

print(x)

le = LabelEncoder()
y = le.fit_transform(y)

print(x)
print(y)