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

le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state= 1) 


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:,3:] = sc.fit_transform(x_train[:,3:])
x_test[:,3:] = sc.transform(x_test[:,3:])

print(x_train)
print(x_test)