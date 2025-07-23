import numpy as np
import pandas as pd
import tensorflow as tf

df = pd.read_csv("Churn_Modelling.csv")

X = df.iloc[:,2:-1]
y = df.iloc[:,-1]
print(X.head)
print(y.head)
