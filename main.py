import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 


df = pd.read_csv("Data.csv")


x = df.iloc[:,:-1].values
y = df.iloc[:,-1]

print(x)