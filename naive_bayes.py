import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Social_Network_Ads.csv")

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=69)

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train,y_train)

