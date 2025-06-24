import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("Salary_Data.csv")

x = df.iloc[:,:-1].values 
y = df.iloc[:,-1].values 


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#Visualising training set 
plt.scatter(x_train,y_train, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title("Sales vs Experience(Training set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#Visualising test set
plt.scatter(x_test,y_test, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title("Sales vs Experience(Test set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

print(regressor.predict([[10]]))