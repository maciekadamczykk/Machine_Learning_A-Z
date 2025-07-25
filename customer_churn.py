import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Churn_Modelling.csv")

X = df.iloc[:,3:-1].to_numpy()
y = df.iloc[:,-1].to_numpy()

print(X)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[1,2])],remainder='passthrough')
X = np.array(ct.fit_transform(X)) 
print(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train) 

ann = tf.keras.models.Sequential()
#First layer
ann.add(tf.keras.layers.Dense(units=8,activation='relu'))
#Second layer 
ann.add(tf.keras.layers.Dense(units=8,activation='relu'))
#Output layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

ann.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#Training ann
ann.fit(X_train,y_train,batch_size = 32, epochs=300)