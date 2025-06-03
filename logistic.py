import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


df = pd.read_csv('Social_Network_Ads.csv')

X = df.iloc[:,:-1].values 
y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

classifier = LogisticRegression(random_state=42)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test,y_pred)

cf_matrix = confusion_matrix(y_test,y_pred)

sns.heatmap(cf_matrix, annot=True, cmap="Blues")
plt.show()
