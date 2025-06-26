import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

import re
import nltk
import ssl

# Fix SSL certificate issue on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download stopwords
nltk.download("stopwords") 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0,len(df)):
    review = re.sub('[^a-zA-Z]', " ", df["Review"][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer() 
    all_stopwords = stopwords.words("english")
    all_stopwords.remove("not")
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)] 
    review = ' '.join(review) 
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1200)

X = cv.fit_transform(corpus).toarray()
y = df.iloc[:,-1].to_numpy()

#Naive Bayes

from sklearn.model_selection import train_test_split

X_test, X_train, y_test, y_train = train_test_split(X,y,test_size=0.2,random_state=28)

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,y_train)
y_pred = nb.predict(X_test) 

from sklearn.metrics import confusion_matrix, accuracy_score
acc = accuracy_score(y_test,y_pred)
print(acc)
import seaborn as sns

from sklearn.linear_model import LogisticRegression

regressor = LogisticRegression()
regressor.fit(X_train,y_train)
y_pred2 = regressor.predict(X_test)

c_matrix = confusion_matrix(y_test,y_pred2)
acc2 = accuracy_score(y_test,y_pred2)
print(acc2)
sns.heatmap(c_matrix,annot=True)
plt.show()
