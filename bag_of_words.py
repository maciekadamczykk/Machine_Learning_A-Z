import numpy as np
import pandas as pd

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

cv = CountVectorizer(max_features=1500)

X = cv.fit_transform(corpus).toarray()
y = df.iloc[:,-1].to_numpy()


print(len(X[0]))