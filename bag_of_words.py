import numpy as np
import pandas as pd

df = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

import re
import nltk 
nltk.download("stopwords") 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0,len(df)):
    review = re.sub('[^a-zA-Z]', " ", df["Review"][i])
    review = review.lower()
    review = review.split()
