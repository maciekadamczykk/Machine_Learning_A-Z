import numpy as np
import pandas as pd

df = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

print(df.head())

import re
import nltk 
nltk.download("stopwords") 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer