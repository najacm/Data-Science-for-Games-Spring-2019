import pandas as pd
import numpy as np
import spacy
sp = spacy.load('en')
import nltk

from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='english')


# Import data
file = open("spamData.txt", "r+")

'''
1. DATA PREPROCESSING
'''


#dataframe of spam data
attr = ['label','text']
full_df = pd.DataFrame(columns=attr)

lines = file.readlines()


for i in range(len(lines)):
    splittedline = lines[i].split("\t")
    full_df.loc[i] = splittedline

#Ham or spam as binary representation
full_df['label'].replace('ham', 1)
full_df['label'].replace('spam', 0)
labels = np.asarray(full_df['label'])
texts = np.asarray(full_df['text'])

df = full_df

''''''
'''
Splitting data into train, validation and test
50% train, 50% validation and test. 80% is validation and 20% is test

''''''
'''
# we will use 20% of data as training atm, just to make quicker
training_samples = int(len(df) * .2)
validation_samples = int(len(df) - training_samples)

texts_train = texts[:training_samples]
labels_train = labels[:training_samples]

filtered_texts =[]

# Different preprocessing tried:

#remove stopwords in data
stop_words = set(stopwords.words('english'))
for w in texts:
    if w not in stop_words:
        filtered_texts.append(w)


#Stemming in data, grouping data computing, computer, compute = comput
for w in texts:
    filtered_texts.append(stemmer.stem(w))


#Lemmatizing in data, grouping data with actual words
for text in texts:
    sentence = sp(text)
    for word in sentence:
        filtered_texts.append(word.lemma_)
