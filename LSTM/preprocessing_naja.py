import pandas as pd
import numpy as np
import spacy
sp = spacy.load('en')
import nltk

from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='english')

import os
from symspellpy.symspellpy import SymSpell  # import the module

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
'''
# finding misspelled words
# maximum edit distance per dictionary precalculation
max_edit_distance_dictionary = 2
prefix_length = 7
# create object
sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)

# create dictionary using corpus.txt
if not sym_spell.create_dictionary("C:/users/najam/PycharmProjects/dataScienceGamesCourse/frequency_dictionary_en_82_765.txt"):
    print("Corpus file not found")
#check how dictionary looks
#for key, count in sym_spell.words.items():
#    print("{} {}".format(key, count))
'''
'''
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
        '''

import os
from symspellpy.symspellpy import SymSpell, Verbosity  # import the module

# maximum edit distance per dictionary precalculation
max_edit_distance_dictionary = 3
prefix_length = 7
# create object
sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
# load dictionary
dictionary_path = os.path.join(os.path.dirname('C:/users/najam/PycharmProjects/dataScienceGamesCourse/'),
                                   "frequency_dictionary_en_82_765.txt")
term_index = 0  # column of the term in the dictionary text file
count_index = 1  # column of the term frequency in the dictionary text file
if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
    print("Dictionary file not found")

#what to look up
input_term = ("boi giirl trouble")

# create origional tokenized string to compare with later
original_string = sym_spell.lookup_compound(input_term,
                                            0)
for word in original_string:
    print(word.term)

max_edit_distance_lookup = 3 # max edit distance per lookup (per single word, not per whole input string)
suggestions = sym_spell.lookup_compound(input_term,
                                            max_edit_distance_lookup)

# display suggestion term, edit distance, and term frequency
for word in suggestions:
    print("{}, {}, {}".format(word.term, word.distance,word.count))