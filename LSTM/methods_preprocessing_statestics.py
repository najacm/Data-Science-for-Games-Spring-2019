import pandas as pd
import numpy as np
import spacy
sp = spacy.load('en')
import re
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='english')
# nltk.download('punkt') # todo: only needs to run ones
import os
from symspellpy.symspellpy import SymSpell, Verbosity  # import symspell module

# Import data
file = open("spamData.txt", "r+")

'''
1. DATA PREPROCESSING
'''

attr = ['label','text']
full_df = pd.DataFrame(columns=attr)
lines = file.readlines()
for i in range(len(lines)):
    splittedline = lines[i].split("\t")
    full_df.loc[i] = splittedline
full_df['label'].replace('ham', 1)
full_df['label'].replace('spam', 0)
labels = np.asarray(full_df['label'])
texts = np.asarray(full_df['text'])

df = full_df

'''
Splitting data into train, validation and test
50% train, 50% validation and test. 80% is validation and 20% is test
'''

# we will use 20% of data as training atm, just to make quicker
training_samples = int(len(df) * .2)
validation_samples = int(len(df) - training_samples)

texts_train = texts[:training_samples]
labels_train = labels[:training_samples]




''' STATISTICAL PREPROCESSING '''
''' THE FOLLOWING METHODS CAN BE USED
find_longest_word()
find_digits_sum()
find_special_char_sum()
find_uppercase()
find_unknown_words()
find_length_of_text()
find_no_of_words()'''

df_extra_features = df

''' FIND LONGEST WORD '''

def find_longest_word():
    df_extra_features['longest_word'] = '0'
    row_counter = 0
    for text in df_extra_features['text']:
        def find_longest_word(word_list):
            longest_word = ''
            longest_size = 0

            for word in word_list:
                if len(word) > longest_size:
                    longest_word = word
                    longest_size = len(word)
            return longest_word

        without_specials = re.sub(r'\W', ' ', text) # exclude special characters, else punctiation will be part of the word
        words = without_specials
        word_list = words.split()
        longest_word = find_longest_word(word_list)
        df_extra_features.loc[row_counter, 'longest_word'] = len(longest_word)
        row_counter = row_counter + 1


''' FIND SUM OF DIGITS '''
def find_digits_sum():
    row_counter = 0
    df_extra_features['sum_of_digits'] = '0'

    for text in df_extra_features['text']:
        digits_in_text = sum(c.isdigit() for c in text)
        df_extra_features.loc[row_counter, 'sum_of_digits'] = digits_in_text
        row_counter = row_counter + 1


''' FIND SUM OF SPECIAL CHARACTERS'''
def find_special_char_sum():
    df_extra_features['sum_of_special_char'] = '0'
    row_counter = 0
    for text in df_extra_features['text']:
        only_specials = re.sub('[^\^&*$!?,.%()+#-]', '', text)  # todo: solution for now is to manualy tell it what characters to look for
        df_extra_features.loc[row_counter, 'sum_of_special_char'] = len(only_specials)
        row_counter = row_counter + 1

'''FIND IF STRING CONTAINS UPPERCASE WORDS'''
def find_uppercase():
    df_extra_features['has_upper_case'] = False
    row_counter = 0
    def is_token_upper(token):
        token_is_upper = False
        char_counter = len(token)
        if char_counter == 1:  # if word is just one letter we don't want it to be calculated as uppercase
            token_is_upper = False
            return token_is_upper
        for char in token:
            if 'a' <= char <= 'z':
                token_is_upper = False
                return False
            elif 'A' <= char <= 'Z':
                token_is_upper = True
        return token_is_upper


    def contains_upper(tokens):

        for token in tokens:
            temp_containts_upper = is_token_upper(token)
            if temp_containts_upper:
                return True
        return False

    for text in df_extra_features['text']:
        text = re.sub(r'\W', ' ', text)
        tokens = word_tokenize(text)
        text_contains = contains_upper(tokens)

        if text_contains:
            df_extra_features.loc[row_counter, 'has_upper_case'] = True
        row_counter = row_counter + 1

'''FIND NUMBER OF MISPELLED WORDS IN TEXT (ACCORDING TO SYMSPELL)'''     # todo: needs our own toke to work with e.g. don't
def find_unknown_words():
    # todo: egennavne skal optimalt set ikke med her!!
    df_extra_features['no_of_unknown_words'] = 0

    # maximum edit distance per dictionary precalculation # todo: not understanding
    max_edit_distance_dictionary = 3
    prefix_length = 7 # unknown

    sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)  # create symspell object

    # load dictionary
    dictionary_path = os.path.join(os.path.dirname('C:/Users/najam/PycharmProjects/dataScienceGamesCourse/Data-Science-for-Games-Spring-2019/LSTM/'),
                                   "frequency_dictionary_en_82_765.txt") # todo: update path or make general

    term_index = 0  # column of the term in the dictionary text file
    count_index = 1  # column of the term frequency in the dictionary text file
    if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
        print("Dictionary file not found")

    row_counter = 0

    for text in df_extra_features['text']:
        text = text.lower() # gets lower case text string
        text = re.sub(r'\W', ' ', text)  # todo doesnt know '-' or '. needs general thing that will remove all special chars
        text = re.sub("^\d+\s|\s\d+\s|\s\d+$", ' ', text)  # remove digits, only removes digits that are not connected to letters
        tokens2 = word_tokenize(text)

        no_of_unknown_words = 0
        for token in tokens2:
            char_counter = len(token)
            if char_counter > 3: # should only look up is token is 3 or more chars
                results_lookup = sym_spell.lookup(token, Verbosity.TOP, max_edit_distance=2,
                                              ignore_token="\d{2}\w*\b|\d+\W+\d+\b|\d\w*\b|[!@£#$%^&*();,.?:{}/|<>]")
                for word in results_lookup:
                    if word.distance > 0:
                        no_of_unknown_words = no_of_unknown_words + 1

        df_extra_features.loc[row_counter, 'no_of_unknown_words'] = no_of_unknown_words
        row_counter = row_counter + 1

def find_length_of_text(with_space = True):
    df_extra_features['length_of_text'] = 0
    row_counter = 0

    for text in df_extra_features['text']:
        if not with_space:
            text = re.sub(' ', '', text)
        char_counter = 0
        for char in text:
            char_counter = char_counter + 1
        df_extra_features.loc[row_counter, 'length_of_text'] = char_counter
        row_counter = row_counter + 1

def find_no_of_words():
    df_extra_features['no_of_words'] = 0
    row_counter = 0

    for text in df_extra_features['text']:
        text = re.sub(r'\W', ' ', text)  # remove special characters
        text = re.sub("^\d+\s|\s\d+\s|\s\d+$", ' ', text)  # remove digits
        word_counter = 0
        tokens = word_tokenize(text)
        for token in tokens:
            word_counter = word_counter + 1
        df_extra_features.loc[row_counter, 'word_counter'] = word_counter
        row_counter = row_counter + 1
