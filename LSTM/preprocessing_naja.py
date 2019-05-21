import pandas as pd
import numpy as np
import spacy
sp = spacy.load('en')
import nltk
import re
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='english')
# nltk.download('punkt') # todo: only needs to run ones
import os #used to look up file
from symspellpy.symspellpy import SymSpell, Verbosity  # import symspell module

# Import data
file = open("spamData.txt", "r+")

'''
1. DATA PREPROCESSING
'''

# count_re = re.sub(r'[^\w]', ' ', text) # remove special characters/regular expreseions


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


''' NAJA PREPROCESSING '''

#FINDING MISPELLED WORDS
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

test_for_mispelled = texts[:50]
texts_to_check_mispelled = test_for_mispelled
max_edit_distance_lookup = 1

for text in texts_to_check_mispelled:

    org_text = text
    text = text.lower() # since symspell is calculating uppercase words as mispelled, we need to change the string to all lowercase first
    # print(text)
    input_term = text

    # find number of special characters
    without_specials = re.sub('[^\^&*$!?,.%]+', '', text) # todo: solution for now is to manualy tell it what characters to look for - alternative: find all non-special characters and substract from original  text (re.findall('[\w]') only_specials = re.sub(r'\w', ' ', text)
    # print('special characters: ', len(without_specials))

    # find number of digits
    numbers_in_text = sum(c.isdigit() for c in text)
    # print("digits in text: ", numbers_in_text)

    # find longest word
    def find_longest_word(word_list):
        longest_word = ''
        longest_size = 0

        for word in word_list:
            if len(word) > longest_size:
                longest_word = word
                longest_size = len(word)
        return longest_word

    without_specials = re.sub(r'\W', ' ', text)
    words = without_specials
    word_list = words.split()
    longest_word = find_longest_word(word_list)

    # print("longest word: ", len(longest_word))

    #make tokens without special characters or digits
    strings_to_tokenizer = org_text
    strings_to_tokenizer = re.sub(r'\W', ' ', strings_to_tokenizer)
    strings_to_tokenizer = re.sub("^\d+\s|\s\d+\s|\s\d+$", '', strings_to_tokenizer)



    tokens = word_tokenize(strings_to_tokenizer)

    # find uppercase words todo: needs tokens to work
    def is_text_upper(user_string):
        is_upper = False
        char_counter = len(user_string)
        if char_counter == 1: # if word is just one letter we don't want it to be calculated as uppercase
            return False
        for char in user_string:
            if 'a' <= char <= 'z':
                is_upper = False
                return is_upper
            elif 'A' <= char <= 'Z': # update so it checks all charactesrs
                is_upper = True
        return is_upper

    for token in tokens:
        testString = is_text_upper(token)
        # print("uppercase: ", str(testString))


    # todo: needs our own toke to work with e.g. don't
    strings_to_tokenizer2 = text # gets lower case text string
    strings_to_tokenizer2 = re.sub('[\&*$!?,.%#;=]+', ' ', strings_to_tokenizer2) #todo doesnt know -
    strings_to_tokenizer2 = re.sub("^\d+\s|\s\d+\s|\s\d+$", ' ', strings_to_tokenizer2)# remove digits
    tokens2 = word_tokenize(strings_to_tokenizer2)

    no_of_unknown_words = 0
    print(text)
    for token in tokens2:
        results_lookup = sym_spell.lookup(token, Verbosity.TOP, max_edit_distance=2,ignore_token="\d{2}\w*\b|\d+\W+\d+\b|\d\w*\b|[!@£#$%^&*();,.?:{}/|<>]")
        for word in results_lookup:
            if word.distance > 0:
                no_of_unknown_words = no_of_unknown_words + 1
                print(token, " ", "{}, {}".format(word.term, word.distance)) # for testing
    print("number of unknown words: ", no_of_unknown_words)

    # look up a line of words
    ''' suggestions = sym_spell.lookup_compound(input_term, max_edit_distance_lookup)
    for word in suggestions:
        print("{}, {}".format(word.term, word.distance))
    '''
    print("*****")