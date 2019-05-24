import pandas as pd
import numpy as np
import spacy
sp =  spacy.load('en')
import nltk 
#nltk.download('wordnet')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils import to_categorical
from keras import metrics
from keras.layers import concatenate
from keras.layers import Input
from keras.layers import Bidirectional


from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='english')
from collections import Counter
from keras import backend as K

from sklearn.metrics import confusion_matrix

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
import csv



def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#Clip removes values not within interval {0,1]
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon()) # epsilon is used for never divisions by 0
    
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#Clip removes values not within interval {0,1]
    actual_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (actual_positives + K.epsilon()) # epsilon is used for never divisions by 0

    return recall 

def f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#Clip removes values not within interval {0,1]
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon()) 
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#Clip removes values not within interval {0,1]
    actual_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (actual_positives + K.epsilon()) # epsilon is used for never divisions by 0
    
    f1 = 2*(precision*recall)/(precision+recall)
    return f1

def meta_model():
        
    csv_file = '/Users/MasterWillis/ownCloud/ITU/gitDS/Data-Science-for-Games-Spring-2019/LSTM/metadata_actual.csv'
    csv_reader = csv.reader(csv_file, delimiter=',')    
    meta_df= pd.read_csv(csv_file)

    #attr = ['label','longest_word', 'no_of_words']
    
    df_ham = meta_df.loc[meta_df['label']=='ham']
    df_spam = meta_df.loc[meta_df['label']=='spam']
    length = len(df_spam)
    df_ham = df_ham[:length] 
    df_concat = pd.concat([df_ham, df_spam])
    meta_df = df_concat
    
    #df2 = df_meta[attr]
    meta_X = np.asarray(meta_df[['longest_word','no_of_words','length_of_text','sum_of_digits', 'has_upper_case']])   
    print(meta_X)
#    meta_X = np.asarray(meta_df[['has_upper_case','length_of_text']])   
    
    meta_vocabulary_size = np.max(meta_X)+1 #len(np.unique(meta_X))+3 #Forstår ikke, har brugt tal fra lab10 og Yelp  
    meta_embed_dim = 128 #lab10+Yelp. Måske ønsket antal dimensioner pr. objekt.
    meta_max_sequence = meta_X.shape[1] #Max antal ord i en sætning
    meta_lstm_out = 196
    meta_batch_size = 32 # antal neuroner (del at træningssæt) i en kørsel. 
    
    meta_model = Sequential()
    meta_model.add(Embedding(meta_vocabulary_size, meta_embed_dim,input_length = meta_max_sequence))#, dropout = 0.2))
    meta_model.add(LSTM(meta_lstm_out, dropout = 0.2))
    meta_model.add(Dense(2, activation='sigmoid'))
    meta_model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy', precision, recall, f1])#model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(meta_model.summary())
    
    
    #FEED ACTUAL DATA TO MODEL
    meta_Y = pd.get_dummies(meta_df['label'].values)
    meta_X_train, meta_X_valid, meta_Y_train, meta_Y_valid = train_test_split(meta_X,meta_Y, test_size = 0.50, random_state = 36)
    
    results = meta_model.fit(
        meta_X_train,
        meta_Y_train,
        batch_size = meta_batch_size,
        epochs = 5 
    )
    
    # Predicting the Test set results
    meta_y_pred = meta_model.predict(meta_X_train)
    
    #from ONE (One hot encodings) to one vector of labels 
    
    meta_cm = confusion_matrix(meta_Y_train.values.argmax(axis=1), meta_y_pred.argmax(axis=1))
     
    print(meta_cm)
    
    return meta_model

    
#meta_branch= meta_model()









'''
1. DATA PREPROCESSING
'''


def preprocess_data():
    
    #Import data
    file = open("spamData.txt", "r+")

    #dataframe of spam data 
    attr = ['label','text']
    raw_data = pd.DataFrame(columns=attr)
    
    lines = file.readlines()
    
    for i in range(len(lines)):
        splittedline = lines[i].split("\t")
        raw_data.loc[i] = splittedline
    
    df_ham = raw_data.loc[raw_data['label']=='ham']
    df_spam = raw_data.loc[raw_data['label']=='spam']
    length = len(df_spam)
    df_ham = df_ham[:length] 
    df_concat = pd.concat([df_ham, df_spam])
    return df_concat     


def init_model(df):
   
    #Get tokens
    tokenizer = Tokenizer(oov_token=True)
    tokenizer.fit_on_texts(df['text'].values) 
    all_seq_as_indices = tokenizer.texts_to_sequences(df['text'])
    X = pad_sequences(all_seq_as_indices)
    
    
    vocabulary_size = len(tokenizer.word_index)+1 #Forstår ikke, har brugt tal fra lab10 og Yelp
    embed_dim = 128 #lab10+Yelp. Måske ønsket antal dimensioner pr. objekt.
    max_sequence = X.shape[1] #Max antal ord i en sætning
    lstm_out = 196
    batch_size = 32 # antal neuroner (del at træningssæt) i en kørsel. 
    print(vocabulary_size ,   ' <- vo  '  , max_sequence )
    model = Sequential()
    model.add(Embedding(vocabulary_size, embed_dim))#, dropout = 0.2))
    model.add(LSTM(lstm_out, dropout = 0.2))
    model.add(Dense(2,activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy', precision, recall])#model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
  
    #FEED ACTUAL DATA TO MODEL
    Y = pd.get_dummies(df['label'].values)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size = 0.50, random_state = 36)
   
    
    results = model.fit(
        X_train,
        Y_train,
        batch_size =batch_size,
        epochs = 6, #antal gange hele træningsdatasættet gennemløbes
    #    verbose = 2,
    #    validation_data=(X_valid, Y_valid)
    )

        
    '''# Predicting the Test set results
    y_pred = model.predict(X_train)
    
    
    # Creating the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    
    #from ONE (One hot encodings) to one vector of labels 
    cm = confusion_matrix(Y_train.values.argmax(axis=1), y_pred.argmax(axis=1))
'''
    
    
    
    
    return model

def split_data(df):
        
    #FEED ACTUAL DATA TO MODEL
    Y = pd.get_dummies(df['label'].values)
#    X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size = 0.50, random_state = 36)
    return train_test_split(X,Y, test_size = 0.50, random_state = 36)

'''results = model_single.fit(
    X_train,
    Y_train,
    batch_size =batch_size,
    epochs = 1, #antal gange hele træningsdatasættet gennemløbes
#    verbose = 2,
#    validation_data=(X_valid, Y_valid)
)'''

df = preprocess_data()
meta_branch= init_model(df)


#data = preprocess_data()
#nlp_branch= init_model(data)
#
#combined = concatenate([meta_branch.output, nlp_branch.output])
#
## our model will accept the inputs of the two branches and
## then output a single value
##model = Model(inputs=[meta_branch.input, nlp_branch.input], outputs=z)
#LSTM_output = LSTM(128,dropout=0.2)(combined)
#model = Model(input=[meta_branch.input, nlp.input],output=LSTM_output)








#results.history

#eval = model_single.evaluate(X_valid, Y_valid)#, verbose = 2)#, batch_size = batch_size)
#print("Score: %.2f" % (score))
#print("Validation Accuracy: %.2f" % (acc))

'''
from sklearn.metrics import confusion_matrix

# Predicting the Test set results
y_pred = model.predict(X_train)


# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix

#from ONE (One hot encodings) to one vector of labels 
cm = confusion_matrix(Y_train.values.argmax(axis=1), y_pred.argmax(axis=1))
cm = confusion_matrix(Y_train.values.argmax(axis=1), y_pred.argmax(axis=1))'''
