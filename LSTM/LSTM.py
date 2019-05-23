import pandas as pd
import numpy as np
import spacy
sp =  spacy.load('en')
import nltk 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras import metrics
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='english')


from keras import backend as K


#Import data
file = open("spamData_sample.txt", "r+")

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
#full_df['label'].replace('ham', 1)
#full_df['label'].replace('spam', 0)
labels = np.asarray(full_df['label'])
texts = np.asarray(full_df['text'])

df = full_df

'''
Splitting data into train, validation and test
50% train, 50% validation and test. 80% is validation and 20% is test

'''
'''
# we will use 20% of data as training atm, just to make quicker
training_samples = int(len(df) * .2)
validation_samples = int(len(df) - training_samples)

texts_train = texts[:training_samples]
labels_train = labels[:training_samples]
'''
filtered_texts =[]

#Different preprocessing tried: 
    
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
  
    
#Get tokens
tokenizer = Tokenizer(oov_token=True)
tokenizer.fit_on_texts(df['text'].values)

all_seq_as_indices = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(all_seq_as_indices)

'''
Fit_on_texts henter alle ord i alle beskeder, giver et index. 
texts_to_sequences opdeler alle beskeder i deres index i stedet i ord. nye ord får 
index 1. Se nedenstående eksempel, goooooogoskhdfkjh goooooogoskhd    er de nye ord.
test = ['he is goooooogoskhdfkjh goooooogoskhd    fine', 'i am fine too']

print(tokenizer.word_index)
print(all_seq_as_index)
print('se at skøre ord har 1-taller i nedenstående')
print(tokenizer.texts_to_sequences(test))
'''

vocabulary_size = len(tokenizer.word_index)+1 #Forstår ikke, har brugt tal fra lab10 og Yelp
embed_dim = 128 #lab10+Yelp. Måske ønsket antal dimensioner pr. objekt.
max_sequence = X.shape[1] #Max antal ord i en sætning
lstm_out = 196
batch_size = 32 # antal neuroner (del at træningssæt) i en kørsel. 

'''
Sequential() er model, der holder alle LSTM lag, (et lag er data i hver iteration)
Embedding() er laver positive tal der kommer ind i modellen til vektorer, tror noget normalisering
dropout dropper samples, hvis modellen overfittes. Jeg ved ikke hvorfor 0.2, det er bare fra Yelp g lab10
Optimizer = adam, klassisk optimizer. Lab10+yelp bruger begge denne
Begrundelse for loss = binary https://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/
'''

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#Clip removes values not within interval {0,1]
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon()) # epsilon is used for never divisions by 0
    
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """

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

model = Sequential()
model.add(Embedding(vocabulary_size, embed_dim,input_length = max_sequence))#, dropout = 0.2))
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
    epochs = 1, #antal gange hele træningsdatasættet gennemløbes
#    verbose = 2,
#    validation_data=(X_valid, Y_valid)
)
results.history

eval = model.evaluate(X_valid, Y_valid)#, verbose = 2)#, batch_size = batch_size)
#print("Score: %.2f" % (score))
#print("Validation Accuracy: %.2f" % (acc))


from sklearn.metrics import confusion_matrix

# Predicting the Test set results
y_pred = model.predict(X_train)


# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix

#from ONE (One hot encodings) to one vector of labels 
cm = confusion_matrix(Y_train.values.argmax(axis=1), y_pred.argmax(axis=1))
cm = confusion_matrix(Y_train.values.argmax(axis=1), y_pred.argmax(axis=1))
