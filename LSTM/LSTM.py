import pandas as pd
import numpy as np
import spacy
sp =  spacy.load('en')
import nltk 
nltk.download('wordnet')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras import metrics
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='english')

from collections import Counter


from keras import backend as K


'''
1. DATA PREPROCESSING
'''

def preprocess_data():
    
    #Import data
    file = open("spamData_sample.txt", "r+")

    #dataframe of spam data 
    attr = ['label','text']
    raw_data = pd.DataFrame(columns=attr)
    
    lines = file.readlines()
    
    for i in range(len(lines)):
        splittedline = lines[i].split("\t")
        raw_data.loc[i] = splittedline
    
    #Ham or spam as binary representation
    #full_df['label'].replace('ham', 1)
    #full_df['label'].replace('spam', 0)
#    labels = np.asarray(raw_data['label'])
#   texts = np.asarray(raw_data['text'])
    return raw_data    


df = preprocess_data()


def remove_stopwords(df):
    #remove stopwords in data
    stop_words = set(stopwords.words('english'))
    df['text_no_stopword'] = df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop_words]))
    return df

def stem_data(df):
    
    #Stemming in data, grouping data computing, computer, compute = comput
    for w in df['text']: 
        filtered_texts.append(stemmer.stem(w))    
    return filtered_texts


def lemmatize_text(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

df['text_lemmatized'] = df.text.apply(lemmatize_text)

def get_most_common_words(df,label,no_of_words):
    counter = Counter()
    c =Counter(" ".join(full_df[label]).split()).most_common(no_of_words) 
    return c



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
