import pandas as pd
import csv
import numpy as np
import spacy

from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords

def read_meta_data():
    csv_file = '/Users/MasterWillis/ownCloud/ITU/gitDS/Data-Science-for-Games-Spring-2019/LSTM/metadata_actual.csv'
    csv_reader = csv.reader(csv_file, delimiter=',')    
    meta_df= pd.read_csv(csv_file)

    #attr = ['label','longest_word', 'no_of_words']
    
    df_ham = meta_df.loc[meta_df['label']=='ham']
    df_spam = meta_df.loc[meta_df['label']=='spam']
    length = len(df_spam)
    df_ham = df_ham[:length] 
    meta_df = pd.concat([df_ham, df_spam])
    return meta_df

def read_nlp_data():
    file = open("spamData.txt", "r+")
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
    nlp_df = pd.concat([df_ham, df_spam])
    return nlp_df 




def remove_stopwords(df, label):
    stop_words = set(stopwords.words('english'))
    df['text_no_stopword'] = df[label].apply(lambda x: ' '.join([item for item in x.split() if item not in stop_words]))
    return df
'''
def stem_text(text):

    sp =  spacy.load('en')
    sentence = sp(text)
    stemmer = SnowballStemmer(language='english')
    stemmed_text = [stemmer.stem(w) for w in sentence]
    return stemmed_text 
'''

def lemmatize_text(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
    return lemmatized_text 


#nlp_df['text_lemmatized'] = nlp_df.text.apply(lemmatize_text)
#nlp_df['text_stemmed'] = nlp_df.text.apply(stem_text)

#nlp_df = remove_stopwords(nlp_df, 'text')

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



def train_valid_split(nlp_df, meta_df):
    no_of_samples = len(nlp_df)
    labels = np.asarray(nlp_df['label'])
    texts = np.asarray(nlp_df['text'])
    meta = np.asarray(meta_df[['longest_word','no_of_words','length_of_text','sum_of_digits', 'has_upper_case']])
    
    np.random.seed(42)
    indices = np.arange(no_of_samples)
    np.random.shuffle(indices)
    
    train_size = int(0.5 * no_of_samples)
    valid_size = int(no_of_samples - train_size)
    
    
    texts = texts[indices]
    toke = Tokenizer(oov_token=True)
    toke.fit_on_texts(texts)
    texts_as_indices = toke.texts_to_sequences(texts)
    padded_texts_as_indices = pad_sequences(texts_as_indices)
    X_nlp_train = padded_texts_as_indices[:train_size]
    X_nlp_valid = padded_texts_as_indices[train_size:]
    
    meta = meta[indices]
    X_meta_train = meta[:train_size]
    X_meta_valid = meta[train_size:]
    
    labels = labels[indices]
    labels_as_OHE = pd.get_dummies(labels)
    Y_train = labels_as_OHE[:train_size]
    Y_valid = labels_as_OHE[train_size:]

    return X_nlp_train, X_nlp_valid, X_meta_train, X_meta_valid, Y_train, Y_valid


meta_df = read_meta_data()
nlp_df = read_nlp_data()
X_nlp_train, X_nlp_valid, X_meta_train, X_meta_valid, Y_train, Y_valid = train_valid_split(nlp_df, meta_df)


nlp_input_shape = (X_nlp_train.shape[1],)

nlp_input = Input(shape=(81,), dtype='int32', name='nlp_input')
x = Embedding(output_dim=2, input_dim=10000, input_length=81)(nlp_input)
lstm_out = LSTM(32)(x)

#inpu_dim = Vocab size i både BLP og meta data

'''
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

'''




meta_output = Dense(2, activation='softmax', name='meta_output')(lstm_out)
meta_input = Input(shape=(5,), name='meta_input')
x = concatenate([lstm_out, meta_input])

# We stack a deep densely-connected network on top
x = Dense(15, activation='relu')(x)
#x = Dense(64, activation='relu')(x)
#x = Dense(64, activation='relu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(2, activation='sigmoid', name='main_output')(x)
model = Model(inputs=[nlp_input, meta_input], outputs=[main_output, meta_output])
model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics = ['accuracy', precision, recall, f1], loss_weights=[1., 1.])
#prev.loss_weights[1,0.2]
model.fit([X_nlp_train, X_meta_train], [Y_train, Y_train], epochs=5, batch_size=32)

