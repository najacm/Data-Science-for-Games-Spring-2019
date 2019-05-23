import pandas as pd
import csv
from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer


def read_meta_data():
    csv_file = '/Users/MasterWillis/ownCloud/ITU/gitDS/Data-Science-for-Games-Spring-2019/LSTM/metadata.csv'
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

meta_df = read_meta_data()
nlp_df = read_nlp_data()



meta_X = np.asarray(meta_df[['longest_word','no_of_words','length_of_text','sum_of_special_char','sum_of_digits','has_upper_case']])   
meta_Y = pd.get_dummies(meta_df['label'].values)
meta_X_train, meta_X_valid, meta_Y_train, meta_Y_valid = train_test_split(meta_X,meta_Y, test_size = 0.50, random_state = 36)

tokenizer = Tokenizer(oov_token=True)
tokenizer.fit_on_texts(nlp_df['text'].values)
all_seq_as_indices = tokenizer.texts_to_sequences(nlp_df['text'])
nlp_X = pad_sequences(all_seq_as_indices)
nlp_Y = pd.get_dummies(nlp_df['label'].values)
nlp_X_train, nlp_X_valid, nlp_Y_train, nlp_Y_valid = train_test_split(nlp_X,nlp_Y, test_size = 0.50, random_state = 36)


'''
texts = np.asarray(nlp_dlf['text'].shape)
labels = np.asarray(labels)

np.random.seed(42)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)

texts = texts[indices]
labels = labels[indices]
training_samples = int(5572 * .8)
validation_samples = int(5572 - training_samples)
texts_train = texts[:training_samples]
y_train = labels[:training_samples]
texts_test = texts[training_samples:]
y_test = labels[training_samples:]




'''













nlp_input = Input(shape=(81,), dtype='int32', name='nlp_input')
x = Embedding(output_dim=512, input_dim=10000, input_length=81)(nlp_input)
lstm_out = LSTM(32)(x)

meta_output = Dense(2, activation='sigmoid', name='meta_output')(lstm_out)
meta_input = Input(shape=(6,), name='meta_input')
x = concatenate([lstm_out, meta_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(2, activation='sigmoid', name='main_output')(x)
model = Model(inputs=[nlp_input, meta_input], outputs=[main_output, meta_output])
model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics = ['accuracy', precision, recall, f1], loss_weights=[1., 0.2])

#meta_model.compile(optimizer='adam',loss = 'binary_crossentropy', metrics = ['accuracy', precision, recall, f1])#model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])

#model.fit([headline_data, additional_data], [labels, labels], epochs=50, batch_size=32)
model.fit([nlp_X_train, meta_X_train], [nlp_Y_train, meta_Y_train], epochs=1, batch_size=32)

