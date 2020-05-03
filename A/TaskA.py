import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Input, Multiply
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import os


embed_dim = 128
lstm_out = 196
batch_size = 32

def load_dataset_A(filepath):
    data = pd.read_table(filepath,header=None, names=['iid','sentiment','text','timestamp'])
    return data

def data_preprocessing_A(data, max_features):
    data = data[['text','sentiment']]
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('\s@[a-z_0-9]*\s',' ',x)))
    data['text'] = data['text'].apply((lambda x: re.sub('[~#-,!:;()\?""%&=\$]','',x)))
    data['text'] = data['text'].apply((lambda x: re.sub('\shttp[s]?://*$','',x)))

    # max_features = 2000
    tokenizer = Tokenizer(nb_words=max_features, split=' ')
    tokenizer.fit_on_texts(data['text'].values)
    X = tokenizer.texts_to_sequences(data['text'].values)
    X = pad_sequences(X)

    Y = pd.get_dummies(data['sentiment']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)

    return X_train, X_test, Y_train, Y_test

def A(timestep, max_features):
    input_1 = Input(shape=(timestep,))
    input_emb = Embedding(max_features, embed_dim,input_length = timestep)(input_1)
    # attention_probs = Dense(X.shape[1], activation='softmax', name='attention_vec')(input_emb)
    # attention_mul =  Multiply()([attention_probs, input_1])
    # lstm = Bidirectional(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))(attention_mul)
    lstm = Bidirectional(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))(input_emb)
    result = Dense(3,activation='softmax')(lstm)
    model = Model(inputs=input_1, outputs=result)
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    return model

def train_A(model, X_train, Y_train, batch_size):
    history = model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)
    print(history.history.keys())
    return history.history['acc'][-1]

def test_A(model, X_test, Y_test, batch_size):
    score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))
    return score, acc
