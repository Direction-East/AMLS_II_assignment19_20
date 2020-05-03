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

def load_dataset_B(filepath):
    data = pd.read_table(filepath,header=None, names=['iid','target','sentiment','text','timestamp'])
    return data

def data_preprocessing_B(data, max_features):
    data = data[['text','target','sentiment']]
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('[\s]?@[a-z_0-9]*\s',' ',x)))
    data['text'] = data['text'].apply((lambda x: re.sub('[\s]?#[a-z_0-9]*\s',' ',x)))
    data['text'] = data['text'].apply((lambda x: re.sub('\shttp[s]?://[.a-z0-9/]*','',x)))
    data['text'] = data['text'].apply((lambda x: re.sub('[~#-,!:;()\?""%&=\$]','',x)))
    data['text'] = data['text'].apply((lambda x: x.rstrip().lstrip()))

    df = pd.DataFrame(columns = ['left_text','right_text'])

    for i in range(len(data)):
        target_index = data['text'][i].find(data['target'][i])
        left_text = data['text'][i][:target_index]
        right_text = data['text'][i][target_index+len(data['target'][i])+1:]
        text = pd.Series([left_text,right_text],index = df.columns)
        df = df.append(text, ignore_index=True)

    data = pd.concat([data, df], axis=1)

    tokenizer = Tokenizer(nb_words=max_features, split=' ')
    tokenizer.fit_on_texts(data['text'].values)
    X = tokenizer.texts_to_sequences(data['text'].values)
    X = pad_sequences(X)
    X_left = tokenizer.texts_to_sequences(data['left_text'].values)
    X_left = pad_sequences(X_left, maxlen=X.shape[1])
    X_right = tokenizer.texts_to_sequences(data['right_text'].values)
    X_right = pad_sequences(X_right,maxlen=X.shape[1])

    X_left = X_left.reshape(X_left.shape[0],X_left.shape[1],1)
    X_right = X_right.reshape(X_right.shape[0],X_right.shape[1],1)
    X = X.reshape(X.shape[0],X.shape[1],1)
    Y = pd.get_dummies(data['sentiment']).values
    X_train, X_test,X_left_train, X_left_test, X_right_train, X_right_test, Y_train, Y_test = train_test_split(X, X_left, X_right, Y, test_size = 0.33, random_state = 42)
    # print(X_left_train.shape,Y_train.shape)
    # print(X_left_test.shape,Y_test.shape)

    return X_train, X_test,X_left_train, X_left_test, X_right_train, X_right_test, Y_train, Y_test

def B(timestep, max_features):
    # model = Sequential()
    # model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))

    left_input = Input(shape=(timestep,1,))
    left_emb = Embedding(max_features, embed_dim,input_length = timestep)(left_input)
    left_dropout = Dropout(0.3)(left_emb)
    # left_attn = AttentionDecoder(lstm_out, timestep, name='left_attn')(left_emb)
    attention_probs_left = Dense(timestep, activation='softmax', name='attention_vec_left')(left_dropout)
    attention_mul_left =  Multiply()([attention_probs_left, left_input])


    right_input = Input(shape=(timestep,1,))
    right_emb = Embedding(max_features, embed_dim,input_length = timestep)(right_input)
    right_dropout = Dropout(0.3)(left_emb)
    # right_attn = AttentionDecoder(lstm_out, timestep, name='right_attn')(right_emb)
    attention_probs_right = Dense(timestep, activation='softmax', name='attention_vec_right')(right_dropout)
    attention_mul_right =  Multiply()([attention_probs_right, right_input])


    full_sent_input = Input(shape=(timestep,1,))
    full_sent_emb = Embedding(max_features, embed_dim,input_length = timestep)(full_sent_input)
    full_sent_dropout = Dropout(0.3)(full_sent_emb)
    # full_sent_attn = AttentionDecoder(lstm_out, X.shape[1], name='full_sent_attn')(full_sent_emb)
    attention_probs_full_sent = Dense(timestep, activation='softmax', name='attention_vec_full_sent')(full_sent_dropout)
    attention_mul_full_sent =  Multiply()([attention_probs_full_sent, full_sent_input])


    # conc = concatenate([left_attn,full_sent_attn,right_attn])
    conc = concatenate([attention_mul_left, attention_mul_full_sent, attention_mul_right])

    lstm = LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2)(conc)
    result = Dense(2,activation='softmax')(lstm)
    model = Model(inputs=[left_input, full_sent_input, right_input], outputs=result)
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())

    return model

def train_B(model, X_left_train, X_train, X_right_train, Y_train, epochs, batch_size):
    history = model.fit([X_left_train, X_train, X_right_train], Y_train, epochs=epochs, batch_size=batch_size, verbose = 2)
    return history.history['acc'][-1]

def test_B(model, X_left_test, X_test, X_right_test, Y_test, batch_size):
    score,acc = model.evaluate([X_left_test, X_test, X_right_test], Y_test, verbose = 2, batch_size = batch_size)
    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))
    return score, acc
