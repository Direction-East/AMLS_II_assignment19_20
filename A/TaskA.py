import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Input, Multiply, Dropout
from sklearn.model_selection import train_test_split
import re

# some variable definition to be used in later stage
# these could be tuned for better performance
embed_dim = 128
lstm_out = 196

def load_dataset_A(filepath):
    ''' load the data for task A
    # Arguments
        filepath: the data directory for task A
    # Returns
        data in pandas dataframe format
    '''
    data = pd.read_table(filepath,header=None, names=['iid','sentiment','text','timestamp'])
    return data

def data_preprocessing_A(data, max_features):
    ''' preprocess the data for task A
    # Arguments
        data: data obtained for task A
        max_features: maximum feature number limitation for the tokenizer
    # Returns
        X_train: input data for training
        X_test: input data for testing
        Y_train: output data for training
        Y_test: output data for testing
    '''
    # only take the columns that are useful
    data = data[['text','sentiment']]

    # filter out useless words
    data['text'] = data['text'].apply(lambda x: x.lower()) # use lower case words only
    data['text'] = data['text'].apply((lambda x: re.sub('[\s]?@[a-z_0-9]*\s',' ',x))) # filter out @people words
    data['text'] = data['text'].apply((lambda x: re.sub('[\s]?#[a-z_0-9]*\s',' ',x))) # filter out #topic words
    data['text'] = data['text'].apply((lambda x: re.sub('\shttp[s]?://[.a-z0-9/]*','',x))) # filter out URL links
    data['text'] = data['text'].apply((lambda x: re.sub('[~#-,!:;()\?""%&=\$]','',x))) # filter out special symbols in sentences
    data['text'] = data['text'].apply((lambda x: x.rstrip().lstrip()))

    # tokenizer the words
    tokenizer = Tokenizer(nb_words=max_features, split=' ') # max_features = 2000
    tokenizer.fit_on_texts(data['text'].values)
    X = tokenizer.texts_to_sequences(data['text'].values)
    X = pad_sequences(X) # pad short sentences with 0 so the dimensions are consistent

    # change the sentiment labels into numerical labels
    Y = pd.get_dummies(data['sentiment']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)

    return X_train, X_test, Y_train, Y_test

def A(timestep, max_features):
    ''' define the model used for task A
    # Arguments
        timestep: number of words in the sentence
        max_features: maximum feature number limitation from the tokenizer
    # Returns
        model: the model for task A
    '''
    input_1 = Input(shape=(timestep,))
    input_emb = Embedding(max_features, embed_dim,input_length = timestep)(input_1)
    emb_dropout = Dropout(0.3)(input_emb)
    # attention_probs = Dense(X.shape[1], activation='softmax', name='attention_vec')(input_emb)
    # attention_mul =  Multiply()([attention_probs, input_1])
    # lstm = Bidirectional(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))(attention_mul)
    lstm = Bidirectional(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))(emb_dropout)
    result = Dense(3,activation='softmax')(lstm)
    model = Model(inputs=input_1, outputs=result)
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    return model

def train_A(model, X_train, Y_train, epochs, batch_size):
    ''' Train the model for task A
    # Arguments
        model: the model for task A
        X_train: input data for training
        Y_train: output data for training
        epochs: number of times going through the dataset
        batch_size: number of samples every batch
    # Returns
        training accuracy
    '''
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose = 2)
    # print(history.history.keys())
    return history.history['accuracy'][-1]

def test_A(model, X_test, Y_test, batch_size, validation_size):
    ''' Test the model of task A
    # Arguments
        model: the model for task A
        X_test: input data for testing
        Y_test: output data for testing
        batch_size: number of samples every batch
    # Returns
        validation accuracy
        testing accuracy
    '''
    X_validate = X_test[-validation_size:]
    Y_validate = Y_test[-validation_size:]
    X_test = X_test[:-validation_size]
    Y_test = Y_test[:-validation_size]
    score_valid, acc_valid = model.evaluate(X_validate , Y_validate , verbose = 2, batch_size=batch_size)
    score_test, acc_test = model.evaluate(X_test, Y_test, verbose = 2, batch_size=batch_size)
    print("acc_valid: %.2f" % (acc_valid))
    print("acc_test: %.2f" % (acc_test))
    return acc_valid, acc_test
