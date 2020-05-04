import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Input, Multiply, Dropout, concatenate
from sklearn.model_selection import train_test_split
import re

# some variable definition to be used in later stage
# these could be tuned for better performance
embed_dim = 128
lstm_out = 196

def load_dataset_B(filepath):
    ''' load the data for task B
    # Arguments
        filepath: the data directory for task B
    # Returns
        data in pandas dataframe format
    '''
    data = pd.read_table(filepath,header=None, names=['iid','target','sentiment','text','timestamp'])
    return data

def data_preprocessing_B(data, max_features):
    ''' preprocess the data for task B
    # Arguments
        data: data obtained for task B
        max_features: maximum feature number limitation for the tokenizer
    # Returns
        X_train: input data for training
        X_test: input data for testing
        X_left_train: the left part to the target word of the sentence as input data for training
        X_left_test: the left part to the target word of the sentence as input data for testing
        X_right_train: the right part to the target word of the sentence as input data for training
        X_right_test:the right part to the target word of the sentence as input data for testing
        Y_train: output data for training
        Y_test: output data for testing
    '''

    # only take the columns that are useful
    data = data[['text','target','sentiment']]

    # filter out useless words
    data['text'] = data['text'].apply(lambda x: x.lower()) # use lower case words only
    data['text'] = data['text'].apply((lambda x: re.sub('[\s]?@[a-z_0-9]*\s',' ',x))) # filter out @people words
    data['text'] = data['text'].apply((lambda x: re.sub('[\s]?#[a-z_0-9]*\s',' ',x))) # filter out #topic words
    data['text'] = data['text'].apply((lambda x: re.sub('\shttp[s]?://[.a-z0-9/]*','',x))) # filter out URL links
    data['text'] = data['text'].apply((lambda x: re.sub('[~#-,!:;()\?""%&=\$]','',x))) # filter out special symbols in sentences
    data['text'] = data['text'].apply((lambda x: x.rstrip().lstrip()))

    # split the sentence into two parts(left and right) using the target word
    df = pd.DataFrame(columns = ['left_text','right_text'])
    for i in range(len(data)):
        target_index = data['text'][i].find(data['target'][i])
        left_text = data['text'][i][:target_index]
        right_text = data['text'][i][target_index+len(data['target'][i])+1:]
        text = pd.Series([left_text,right_text],index = df.columns)
        df = df.append(text, ignore_index=True)

    data = pd.concat([data, df], axis=1)

    # tokenizer the words
    tokenizer = Tokenizer(nb_words=max_features, split=' ')
    tokenizer.fit_on_texts(data['text'].values)
    X = tokenizer.texts_to_sequences(data['text'].values)
    X = pad_sequences(X) # pad short sentences with 0 so the dimensions are consistent
    X_left = tokenizer.texts_to_sequences(data['left_text'].values)
    X_left = pad_sequences(X_left, maxlen=X.shape[1])
    X_right = tokenizer.texts_to_sequences(data['right_text'].values)
    X_right = pad_sequences(X_right,maxlen=X.shape[1])

    # X_left = X_left.reshape(X_left.shape[0],X_left.shape[1],1)
    # X_right = X_right.reshape(X_right.shape[0],X_right.shape[1],1)
    # X = X.reshape(X.shape[0],X.shape[1],1)

    # change the sentiment labels into numerical labels
    Y = pd.get_dummies(data['sentiment']).values
    X_train, X_test,X_left_train, X_left_test, X_right_train, X_right_test, Y_train, Y_test = train_test_split(X, X_left, X_right, Y, test_size = 0.33, random_state = 42)


    # print(X_left_train.shape,Y_train.shape)
    # print(X_left_test.shape,Y_test.shape)

    return X_train, X_test,X_left_train, X_left_test, X_right_train, X_right_test, Y_train, Y_test

def B(timestep, max_features):
    ''' define the model used for task B
    # Arguments
        timestep: number of words in the sentence
        max_features: maximum feature number limitation from the tokenizer
    # Returns
        model: the model for task B
    '''

    left_input = Input(shape=(timestep,))
    left_emb = Embedding(max_features, embed_dim,input_length = timestep)(left_input)
    left_dropout = Dropout(0.3)(left_emb)
    # add attention layer after embedding before lstm
    attention_probs_left = Dense(timestep, activation='softmax', name='attention_vec_left')(left_dropout)
    attention_mul_left =  Multiply()([attention_probs_left, left_input])


    right_input = Input(shape=(timestep,))
    right_emb = Embedding(max_features, embed_dim,input_length = timestep)(right_input)
    right_dropout = Dropout(0.3)(left_emb)
    attention_probs_right = Dense(timestep, activation='softmax', name='attention_vec_right')(right_dropout)
    attention_mul_right =  Multiply()([attention_probs_right, right_input])


    full_sent_input = Input(shape=(timestep,))
    full_sent_emb = Embedding(max_features, embed_dim,input_length = timestep)(full_sent_input)
    full_sent_dropout = Dropout(0.3)(full_sent_emb)
    attention_probs_full_sent = Dense(timestep, activation='softmax', name='attention_vec_full_sent')(full_sent_dropout)
    attention_mul_full_sent =  Multiply()([attention_probs_full_sent, full_sent_input])


    # concatenate the three networks to get the Contextualized attention
    conc = concatenate([attention_mul_left, attention_mul_full_sent, attention_mul_right])

    lstm = LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2)(conc)
    result = Dense(2,activation='softmax')(lstm)
    model = Model(inputs=[left_input, full_sent_input, right_input], outputs=result)
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())

    return model

def train_B(model, X_left_train, X_train, X_right_train, Y_train, epochs, batch_size):
    ''' Train the model for task B
    # Arguments
        model: the model for task B
        X_train, X_left_train, X_right_train: input data for training
        Y_train: output data for training
        epochs: number of times going through the dataset
        batch_size: number of samples every batch
    # Returns
        training accuracy
    '''
    history = model.fit([X_left_train, X_train, X_right_train], Y_train, epochs=epochs, batch_size=batch_size, verbose = 2)
    return history.history['accuracy'][-1]

def test_B(model, X_left_test, X_test, X_right_test, Y_test, batch_size, validation_size):
    ''' Test the model of task B
    # Arguments
        model: the model for task B
        X_test, X_left_test, X_right_test: input data for testing
        Y_test: output data for testing
        batch_size: number of samples every batch
    # Returns
        validation accuracy
        testing accuracy
    '''

    X_validation = X_test[-validation_size:]
    X_left_validation = X_left_test[-validation_size:]
    X_right_validation = X_right_test[-validation_size:]
    Y_validation = Y_test[-validation_size:]

    score_valid, acc_valid = model.evaluate([X_left_validation, X_validation, X_right_validation], Y_validation, verbose = 2, batch_size=batch_size)

    X_test = X_test[:-validation_size]
    X_left_test = X_left_test[:-validation_size]
    X_right_test = X_right_test[:-validation_size]
    Y_test = Y_test[:-validation_size]

    score_test, acc_test = model.evaluate([X_left_test, X_test, X_right_test], Y_test, verbose = 2, batch_size=batch_size)
    print("acc_valid: %.2f" % (acc_valid))
    print("acc_test: %.2f" % (acc_test))
    return acc_valid, acc_test
