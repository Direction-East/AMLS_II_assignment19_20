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

from A.TaskA import *
from B.TaskB import *

# ======================================================================================================================
# Data preprocessing

epochs_A = 7
batch_size_A = 32
max_features_A = 2000

taskA_dir = './Datasets/4A-English'
taskA_dev_data_filepath = os.path.join(taskA_dir, 'SemEval2017-task4-dev.subtask-A.english.INPUT.txt')
data_A = load_dataset_A(taskA_dev_data_filepath)
X_train_A, X_test_A, Y_train_A, Y_test_A = data_preprocessing_A(data_A, max_features_A)


epochs_B = 7
batch_size_B = 32
max_features_B = 2000

taskB_dir = './Datasets/4B-English'
taskB_dev_data_filepath = os.path.join(taskB_dir, 'SemEval2017-task4-dev.subtask-BD.english.INPUT.txt')
data_B = load_dataset_B(taskB_dev_data_filepath)
X_train_B, X_test_B, X_left_train_B, X_left_test_B, X_right_train_B, X_right_test_B, Y_train_B, Y_test_B = data_preprocessing_B(data_B, max_features_B)

# data_train, data_val, data_test = data_preprocessing(args...)
# ======================================================================================================================
# Task A
model_A = A(X_train_A.shape[1], max_features_A)
acc_A_train = train_A(model_A, X_train_A, Y_train_A, epochs_A, batch_size_A)
score_A_test, acc_A_test = test_A(model_A, X_test_A, Y_test_A, batch_size_A)

# model_A = A(args...)                 # Build model object.
# acc_A_train = model_A.train(args...) # Train model based on the training set (you should fine-tune your model based on validation set.)
# acc_A_test = model_A.test(args...)   # Test model based on the test set.
# Clean up memory/GPU etc...             # Some code to free memory if necessary.


# ======================================================================================================================
# Task B
model_B = B(X_train_B.shape[1], max_features_B)
acc_B_train = train_B(model_B, X_left_train_B, X_train_B, X_right_train_B, Y_train_B, epochs_B, batch_size_B)
score_B_test, acc_B_test = test_B(model_B, X_left_test_B, X_test_B, X_right_test_B, Y_test_B, batch_size_B)
# model_B = B(args...)
# acc_B_train = model_B.train(args...)
# acc_B_test = model_B.test(args...)
# Clean up memory/GPU etc...




# ======================================================================================================================
## Print out your results with following format:
print('TA:{},{};TB:{},{};'.format(acc_A_train, acc_A_test,
                                                        acc_B_train, acc_B_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A_train = 'TBD'
# acc_B_test = 'TBD'
