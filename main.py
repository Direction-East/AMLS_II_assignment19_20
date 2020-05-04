# import libraries
import os

from A.TaskA import *
from B.TaskB import *

# ======================================================================================================================
# Data preprocessing

# some variable definition to be used in later stage
# these could be tuned for better performance
epochs_A = 7
batch_size_A = 32
max_features_A = 2000
validation_size_A = 1500
Load_weight_from_file = False

# directory of the data for task A
taskA_dir = './Datasets/4A-English'
taskA_dev_data_filepath = os.path.join(taskA_dir, 'SemEval2017-task4-dev.subtask-A.english.INPUT.txt')
# load data
data_A = load_dataset_A(taskA_dev_data_filepath)
# pre-process the data
X_train_A, X_test_A, Y_train_A, Y_test_A = data_preprocessing_A(data_A, max_features_A)


epochs_B = 7
batch_size_B = 32
max_features_B = 2000
validation_size_B = 1500

taskB_dir = './Datasets/4B-English'
taskB_dev_data_filepath = os.path.join(taskB_dir, 'SemEval2017-task4-dev.subtask-BD.english.INPUT.txt')
data_B = load_dataset_B(taskB_dev_data_filepath)
X_train_B, X_test_B, X_left_train_B, X_left_test_B, X_right_train_B, X_right_test_B, Y_train_B, Y_test_B = data_preprocessing_B(data_B, max_features_B)


# ======================================================================================================================
# Task A
# model_A = A(X_train_A.shape[1], max_features_A) # Build model object.
# acc_A_train = train_A(model_A, X_train_A, Y_train_A, epochs_A, batch_size_A) # Train model based on the training set
# acc_A_valid, acc_A_test = test_A(model_A, X_test_A, Y_test_A, batch_size_A, validation_size_A) # Test model based on the test set.
# model_A.save_weights('./Datasets/TaskA_BiLSTM') # save model, change the filename otherwise will overwrite
#
# # ======================================================================================================================
# # Task B
# model_B = B(X_train_B.shape[1], max_features_B)
# acc_B_train = train_B(model_B, X_left_train_B, X_train_B, X_right_train_B, Y_train_B, epochs_B, batch_size_B)
# acc_B_valid, acc_B_test = test_B(model_B, X_left_test_B, X_test_B, X_right_test_B, Y_test_B, batch_size_B, validation_size_B)
# model_B.save_weights('./Datasets/TaskB_ContextualizedAttention+BiLSTM') # save model, change the filename otherwise will overwrite
# Load_weight_from_file = False

# ======================================================================================================================
# Load model
# comment out this entire section if training
# comment out entire Section Task A and Section Task B above if loading model
Load_weight_from_file = True

model_A = A(X_train_A.shape[1], max_features_A) # create model object.
model_A.load_weights('./Datasets/TaskA_BiLSTM') # load weight
acc_A_valid, acc_A_test = test_A(model_A, X_test_A, Y_test_A, batch_size_A, validation_size_A) # evaluate the model

model_B = B(X_train_B.shape[1], max_features_B)
model_B.load_weights('./Datasets/TaskB_ContextualizedAttention+BiLSTM')
acc_B_valid, acc_B_test = test_B(model_B, X_left_test_B, X_test_B, X_right_test_B, Y_test_B, batch_size_B, validation_size_B)

# ======================================================================================================================
## Result
if(Load_weight_from_file):
    print('TA: test accuracy = {acc_A_test:.2f};\nTB: test accuracy = {acc_B_test:.2f};'.format(acc_A_test=acc_A_test, acc_B_test=acc_B_test))
else:
    print('TA: train accuracy = {acc_A_train:.2f}, test accuracy = {acc_A_test:.2f};\nTB: train accuracy = {acc_B_train:.2f}, test accuracy = {acc_B_test:.2f};'.format(acc_A_train =acc_A_train , acc_A_test=acc_A_test,acc_B_train=acc_B_train, acc_B_test=acc_B_test))
