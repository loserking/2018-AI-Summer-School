import os
import sys
import argparse
import numpy as np
import pandas as pd
from random import shuffle
from math import log, floor
from keras import optimizers
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation

def load_data(train_data_path, train_label_path, test_data_path, test_label_path):
    X_train = pd.read_csv(train_data_path, sep=',', header=0)
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    Y_train = np.array(Y_train.values)
    X_test = pd.read_csv(test_data_path, sep=',', header=0)
    X_test = np.array(X_test.values)
    Y_test = pd.read_csv(test_label_path, sep=',', header=0)
    Y_test = Y_test['label']
    Y_test = np.array(Y_test.values)

    return (X_train, Y_train, X_test, Y_test)

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

def train(X_train, Y_train):

    # TODO, 1
    # Define model arch
    model = Sequential()
    feat_dims = X_train.shape[1]
    model.add(Dense(input_dim = feat_dims, units = 1, use_bias = True))
    model.add(Activation('sigmoid'))

    # TODO, 2
    # Define optimizer
    sgd = optimizers.SGD(lr=0.001)

    # TODO, 3
    # Compile model
    model.compile(loss = 'binary_crossentropy',
                  optimizer = sgd,
                  metrics = ['accuracy'])

    # TODO, 4
    # Start training
    model.fit(X_train, Y_train, epochs = 10)
    model.save('./logistic_params/logistic.h5')

def infer(X_test, Y_test):

    # TODO, 5
    # load and inference
    model = load_model('./logistic_params/logistic.h5')
    loss, acc = model.evaluate(X_test, Y_test)

    print('Accuracy : ' + str(acc))

def main(opts):
    # Load feature and label
    X_train, Y_train, X_test, Y_test = load_data(opts.train_data_path,
                                                 opts.train_label_path,
                                                 opts.test_data_path,
                                                 opts.test_label_path)

    # Normalization
    X_train, X_test = normalize(X_train, X_test)

    # To train or to infer
    if opts.train:
        train(X_train, Y_train)
    elif opts.infer:
        infer(X_test, Y_test)
    else:
        print("Error: Argument --train or --infer not found")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logistic Regression with Gradient Descent Method')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', default=False,
                        dest='train', help='Input --train to Train')
    group.add_argument('--infer', action='store_true',default=False,
                        dest='infer', help='Input --infer to Infer')
    parser.add_argument('--train_data_path', type=str,
                        default='feature/X_train', dest='train_data_path',
                        help='Path to training data')
    parser.add_argument('--train_label_path', type=str,
                        default='feature/Y_train', dest='train_label_path',
                        help='Path to training data\'s label')
    parser.add_argument('--test_data_path', type=str,
                        default='feature/X_test', dest='test_data_path',
                        help='Path to testing data')
    parser.add_argument('--test_label_path', type=str,
                        default='feature/ans.csv', dest='test_label_path',
                        help='Path to testing data')
    opts = parser.parse_args()
    main(opts)
