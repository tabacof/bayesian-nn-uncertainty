# -*- coding: utf-8 -*-
#!/usr/bin/env python
#
# Pedro Tabacof
# tabacof at gmail dot com
# April 2016
#
# Bayesian uncertainty in MNIST classification
#
# CIFAR and MNIST loaders
#
# Based on the MNIST Lasagne example
# https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py

from __future__ import print_function

import sys
import os
import math
import numpy as np
import gzip
import tarfile

import matplotlib.pyplot as plt

# We first define a download function, supporting both Python 2 and 3.
if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

def download(filename, source):
    print("Downloading %s" % filename)
    urlretrieve(source + filename, filename)

# We then define functions for loading MNIST images and labels.
# For convenience, they also download the requested files if needed.
def load_mnist_images(filename, source = 'http://yann.lecun.com/exdb/mnist/'):
    if not os.path.exists(filename):
        download(filename,source)
    # Read the inputs in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)

    data = data.reshape(-1, 784)
    # The inputs come as bytes, we convert them to float32 in range [0,1].
    # (Actually to range [0, 255/256], for compatibility to the version
    # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
    return data / np.float32(256)

def load_mnist_labels(filename, source = 'http://yann.lecun.com/exdb/mnist/'):
    if not os.path.exists(filename):
        download(filename, source)
    # Read the labels in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now, that's exactly what we want.
    return data
        
# Load MNIST dataset
def load_MNIST(inside_labels):
    # We can now download and read the training and test set images and labels.
    X_train_all = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train_all = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test_all = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test_all = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    for i in inside_labels:
        X_train.append(X_train_all[np.where(y_train_all == i)])
        y_train.append(y_train_all[np.where(y_train_all == i)])
        
        X_test.append(X_test_all[np.where(y_test_all == i)])
        y_test.append(y_test_all[np.where(y_test_all == i)])
    
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    for count, label in enumerate(inside_labels):
        y_train[y_train == label] = count
        y_test[y_test == label] = count

    val_threshold = math.trunc(0.1 * len(X_train))
    X_train, X_val = X_train[:-val_threshold], X_train[-val_threshold:]
    y_train, y_val = y_train[:-val_threshold], y_train[-val_threshold:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test,  X_test_all, y_test_all

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_CIFAR10(inside_labels):
    filename = 'cifar-10-python.tar.gz'
    if not os.path.exists(filename):
        download(filename, 'https://www.cs.toronto.edu/~kriz/')
    
    tar = tarfile.open(filename, "r:gz")
    tar.extractall()
    tar.close()
    
    xs = []
    ys = []
    for j in range(5):
      d = unpickle('cifar-10-batches-py/data_batch_'+`j+1`)
      x = d['data']
      y = d['labels']
      xs.append(x)
      ys.append(y)

    d = unpickle('cifar-10-batches-py/test_batch')
    xs.append(d['data'])
    ys.append(d['labels'])

    x = np.concatenate(xs)/np.float32(255)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    plt.imshow(x[0].reshape(32,32,3))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:50000])
    x -= pixel_mean

    # create mirrored images
    X_train_all = x[0:50000,:].astype(np.float32)
    y_train_all = y[0:50000].astype(np.int32)
    X_train_flip = X_train_all[:,::-1]
    y_train_flip = y_train_all
    X_train_all = np.concatenate((X_train_all,X_train_flip),axis=0)
    y_train_all = np.concatenate((y_train_all,y_train_flip),axis=0)

    X_test_all = x[50000:,:].astype(np.float32)
    y_test_all = y[50000:].astype(np.int32)

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    for i in inside_labels:
        X_train.append(X_train_all[np.where(y_train_all == i)])
        y_train.append(y_train_all[np.where(y_train_all == i)])
        
        X_test.append(X_test_all[np.where(y_test_all == i)])
        y_test.append(y_test_all[np.where(y_test_all == i)])
    
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    for count, label in enumerate(inside_labels):
        y_train[y_train == label] = count
        y_test[y_test == label] = count
        
    val_threshold = math.trunc(0.1 * len(X_train))
    X_train, X_val = X_train[:-val_threshold], X_train[-val_threshold:]
    y_train, y_val = y_train[:-val_threshold], y_train[-val_threshold:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test,  X_test_all, y_test_all

