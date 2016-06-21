# -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:56:50 2016

@author: tabacof
"""

import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import numpy as np
import lasagne
import matplotlib
matplotlib.use('Agg')
import time

import datasets

train_weights = True
num_epochs = 50
batch_size = 128

X_train, y_train, X_test, y_test, X_test_all, y_test_all = datasets.load_CIFAR10([0,1,2,3])

dropout_p = 0.5
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')
        
network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=input_var)

# Convolution + pooling + dropout
network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=5, nonlinearity=lasagne.nonlinearities.elu)
network = lasagne.layers.DropoutLayer(network,  p=dropout_p)
network = lasagne.layers.Pool2DLayer(network, pool_size=2)

network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=5, nonlinearity=lasagne.nonlinearities.elu)
network = lasagne.layers.DropoutLayer(network,  p=dropout_p)
network = lasagne.layers.Pool2DLayer(network, pool_size=2)

# Fully-connected + dropout
network = lasagne.layers.DenseLayer(network, num_units=1000, nonlinearity=lasagne.nonlinearities.elu)
network = lasagne.layers.DropoutLayer(network,  p=dropout_p)

network = lasagne.layers.DenseLayer(
        network, num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)

if train_weights:
    # Softmax output
    prediction = lasagne.layers.get_output(network, deterministic=False)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    
    # L2 regularization (weight decay)
    weightsl2 = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    loss += 1e-5*weightsl2
    
    # ADAM training
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params)
    train = theano.function([input_var, target_var], loss, updates=updates)
    
    # Test functions
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var).mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
    test = theano.function([input_var, target_var], [test_loss, test_acc])
    bayesian_test_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var), dtype=theano.config.floatX)
    bayesian_test = theano.function([input_var, target_var], bayesian_test_acc)
    
    def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]
    
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            err = train(inputs, targets)
            train_err += err
            train_batches += 1
    
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = test(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
    
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
    
    test_err = 0
    test_bayes_acc = 0
    test_acc = 0
    test_batches = 0
    bayes_repeat = 32
    for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
        inputs, targets = batch
        
        # Bayesian accuracy (multiple dropout samples)
        bayes_acc = 0.0
        for i, t in zip(inputs, targets):
            bayes_acc += bayesian_test(np.repeat(i[np.newaxis], bayes_repeat, 0), np.repeat(t[np.newaxis], bayes_repeat, 0))
            
        bayes_acc /= batch_size
        test_bayes_acc += bayes_acc
        
        # Standard accuracy (no dropout)
        err, acc = test(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
    print("  bayes accuracy:\t\t{:.2f} %".format(test_bayes_acc / test_batches * 100))
    
    np.savez('cifar_outsiders_pred.npz', *lasagne.layers.get_all_param_values(network))
else:
    with np.load('cifar_outsiders_pred.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values) 
        

