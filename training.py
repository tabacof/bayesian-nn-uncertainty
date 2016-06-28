# -*- coding: utf-8 -*-
#!/usr/bin/env python
#
# Pedro Tabacof
# tabacof at gmail dot com
# April 2016
#
# Bayesian uncertainty in MNIST classification
#
# Training and testing procedures
#
# Based on the MNIST Lasagne example
# https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py

from __future__ import print_function

import time

import numpy as np

# Mini batch iterator for training and testing
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

def train(model, X_train, y_train, X_val, y_val, batch_size, num_epochs, acc_threshold, verbose = True):
    if verbose:
        print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            err = model.train(inputs, targets)
            train_err += err
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = model.test(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        if verbose:
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))
                
        if val_acc / val_batches >= acc_threshold:
            break

def test(model, X_test, y_test, batch_size, bayes_repeat = 32):
    # After training, we compute and print the test error:
    test_err = 0
    test_bayes_acc = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
        inputs, targets = batch
        
        # Bayesian accuracy (multiple dropout samples)
        bayes_acc = 0.0
        for i, t in zip(inputs, targets):
            bayes_acc += model.bayesian_test(np.repeat(i[np.newaxis], bayes_repeat, 0), np.repeat(t[np.newaxis], bayes_repeat, 0))
        bayes_acc /= batch_size
        test_bayes_acc += bayes_acc
        
        # Standard accuracy (no dropout)
        err, acc = model.test(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
    print("  bayes accuracy:\t\t{:.2f} %".format(test_bayes_acc / test_batches * 100))

    return test_acc / test_batches, test_bayes_acc / test_batches
