# -*- coding: utf-8 -*-
#!/usr/bin/env python
#
# Pedro Tabacof
# tabacof at gmail dot com
# April 2016
#
# Bayesian uncertainty in MNIST classification
#
# Dropout MLP model (based on Yarin Gal's approach)
# Variational MLP with Gaussian prior and posterior
#
# Based on the MNIST Lasagne example
# https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py

from __future__ import print_function

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

import lasagne

class mlp_dropout:
    def __init__(self, input_var, target_var, n_in, n_hidden, n_out, dropout_p = 0.5, weight_decay = 0.0):
        l_in = lasagne.layers.InputLayer(shape=(None, n_in),
                                         input_var=input_var)
    
        l_hid1 = lasagne.layers.DenseLayer(
                l_in, num_units=n_hidden,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
    
        l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=dropout_p)
    
        network = lasagne.layers.DenseLayer(
                l_hid1_drop, num_units=n_out,
                nonlinearity=lasagne.nonlinearities.softmax)
    
        # Softmax output
        prediction = lasagne.layers.get_output(network, deterministic=False)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        
        # L2 regularization (weight decay)
        weightsl2 = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
        loss += weight_decay*weightsl2
        
        # SGD training
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.momentum(loss, params, learning_rate=0.01, momentum=0.9)
        self.train = theano.function([input_var, target_var], loss, updates=updates)
    
        # Test functions
        test_loss = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()
        test_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var), dtype=theano.config.floatX)
        self.test = theano.function([input_var, target_var], [test_loss, test_acc])
    
        # Probability and entropy
        self.probabilities = theano.function([input_var], prediction)
        entropy = lasagne.objectives.categorical_crossentropy(prediction, prediction)
        self.entropy_bayesian = theano.function([input_var], entropy)
    
        test_prediction_classical = lasagne.layers.get_output(network, deterministic=True)
        entropy_classical = lasagne.objectives.categorical_crossentropy(test_prediction_classical, test_prediction_classical)
        self.entropy_deterministic = theano.function([input_var], entropy_classical)
               
# Weight initialization helper function
def weight_init(n_in, n_out, name):
    values = np.asarray(np.random.uniform(
        low=-np.sqrt(6. / (n_in + n_out)),
        high=np.sqrt(6. / (n_in + n_out)),
        size=(n_in, n_out)), 
        dtype=theano.config.floatX)

    return theano.shared(value=values, name=name, borrow=True)
    

class mlp_variational:
    def __init__(self, input_var, target_var, n_in, n_hidden, n_out, batch_size):
        # Input to hidden layer weights
        W1_mu = weight_init(n_in, n_hidden, 'W1_mu') # Weights mean
        W1_log_var = weight_init(n_in, n_hidden, 'W1_log_var') # Weights log variance
        
        # Hidden layer to output weights
        W2_mu = weight_init(n_hidden, n_out, 'W2_mu') # Weights mean
        W2_log_var = weight_init(n_hidden, n_out, 'W2_log_var') # Weights log variance
        
        # Biases are not random variables (for convenience)
        b1 = theano.shared(value=np.zeros((n_hidden,), dtype=theano.config.floatX), name='b1', borrow=True)
        b2 = theano.shared(value=np.zeros((n_out,),dtype=theano.config.floatX), name='b2', borrow=True)
         
        # Network parameters
        params = [W1_mu, W1_log_var, W2_mu, W2_log_var, b1, b2]
        
        # Random variables
        srng = MRG_RandomStreams(seed=234)
        rv_hidden = srng.normal((batch_size, n_in, n_hidden))   # Standard normal
        rv_output = srng.normal((batch_size, n_hidden, n_out))  # Standard normal
    
        # MLP
        # Hidden layer
        #hidden_output = T.nnet.relu(T.batched_dot(input_var, W1_mu + T.log(1.0+T.exp(W1_log_var))*rv_hidden) + b1)
        hidden_output = T.nnet.relu(T.batched_dot(input_var, W1_mu + T.exp(W1_log_var)*rv_hidden) + b1)
    
        # Output layer    
        #prediction = T.nnet.softmax(T.batched_dot(hidden_output, W2_mu + T.log(1.0+T.exp(W2_log_var))*rv_output) + b2)
        prediction = T.nnet.softmax(T.batched_dot(hidden_output, W2_mu + T.exp(W2_log_var)*rv_output) + b2)
        
        # KL divergence between prior and posterior
        # For Gaussian prior and posterior, the formula is exact:
        #DKL_hidden = (1.0 + T.log(2.0*T.log(1.0+T.exp(W1_log_var))) - W1_mu**2.0 - 2.0*T.log(1.0+T.exp(W1_log_var))).sum()/2.0
        #DKL_output = (1.0 + T.log(2.0*T.log(1.0+T.exp(W2_log_var))) - W2_mu**2.0 - 2.0*T.log(1.0+T.exp(W2_log_var))).sum()/2.0
        DKL_hidden = (1.0 + 2.0*W1_log_var - W1_mu**2.0 - T.exp(2.0*W1_log_var)).sum()/2.0
        DKL_output = (1.0 + 2.0*W2_log_var - W2_mu**2.0 - T.exp(2.0*W2_log_var)).sum()/2.0
        
        # Negative log likelihood
        nll = T.nnet.categorical_crossentropy(T.clip(prediction, 0.000001, 0.999999), target_var)
        # Complete variational loss    
        loss = nll.mean() - (DKL_hidden + DKL_output)/float(batch_size)
        #loss = nll.mean()
        # SGD training
        updates = sgd(loss, params, 0.01)
        self.train = theano.function([input_var, target_var], loss, updates=updates)
        
        # Test functions
        hidden_output_test = T.nnet.relu(T.dot(input_var, W1_mu) + b1)
        test_prediction = T.nnet.softmax(T.dot(hidden_output_test, W2_mu) + b2)
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var))
        self.test = theano.function([input_var, target_var], [loss, test_prediction, test_acc])
    
        # Probability and entropy
        self.probabilities = theano.function([input_var], prediction)
        entropy = T.nnet.categorical_crossentropy(prediction, prediction)
        self.entropy_bayesian = theano.function([input_var], entropy)
        # Fake deterministic entropy to make the code modular (this should not be used for comparisons)
        self.entropy_deterministic = theano.function([input_var], 0.0*input_var.sum())
        
