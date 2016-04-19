# -*- coding: utf-8 -*-
#!/usr/bin/env python
#
# Pedro Tabacof
# tabacof at gmail dot com
# April 2016
#
# Bayesian uncertainty in MNIST classification
#
# Based on the MNIST Lasagne example
# https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py

from __future__ import print_function

import numpy as np
import scipy.stats
import seaborn as sns

import theano.tensor as T

import datasets
import models
import training

# Experiment parameters

#dataset = "mnist"
dataset = "cifar"

num_epochs = 150 # Number of epochs
batch_size = 100 # Mini batch size (also used for number of posterior samples)
weight_decay = 1e-4 # L2 regularization
dropout_p = 0.5 # Dropout probability
n_hidden = 512 # Number of neurons at hidden layer
inside_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8] # Labels to be trained

# Bayesian approximation method
bayesian_approximation  = "dropout" # Use Gal's variational dropout method
#bayesian_approximation  = "variational" # Use Gaussian variational approximation

n_out = len(inside_labels)

# Load the dataset
print("Loading data...")
if dataset == "mnist":
    n_in = 28*28
    X_train, y_train, X_test, y_test, X_test_all, y_test_all = datasets.load_MNIST(inside_labels)
elif dataset == "cifar":
    n_in = 32*32*3
    X_train, y_train, X_test, y_test, X_test_all, y_test_all = datasets.load_CIFAR10(inside_labels)

# Prepare Theano variables for inputs and targets
input_var = T.matrix('inputs')
target_var = T.ivector('targets')

if bayesian_approximation == "dropout":
    model = models.mlp_dropout(input_var, target_var, n_in, n_hidden, n_out, dropout_p, weight_decay)
elif bayesian_approximation == "variational":
    model = models.mlp_variational(input_var, target_var, n_in, n_hidden, n_out, batch_size)
    
# Mini-batch training with SGD
training.train(model, X_train, y_train, batch_size, num_epochs)
# Mini-batch testing
training.test(model, X_test, y_test, batch_size)

# Uncertainty prediction
test_pred_mean = {x:[] for x in range(0,10)}
test_pred_std = {x:[] for x in range(0,10)}
test_entropy_bayesian_v1 = {x:[] for x in range(0,10)}
test_entropy_bayesian_v2 = {x:[] for x in range(0,10)}
test_entropy_deterministic = {x:[] for x in range(0,10)}

print("Total test samples", len(X_test_all))
for i in range(len(X_test_all)):
    probs = model.probabilities(np.tile(X_test_all[i], batch_size).reshape(-1, n_in))
    bayesian_entropy = model.entropy_bayesian(np.tile(X_test_all[i], batch_size).reshape(-1, n_in))
    classical_entropy = model.entropy_deterministic(X_test_all[i][np.newaxis,:])
    predictive_mean = np.mean(probs, axis=0)
    predictive_std = np.std(probs, axis=0)
    test_pred_mean[y_test_all[i]].append(predictive_mean[0])
    test_pred_std[y_test_all[i]].append(predictive_std[0])
    test_entropy_bayesian_v1[y_test_all[i]].append(bayesian_entropy.mean())
    test_entropy_bayesian_v2[y_test_all[i]].append(scipy.stats.entropy(predictive_mean))
    test_entropy_deterministic[y_test_all[i]].append(classical_entropy.mean())

# Plotting
for k in sorted(test_pred_mean.keys()):
    sns.plt.figure()
    sns.plt.hist(test_pred_mean[k], label = "Prediction mean for " + str(k))
    sns.plt.hist(test_entropy_bayesian_v1[k], label = "Bayesian Entropy v1 for " + str(k))
    sns.plt.hist(test_entropy_bayesian_v2[k], label = "Bayesian Entropy v2 for " + str(k))    
    sns.plt.hist(test_pred_std[k], label = "Prediction std for " + str(k))
    #sns.plt.hist(test_entropy_deterministic[k], label = "Classical entropy for " + str(k))
    sns.plt.legend()
    sns.plt.show()

# Anomaly detection
# by classical prediction entropy
def anomaly_detection(anomaly_score_dict, name):
    threshold = np.linspace(0, 1.0, 1000)
    acc = {}
    for t in threshold:
        tp = 0.0
        tn = 0.0
        for l in anomaly_score_dict:
            if l in inside_labels:
                tp += (np.array(anomaly_score_dict[l]) < t).mean()
            else:
                tn += (np.array(anomaly_score_dict[l]) >= t).mean()
        tp /= len(inside_labels)
        tn /= 10.0 - len(inside_labels)
        bal_acc = (tp + tn)/2.0
        f1_score = 2.0*tp/(2.0 + tp - tn)
        acc[t] = [bal_acc, f1_score, tp, tn]
        
    print("{}\tscore\tthreshold\tTP\tTN".format(name))
    sorted_acc = sorted(acc.items(), key= lambda x : x[1][0], reverse = True)
    print("\tbalanced acc\t{:.3f}\t{:.3f}\t\t{:.3f}\t{:.3f}".format(sorted_acc[0][1][0], sorted_acc[0][0], sorted_acc[0][1][2], sorted_acc[0][1][3]))
    sorted_acc = sorted(acc.items(), key= lambda x : x[1][1], reverse = True)
    print("\tf1 score\t{:.3f}\t{:.3f}\t\t{:.3f}\t{:.3f}".format(sorted_acc[0][1][1], sorted_acc[0][0], sorted_acc[0][1][2], sorted_acc[0][1][3]))

anomaly_detection(test_entropy_bayesian_v1, "Bayesian entropy v1")
anomaly_detection(test_entropy_bayesian_v2, "Bayesian entropy v2")
anomaly_detection(test_entropy_deterministic, "Classical entropy")
anomaly_detection(test_pred_std, "Bayesian prediction STD")
