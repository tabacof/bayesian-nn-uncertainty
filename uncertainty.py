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
import seaborn as sns

import theano.tensor as T

import datasets
import models
import training

import pandas as pd

def anomaly(experiment_name,
            dataset = "mnist",
            bayesian_approximation  = "dropout",
            inside_labels = [0, 1],
            num_epochs = 50,
            batch_size = 128,
            weight_decay = 1e-5,
            dropout_p = 0.5,
            fc_layers = [512, 512],
            plot = True):
    """
    This methods trains a neural network classifier on a subset of classes.
    After the training, it uses uncertainty measures (e.g. entropy) to detect anomalies.
    The anomalous classes are the ones that are not part of the training subset.
    
    dataset = "mnist" or "cifar"
    For MNIST we use a fully-connected MLP.
    For CIFAR10 we use a convolutional net (similar to LeNet)
    
    bayesian_approximation = "dropout" for Yarin Gal's method - work either with MNIST 
    bayesian_approximation = "variational" for fully-factorized Gaussian variational approximation - only work with MNIST.
    
    inside_labels are the subset of trained classes, the other classes are only used for testing.             
    """

    n_out = len(inside_labels)
    
    # Prepare Theano variables for inputs and targets
    
    # Load the dataset
    print("Loading data...")
    if dataset == "mnist":
        input_var = T.matrix('inputs')
        target_var = T.ivector('targets')
        n_in = [28*28]
        X_train, y_train, X_test, y_test, X_test_all, y_test_all = datasets.load_MNIST(inside_labels)
        if bayesian_approximation == "dropout":
            model = models.mlp_dropout(input_var, target_var, n_in, n_out, fc_layers, dropout_p, weight_decay)
        elif bayesian_approximation == "variational":
            model = models.mlp_variational(input_var, target_var, n_in, n_out, fc_layers, batch_size, len(X_train)/float(batch_size))
    elif dataset == "cifar":
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')
    
        n_in = [3, 32, 32]
        X_train, y_train, X_test, y_test, X_test_all, y_test_all = datasets.load_CIFAR10(inside_labels)
        model = models.convnet_dropout(input_var, target_var, n_in, n_out, dropout_p, weight_decay)
    
    df = pd.DataFrame()

    # Mini-batch training with ADAM
    training.train(model, X_train, y_train, X_test, y_test, batch_size, num_epochs)
    # Mini-batch testing
    acc, bayes_acc = training.test(model, X_test, y_test, batch_size)
    df.set_value(experiment_name, "test_acc", acc)
    df.set_value(experiment_name, "bayes_test_acc", bayes_acc)

    # Uncertainty prediction
    test_pred_mean = {x:[] for x in range(10)}
    test_pred_std = {x:[] for x in range(10)}
    test_entropy_bayesian = {x:[] for x in range(10)}
    test_entropy_deterministic = {x:[] for x in range(10)}
    
    for i in range(len(X_test_all)):
        probs = model.probabilities(np.tile(X_test_all[i], batch_size).reshape([-1] + n_in))
        bayesian_entropy = model.entropy_bayesian(np.tile(X_test_all[i], batch_size).reshape([-1] + n_in))
        classical_entropy = model.entropy_deterministic(X_test_all[i][np.newaxis,:])
        predictive_mean = np.mean(probs, axis=0)
        predictive_std = np.std(probs, axis=0)
        test_pred_mean[y_test_all[i]].append(predictive_mean[1])
        test_pred_std[y_test_all[i]].append(predictive_std.mean())
        test_entropy_bayesian[y_test_all[i]].append(bayesian_entropy.mean())
        test_entropy_deterministic[y_test_all[i]].append(classical_entropy.mean())
    
    # Plotting
    if plot:
        for k in sorted(test_pred_mean.keys()):
            sns.plt.figure()
            #sns.plt.hist(test_pred_mean[k], label = "Prediction mean for " + str(k))
            sns.plt.hist(test_entropy_bayesian[k], label = "Bayesian Entropy v1 for " + str(k))
            sns.plt.hist(test_pred_std[k], label = "Prediction std for " + str(k))
            #sns.plt.hist(test_entropy_deterministic[k], label = "Classical entropy for " + str(k))
            sns.plt.legend()
            sns.plt.show()
    
    # Anomaly detection
    # by classical prediction entropy
    def anomaly_detection(anomaly_score_dict, name, df):
        threshold = np.logspace(-30, 1.0, 1000)
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
        df.set_value(experiment_name, name + ' bal_acc', sorted_acc[0][1][0])   
        df.set_value(experiment_name, name + ' bal_acc_threshold', sorted_acc[0][0])        

        print("\tbalanced acc\t{:.3f}\t{:.6f}\t\t{:.3f}\t{:.3f}".format(sorted_acc[0][1][0], sorted_acc[0][0], sorted_acc[0][1][2], sorted_acc[0][1][3]))
        sorted_acc = sorted(acc.items(), key= lambda x : x[1][1], reverse = True)
        df.set_value(experiment_name, name + ' f1_score', sorted_acc[0][1][1])                
        df.set_value(experiment_name, name + ' f1_score_threshold', sorted_acc[0][0])        

        print("\tf1 score\t{:.3f}\t{:.6f}\t\t{:.3f}\t{:.3f}".format(sorted_acc[0][1][1], sorted_acc[0][0], sorted_acc[0][1][2], sorted_acc[0][1][3]))
        return df
        
    df.set_value(experiment_name, 'dataset', dataset)    
    df.set_value(experiment_name, 'bayesian_approx', bayesian_approximation)    
    df.set_value(experiment_name, 'inside_labels', str(inside_labels))    
    df.set_value(experiment_name, 'num_epochs', num_epochs)    
    df = anomaly_detection(test_entropy_bayesian, "Bayesian entropy", df)
    df = anomaly_detection(test_entropy_deterministic, "Classical entropy", df)
    df = anomaly_detection(test_pred_std, "Bayesian prediction STD", df)

    return df
    