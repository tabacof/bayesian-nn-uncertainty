# -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:56:50 2016

@author: tabacof
"""

import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import numpy as np
import scipy
import lasagne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lasagne.utils import floatX

import time

import datasets
num_epochs = 10
batch_size = 128

X_train, y_train, X_val, y_val, X_test, y_test, X_test_all, y_test_all = datasets.load_CIFAR10(range(10))

dropout_p = 0.5
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')
        
network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=input_var)
                                
l_noise = lasagne.layers.BiasLayer(network, b = np.zeros((3,32,32), dtype = np.float32), shared_axes = 0)
l_noise.params[l_noise.b].remove('trainable')

# Convolution + pooling + dropout
network = lasagne.layers.Conv2DLayer(l_noise, num_filters=192, filter_size=5, nonlinearity=lasagne.nonlinearities.elu)
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
    for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
        inputs, targets = batch
        err, acc = test(inputs, targets)
        print(err, acc)
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

def plot(name, img):
    img = np.copy(img)[0, ::-1, :, :]
    img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
    img *= 255.0
    print "Out of bounds:", np.sum(img < 0.0) + np.sum(img > 255.0)
    plt.figure()
    plt.imshow(img.astype(np.uint8))
    #plt.savefig(name+".pdf", bbox_inches='tight')
    plt.show()

img = X_test[0][np.newaxis]
mean_cifar = np.array([])

plot("Original",img)

l_noise.b.set_value(np.random.uniform(-1e-8, 1e-8, size=(3,32,32)).astype(np.float32))

pred = np.array(lasagne.layers.get_output(network, img, deterministic=True).eval())
top1 = np.argmax(pred)
target = np.zeros(pred.shape)
adv_class = np.random.randint(0, 10)
target[0,adv_class] = 1.0
print "Before ADV top1:", top1
print "True class", y_test[0]
print "Adversarial class", adv_class
input_img = T.tensor4()

prob = lasagne.layers.get_output(network, input_img, deterministic=True)
C = T.scalar()
adv_loss = lasagne.objectives.categorical_crossentropy(prob, floatX(target)).mean() + C*lasagne.regularization.l2(l_noise.b)
adv_grad = T.grad(adv_loss, l_noise.b)

adv_function = theano.function([input_img, C], [adv_loss, adv_grad, prob])

# Optimization function for L-BFGS-B
def fmin_func(x, C = 0.00001):
    l_noise.b.set_value(x.reshape(3, 224, 224).astype(np.float32))
    f, g, _ = adv_function(img, C)
    return float(f), g.flatten().astype(np.float64)
    
# Noise bounds (pixels cannot exceed 0-1)
#bounds = zip(-(mean_vgg-img).flatten(), ((255.0-mean_vgg)-img).flatten())

# L-BFGS-B optimization to find adversarial noise
x, f, d = scipy.optimize.fmin_l_bfgs_b(fmin_func, l_noise.b.get_value().flatten(), bounds = None, maxfun = 50, fprime = None, factr = 1e10, m = 15)
l_noise.b.set_value(x.reshape(3, 32, 32).astype(np.float32))

_, _, pred = adv_function(img, 0.0)
top1 = np.argmax(pred)
print "After ADV top1:", top1
plot("Adv image", img + l_noise.b.get_value())

bayesian_prob = lasagne.layers.get_output(network, input_img, deterministic=False)
bayesian_function = theano.function([input_img], [bayesian_prob])
bayesian_pred = np.zeros(pred.shape)

for _ in range(25):
    bayesian_pred[0, np.argmax(bayesian_function(img))] += 1
    
print "After ADV Bayesian top1:", np.argmax(bayesian_pred)
