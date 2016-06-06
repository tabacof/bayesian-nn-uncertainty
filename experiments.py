# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 22:19:39 2016

@author: tabacof
"""

import uncertainty
import pandas as pd

df = pd.DataFrame()

df = df.append(uncertainty.anomaly("test_mnist_2labels_1_v1", "mnist", num_epochs = 25, inside_labels=[0,1], plot = False))
df = df.append(uncertainty.anomaly("test_mnist_2labels_2_v1", "mnist", num_epochs = 25, inside_labels=[2,7], plot = False))
df = df.append(uncertainty.anomaly("test_mnist_2lavels_3_v1", "mnist", num_epochs = 25, inside_labels=[8,9], plot = False))

df = df.append(uncertainty.anomaly("test_mnist_5labels_1_v1", "mnist", num_epochs = 50, inside_labels=[0,2,4,6,8], plot = False))
df = df.append(uncertainty.anomaly("test_mnist_5labels_2_v1", "mnist", num_epochs = 50, inside_labels=[1,3,5,7,9], plot = False))
df = df.append(uncertainty.anomaly("test_mnist_5labels_3_v1", "mnist", num_epochs = 50, inside_labels=[0,1,2,3,4], plot = False))

df = df.append(uncertainty.anomaly("test_mnist_8labels_1_v1", "mnist", num_epochs = 75, inside_labels=[2,3,4,5,6,7,8,9], plot = False))
df = df.append(uncertainty.anomaly("test_mnist_8labels_2_v1", "mnist", num_epochs = 75, inside_labels=[0,1,3,4,5,6,8,9], plot = False))
df = df.append(uncertainty.anomaly("test_mnist_8labels_3_v1", "mnist", num_epochs = 75, inside_labels=[0,1,2,3,4,5,6,7], plot = False))

df = df.append(uncertainty.anomaly("test_mnist_2labels_1_v2", "mnist", num_epochs = 25, inside_labels=[0,1], plot = False))
df = df.append(uncertainty.anomaly("test_mnist_2labels_2_v2", "mnist", num_epochs = 25, inside_labels=[2,7], plot = False))
df = df.append(uncertainty.anomaly("test_mnist_2lavels_3_v2", "mnist", num_epochs = 25, inside_labels=[8,9], plot = False))

df = df.append(uncertainty.anomaly("test_mnist_5labels_1_v2", "mnist", num_epochs = 50, inside_labels=[0,2,4,6,8], plot = False))
df = df.append(uncertainty.anomaly("test_mnist_5labels_2_v2", "mnist", num_epochs = 50, inside_labels=[1,3,5,7,9], plot = False))
df = df.append(uncertainty.anomaly("test_mnist_5labels_3_v2", "mnist", num_epochs = 50, inside_labels=[0,1,2,3,4], plot = False))

df = df.append(uncertainty.anomaly("test_mnist_8labels_1_v2", "mnist", num_epochs = 75, inside_labels=[2,3,4,5,6,7,8,9], plot = False))
df = df.append(uncertainty.anomaly("test_mnist_8labels_2_v2", "mnist", num_epochs = 75, inside_labels=[0,1,3,4,5,6,8,9], plot = False))
df = df.append(uncertainty.anomaly("test_mnist_8labels_3_v2", "mnist", num_epochs = 75, inside_labels=[0,1,2,3,4,5,6,7], plot = False))


df = df.append(uncertainty.anomaly("test_cifar_2labels_1_v1", "cifar", num_epochs = 25, inside_labels=[0,1], plot = False))
df = df.append(uncertainty.anomaly("test_cifar_2labels_2_v1", "cifar", num_epochs = 25, inside_labels=[2,7], plot = False))
df = df.append(uncertainty.anomaly("test_cifar_2lavels_3_v1", "cifar", num_epochs = 25, inside_labels=[8,9], plot = False))

df = df.append(uncertainty.anomaly("test_cifar_5labels_1_v1", "cifar", num_epochs = 50, inside_labels=[0,2,4,6,8], plot = False))
df = df.append(uncertainty.anomaly("test_cifar_5labels_2_v1", "cifar", num_epochs = 50, inside_labels=[1,3,5,7,9], plot = False))
df = df.append(uncertainty.anomaly("test_cifar_5labels_3_v1", "cifar", num_epochs = 50, inside_labels=[0,1,2,3,4], plot = False))

df = df.append(uncertainty.anomaly("test_cifar_8labels_1_v1", "cifar", num_epochs = 75, inside_labels=[2,3,4,5,6,7,8,9], plot = False))
df = df.append(uncertainty.anomaly("test_cifar_8labels_2_v1", "cifar", num_epochs = 75, inside_labels=[0,1,3,4,5,6,8,9], plot = False))
df = df.append(uncertainty.anomaly("test_cifar_8labels_3_v1", "cifar", num_epochs = 75, inside_labels=[0,1,2,3,4,5,6,7], plot = False))

df = df.append(uncertainty.anomaly("test_cifar_2labels_1_v2", "cifar", num_epochs = 25, inside_labels=[0,1], plot = False))
df = df.append(uncertainty.anomaly("test_cifar_2labels_2_v2", "cifar", num_epochs = 25, inside_labels=[2,7], plot = False))
df = df.append(uncertainty.anomaly("test_cifar_2lavels_3_v2", "cifar", num_epochs = 25, inside_labels=[8,9], plot = False))

df = df.append(uncertainty.anomaly("test_cifar_5labels_1_v2", "cifar", num_epochs = 50, inside_labels=[0,2,4,6,8], plot = False))
df = df.append(uncertainty.anomaly("test_cifar_5labels_2_v2", "cifar", num_epochs = 50, inside_labels=[1,3,5,7,9], plot = False))
df = df.append(uncertainty.anomaly("test_cifar_5labels_3_v2", "cifar", num_epochs = 50, inside_labels=[0,1,2,3,4], plot = False))

df = df.append(uncertainty.anomaly("test_cifar_8labels_1_v2", "cifar", num_epochs = 75, inside_labels=[2,3,4,5,6,7,8,9], plot = False))
df = df.append(uncertainty.anomaly("test_cifar_8labels_2_v2", "cifar", num_epochs = 75, inside_labels=[0,1,3,4,5,6,8,9], plot = False))
df = df.append(uncertainty.anomaly("test_cifar_8labels_3_v2", "cifar", num_epochs = 75, inside_labels=[0,1,2,3,4,5,6,7], plot = False))

df.to_csv("uncertainty_experiments.csv")