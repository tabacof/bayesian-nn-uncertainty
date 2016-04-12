# Classification uncertainty using Bayesian neural networks

We are using Bayesian neural networks for class anomaly detection. For example, say you have a network that classifies cats and dogs. If you put the image of an ostrich as input, shouldn't the network give some kind of hint that it doesn't understand ostriches? We are testing whether Bayesian neural networks can accomplish that via uncertainty information.

We are trying out two Bayesian approaches:
* Yarin Gal's dropout approximation - see [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](http://arxiv.org/abs/1506.02142)
* Variational approximation using normal posteriors - see [Weight Uncertainties in Neural Networks](http://arxiv.org/abs/1505.05424)

In the future we may work with HMC approaches for a complete Bayesian inference.

The class anomalies are detected using (marginalized) entropy and prediction variance thresholds. The thresholds are found in a supervised manner.

See the notebooks for intermediate results. This is a work in progress.
