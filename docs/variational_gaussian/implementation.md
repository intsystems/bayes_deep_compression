# Implementation Design

Implementation of variational inference for NN requires modification of the ```nn.Module``` and a user's loss function.

All submodules of the user's ```Module``` should now sample their weights every forward pass. Module's ```forward``` function should accept new argument - number of samples from posterior to estimate data's liklyhood. The Module should also be able to return MAP-pruned estimator which will be an ordinary ```nn.Module```. This is supposed to be realized through metaclass [```withGaussianPrior```](../../src/variational_gaussian/prior.py).

Loss function is gonna be modified through the [```addBayessianLoss```](../../src/variational_gaussian/loss.py) envelope. It will add KL-diveregence to the user's loss so the final functional to optimize is ELBO.

The main feature of the design is that user stays in the torch framework. From the outside, network works as usual and the training loop is the same as it would be for ordinary ```Module```.Features of bayessian inference in the ```forward()``` and pruning of the model are added.
