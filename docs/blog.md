# Bayessian approach in neural networks for model pruning

## Intro

Using deep learning in solving complex, real-world problems has become quite an engineering routine. But we should never forget about probabalistic sense of our models and loss minimization. So here, we are going to recall that probabalistic framework and extend it to the bayessian framework. Such switching will give us pleasent perks but it is not always for free.

We will present 4 bayessian technics to envelope any task involving neural networks. As an application, we will show how this approaches can help prune our models. We will also look at the implemntation design of *bayessian NN*s and pruning algorithms in our [library]() based on [*pytorch*](). 

## Making neural network bayessian

The usual ML problem setup consist in minimizing the loss function $L$ between the train targets $y$ and model's output $f_{\mathbf{w}}(\mathbf{x})$. The model is some neural network parameterized by $\mathbf{w}$. Actually, the loss together with the model define data's distribution $p(y | \mathbf{x}, \mathbf{w})$. Minimizing the loss w.r.t. $\mathbf{w}$ is equavilent to finding maximum liklyhood estimation of model's parameters.

Now we introduce a prior distribtion on paramters: $p(\mathbf{w} | \Theta)$ . It is generally parameterized by *hyperparameters* $\Theta$ (but it can be fixed distribution too). It kind of changes the game because now target distribution for a **new** object $\hat{\mathbf{x}}$ is

$$
    p(y | \hat{\mathbf{x}}, \Theta) = \mathbb{E}_{p(\mathbf{w} | y, \mathbf{x}, \Theta)} [p(y | \hat{\mathbf{x}}, \mathbf{w})] , \label{new_point}\tag{1}
$$

where $p(\mathbf{w} | y, \mathbf{x}, \Theta)$ is *posterior* distribution of the model's parameters based on the train data. Unfortunately, in case of the NNs finfing the posterior is typically intractable. That leads to the intractability of the prediction. Another problem here is how to choose optimal hyperparameters $\Theta$ if we don't know them from some prior knowledge.

## Why Bayes may be useful?

In spite of the analytical difficulties, bayessian framework has lot to give. You've might already heard that [*L2 regularization*]() is equal to simple gaussian prior, and impose a greate computation stability during training. Futhermore, *pruning* of large nets is possible using the very same gaussian or more sparsity-inducing priors (e.g. [Laplace prior]()). 

The formula ($\ref{new_point}$) implies [*ansambling*]() your model to evaluate prediction for a new data point. It may correct the model if it uses fixed trained weights.

Bayes can also be used to perform [model selection](). In the end, it may be just unavoidable :), see [*hidden state models*]() and learning *mixture of gaussians* [example](). 

We have provided only a handful of applications, but this might be enough to start with. Now we will go through methods that will proof utility of the given theory.

## MCMC

...

## Variational inference

Here we are going to briefly discuss one of the ways to add a bayessian layer of inference to your nerual network - *variational inference* principle. We will see how this approach changes the learning objective and how it is implemented using [*pytorch*]() framework.

To tackle last issue of hyperparameters bayessian inference has special function called [*evidence*]() which is

$$
     p(y | \mathbf{x}, \Theta) = \mathbb{E}_{p(\mathbf{w} | y, \mathbf{x}, \Theta)} [p(y | \mathbf{x}, \mathbf{w})].
$$

It basically indicates how probable the given data under varying hyperparameters. Maximizing this function is a key to finding optimial $\Theta$.

But it can also help with the posterior! Following the [ELBO]() technic, we introduce *variational* distrubution $q(\mathbf{w} | \phi)$ which is parameterized by $\phi$. This distrubition is supposed to approximate the posterior $p(\mathbf{w} | y, \mathbf{x}, \Theta)$ and can be anything we want. Then, it can be shown that the following expression is a lower bound of the evidence

$$
    \text{ELBO}(\Theta, \phi) = \mathbb{E}_{q(\mathbf{w} | \phi)} [p(y | \mathbf{x}, \mathbf{w})] + \text{KL}(q(\mathbf{w} | \phi) || p(\mathbf{w} | \Theta))
$$

or in terms of the loss and the model

$$
    \text{ELBO}(\Theta, \phi) = \mathbb{E}_{q(\mathbf{w} | \phi)} [L(y, f_{\mathbf{w}}(\mathbf{x}))] + \text{KL}(q(\mathbf{w} | \phi) || p(\mathbf{w} | \Theta))
$$

where $\text{KL}(\cdot || \cdot)$ is a [KL-divirgence](). Maximizing it by $\phi$ and $\Theta$ gives us estimation of the optimal hyperparameters and the variational distrubution. 

## Using variational distribution

Actually, if you are not interested in hyperparamters, you can just use trained $q(\mathbf{w} | \phi^*)$ to get the desired prediction:

$$
    p(y | \hat{\mathbf{x}}, \Theta) \approx \mathbb{E}_{q(\mathbf{w} | \phi^*)} [p(y | \hat{\mathbf{x}}, \mathbf{w})].
$$

Moreover, you can find the [MAP]() estimation: $\mathbf{w}^* = \underset{\mathbf{w}}{\text{argmax }} q(\mathbf{w} | \phi^*)$, and estimate the expectation simply as

$$
    p(y | \hat{\mathbf{x}}, \Theta) \approx \mathbb{E}_{q(\mathbf{w} | \phi^*)} [p(y | \hat{\mathbf{x}}, \mathbf{w})] \approx p(y | \hat{\mathbf{x}}, \mathbf{w}^*),
$$

which is refered to just using your model $f_{\mathbf{w}}(\mathbf{x})$ with $\mathbf{w} = \mathbf{w}^*$.

## Choosing variational distribution

As it was mentioned there are no limits on $q(\mathbf{w} | \phi)$ except to be computable. But to make the whole thing practical and use *gradient optimization* it should comply with several requirements.

### [Reparametrization trick]()

In order to estimate and compute the gradient of the expectation in $\text{ELBO}(\Theta, \phi)$ the $q$ must separate the randomness from $\mathbf{w}$. Namely, we introduce some determenistic function $h$ parameterized by $\phi$ and some random variable $\epsilon$, usually with simple distribution $p(\epsilon)$ from which we can sample. So now, randomness of $\mathbf{w}$ is expressed through the randomness of $\epsilon$:

$$
    \mathbf{w} \sim q(\mathbf{w} | \phi) \Leftrightarrow \mathbf{w} = h(\mathbf{w}, \phi, \epsilon), \ \epsilon \sim p(\epsilon)
$$

This trick enables us to estimate the expectation and compute the gradients:

\begin{align}
    \mathbb{E}_{q(\mathbf{w} | \phi)} [p(y | \mathbf{x}, \mathbf{w})] &\approx \frac{1}{K} \sum_{i = 1}^K p(y | \mathbf{x}, \mathbf{w}_i) = \frac{1}{K} \sum_{i = 1}^K L(y, f_{\mathbf{w}_i}(\mathbf{x})), \\
    \nabla \mathbb{E}_{q(\mathbf{w} | \phi)} [p(y | \mathbf{x}, \mathbf{w})] &\approx \frac{1}{K} \sum_{i = 1}^K \nabla L(y, f_{\mathbf{w}_i}(\mathbf{x})).
\end{align}

where $\mathbf{w}_i = h(\mathbf{w}, \phi, \epsilon_i)$, $\epsilon_i \sim p(\epsilon)$ and $K$ is the number of samples.

### KL computation

To optimize $\text{ELBO}(\Theta, \phi)$ it is neccessary to compute KL term between $q(\mathbf{w} | \phi)$ and $p(\mathbf{w} | \Theta)$  plus gradients. As the choice of this distributions is arbitary and task-dependent, we do not discuss it futher. One general solution here can be sample estimation:

$$
    \text{KL}(q(\mathbf{w} | \phi) || p(\mathbf{w} | \Theta)) \approx \frac{1}{M} \sum_{i = 1}^M \log \frac{q(\mathbf{w}_i | \phi)}{p(\mathbf{w}_i | \Theta)},
$$

where $\mathbf{w}_i \sim q(\mathbf{w} | \phi)$.

### Posterior approximation

Ideally, the class of variational distibutions indexed by $\phi$ should contain $p(\mathbf{w} | y, \mathbf{x}, \Theta)$. If $q$ is exactly posterior then the ELBO will be exactly the evidence (not just lower bound)! 

Practicaly, we don't know exact posterior but we know it up to the normalization. It is followed from bayessian rule:

$$
    p(\mathbf{w} | y, \mathbf{x}, \Theta) \propto p(y | \mathbf{w}, \mathbf{x}) p(\mathbf{w} | \Theta)
$$

This can be a hint for choosing variational distibutions class. So if you know that $p(\mathbf{w} | y, \mathbf{x}, \Theta)$ have some special properties, make sure that functions from $q$ class have them too (for example, multimodality).

## Kroneker-factorized Laplace

## 

## Reference

