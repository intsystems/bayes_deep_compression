# Bayessian approach in neural networks for model pruning

## Intro

Using deep learning in solving complex, real-world problems has become quite an engineering routine. But we should never forget about probabalistic sense of our models and loss minimization. So here, we are going to recall that probabalistic framework and extend it to the bayessian framework. Such switching will give us pleasent perks but it is not always for free.

We will present 4 bayessian technics to envelope any task involving neural networks. As an application, we will show how this approaches can help prune our models. We will also look at the implemntation design of *bayessian NN*s and pruning algorithms in our [library]. 

## Making neural network bayessian

The usual ML problem setup consist in minimizing the loss function $L$ between the train targets $y$ and model's output $f_{\mathbf{w}}(\mathbf{x})$. The model is some neural network parameterized by $\mathbf{w}$. So the loss together with the model define data's distribution $p(y | \mathbf{x}, \mathbf{w})$. Minimizing the loss w.r.t. $\mathbf{w}$ is equavilent to finding maximum liklyhood estimation of model's parameters.

The bayessian approach complements the model with a prior distribtion on paramters $p(\mathbf{w} | \Theta)$ . It is generally parameterized by *hyperparameters* $\Theta$ (but it can be fixed distribution too). This move kind of changes the game because now target distribution for a **new** object $\hat{\mathbf{x}}$ is

$$
    p(y | \hat{\mathbf{x}}, \Theta) = \mathbb{E}_{p(\mathbf{w} | y, \mathbf{x}, \Theta)} [p(y | \hat{\mathbf{x}}, \mathbf{w})] , \label{new_point}\tag{1}
$$

where $p(\mathbf{w} | y, \mathbf{x}, \Theta)$ is a *posterior* distribution of the model's parameters based on the train data. Unfortunately, finding the posterior is typically intractable in case of the NNs. That leads to the intractability of the prediction. Another problem here is how to choose optimal hyperparameters $\Theta$ if we don't know them from some expert (prior) knowledge.

## Why Bayes may be useful?

In spite of the analytical difficulties, bayessian framework has a lot to give. You've might already heard that [*L2 regularization*]() is equal to simple gaussian prior. Futhermore, *pruning* of large nets is possible using the very same gaussian or more sparsity-inducing priors (e.g. [Laplace prior]()).

Some other bayes features:

- The formula ($\ref{new_point}$) implies [*ensambling*]() your model to evaluate prediction for a new data point. Therefore more information about true state is used in the final prediction. It also prevents models from being [*over-confident*](https://docs.giskard.ai/en/latest/knowledge/key_vulnerabilities/overconfidence/index.html).

- Bayes can be used to perform [*model selection*](), see [*hidden state models*]() and learning *mixture of gaussians* [example](). 

We have provided only a handful of applications, but this might be enough to start with. Now, we will go through methods that will prove utility of the given theory.

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

### Using variational distribution

Actually, if you are not interested in hyperparamters, you can just use trained $q(\mathbf{w} | \phi^*)$ to get the desired prediction:

$$
    p(y | \hat{\mathbf{x}}, \Theta) \approx \mathbb{E}_{q(\mathbf{w} | \phi^*)} [p(y | \hat{\mathbf{x}}, \mathbf{w})].
$$

Moreover, you can find the [MAP]() estimation: $\mathbf{w}^* = \underset{\mathbf{w}}{\text{argmax }} q(\mathbf{w} | \phi^*)$, and estimate the expectation simply as

$$
    p(y | \hat{\mathbf{x}}, \Theta) \approx \mathbb{E}_{q(\mathbf{w} | \phi^*)} [p(y | \hat{\mathbf{x}}, \mathbf{w})] \approx p(y | \hat{\mathbf{x}}, \mathbf{w}^*),
$$

which is refered to just using your model $f_{\mathbf{w}}(\mathbf{x})$ with $\mathbf{w} = \mathbf{w}^*$.

### Choosing variational distribution

As it was mentioned there are no limits on $q(\mathbf{w} | \phi)$ except to be computable. But to make the whole thing practical and use *gradient optimization* it should comply to several requirements.

#### [Reparametrization trick]()

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

#### KL computation

To optimize $\text{ELBO}(\Theta, \phi)$ it is neccessary to compute KL term between $q(\mathbf{w} | \phi)$ and $p(\mathbf{w} | \Theta)$  plus gradients. As the choice of this distributions is arbitary and task-dependent, we do not discuss it futher. One general solution here can be sample estimation:

$$
    \text{KL}(q(\mathbf{w} | \phi) || p(\mathbf{w} | \Theta)) \approx \frac{1}{M} \sum_{i = 1}^M \log \frac{q(\mathbf{w}_i | \phi)}{p(\mathbf{w}_i | \Theta)},
$$

where $\mathbf{w}_i \sim q(\mathbf{w} | \phi)$.

#### Posterior approximation

Ideally, the class of variational distibutions indexed by $\phi$ should contain $p(\mathbf{w} | y, \mathbf{x}, \Theta)$. If $q$ is exactly posterior then the ELBO will be exactly the evidence (not just lower bound)! 

Practicaly, we don't know exact posterior but we know it up to the normalization. It is followed from bayessian rule:

$$
    p(\mathbf{w} | y, \mathbf{x}, \Theta) \propto p(y | \mathbf{w}, \mathbf{x}) p(\mathbf{w} | \Theta)
$$

This can be a hint for choosing variational distibutions class. So if you know that $p(\mathbf{w} | y, \mathbf{x}, \Theta)$ have some special properties, make sure that functions from $q$ class have them too (for example, multimodality).

### Pruning strategy

Pruning is generally based on the probaility mass of the $q$-distribution in the $w_i = 0$ points. For example, if $q$ is factorized gaussian (Graves) then $\log q(w_i = 0) \propto -\dfrac{\mu_i^2}{2 \sigma_i^2}$. Set upper threshold on this value and we obtain pruning rule for individual weights:

$$
    \left| \dfrac{\mu_i}{\sigma_i} \right| < \lambda.
$$

Similar rules can be derived for more tricky distributions like *log-uniform* or *half-Cauchy* (Christos Louizos). The main practical issue here is whatever $q$ you choose it should **factorize** weights into small groups. Otherwise it would be impossible to compute marginals in reasonable time even with rather small nets.

## Renui-divirgence



## Kroneker-factorized Laplace

The key advantage of this approach is that it is applicable to **trained** NNs with *layer structure*. Imagine we have no prior for now, only liklyhood $\log p(y | \mathbf{w}, \mathbf{x})$. We can use second order approximation around its maximum $\mathbf{w}^*$

$$
    \log p(y | \mathbf{w}, \mathbf{x}) \approx \log p(y | \mathbf{w}^*, \mathbf{x}) + (\mathbf{w} - \mathbf{w}^*)^{\text{T}} H (\mathbf{w} - \mathbf{w}^*).
$$

Here $H$ is a hessian of the liklyhood in the maximum point that we don't know. However, we do know $\mathbf{w}^*$ (it is our trained model). The approximation gives us normal distribution for net's parameters

$$
     \mathbf{w} \sim \mathcal{N}(\mathbf{w}^*, H^{-1}).
$$

Actually, the result would be the same even if we had some fixed prior in the begining. In this case, the liklyhood function is substitued for posterior $p(\mathbf{w} | y, \mathbf{x})$. Now $\mathbf{w}^*$ is the MAP model, $H$ is the hessian of the posterior. Simple illustration of this transition is adding L2-regularization into net's training.

### Pruning strategy

If we knew $H$ and $\mathbf{w}^*$, pruning could be based on the probabilty mass in $w_i = 0$.So it is similar to the pruning in the [variational approach](). The major concern here is again factorization into parameters groups, namely by NN's layers. Hippolyt Ritter et. al. showed that if we assume layer's independence, the hessian will factorize into block-diagonal matrix. Therefore we will have independent normals for each layer! Computing marginals or testing layer parts to be zero is now absolutely feasable.

### Hessian factorization

Some comments on the mentioned hessian factoriztion. Denote hessian on the layer $\lambda$ as $H_{\lambda}$, then

$$
    \mathbb{E}[H_{\lambda}] = \mathbb{E}[\mathcal{Q}_{\lambda}] \otimes \mathbb{E}[\mathcal{H}_{\lambda}],
$$

where $\mathcal{Q}_{\lambda} = a_{\lambda-1}^{\text{T}} a_{\lambda-1}$ is covariance of the incoming activations $a_{\lambda-1}$ and $\mathcal{H}_{\lambda} = \dfrac{\partial^2 L}{\partial h_{\lambda} \partial h_{\lambda}}$ is the Hessian of the loss w.r.t. the linear pre-activations $h_{\lambda}$. 

Expectations are alwayes changed for sample estimations. Heavy $\mathcal{H}_{\lambda}$ can be estimated using KFRA or KFAC algorithms. In terms of the implementation these are most cumbersome.

Ultimately, the distribution on layer weights is [matrix normal]()

$$
    \mathbf{w}_{\lambda} \sim \mathcal{MN}(\mathbf{w}^*_{\lambda}, \mathcal{Q}_{\lambda}^{-1}, \mathcal{H}_{\lambda}).
$$

## Reference

