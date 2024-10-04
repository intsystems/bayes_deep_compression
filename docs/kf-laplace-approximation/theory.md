
```
    You probably won't use this method in production. Yet until now(2024) this method used for scientific research.
```

## Literature

Articles:
1) A Scalable Laplace Approximation for Neural Networks https://openreview.net/forum?id=Skdvd2xAZ
2) Practical Gauss-Newton Optimisation for Deep Learning https://arxiv.org/pdf/1706.03662

You probably want to start from second article is it contain rigorous proofs and explanation of numerical method.

Probably you will also find useful article from [medium](https://towardsdatascience.com/introducing-k-fac-and-its-application-for-large-scale-deep-learning-4e3f9b443414)

## Main insights: 
1) Hessian of even small network is intractable -> we need seamless approximation for numerical computation.
2) We can abuse architecture of neural network organized in consequential layers. Therefore, if we put net in one vector $\text{vec}(W)$, it's hessian will be mostly diagonal. 
3) Rigorous exploitation of this fact is called kronecker approximation and can be wrapped with tensor algebra tricks, which mostly about fact that inversion of diagonal matrix is just putting diagonal elements to denominator.
4) Bayessian framework use approximate hessian for modeling probability density of neural net weights. $ W \sim MN(W,H^{-1})$



## Recap (Skip freely if you are well acknowledged with optimization methods):
Recall that in deep learning hardest part is selecting learning rate. Second order methods provides framework for finding optimal steps. Recall 

$$
    f(\phi + \delta x) =
$$

Deriving $\delta x = H^{-1} \nabla_\theta f$

This is it, Hessian provides convenience for selecting steps in gradient methods. [Gauss-Newton method](https://en.m.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm) provides further convenience via seamless approximation:

$$
    \frac{\partial log(p)}{\partial^2 \theta} = \frac{\nabla_\theta p \nabla_\theta p}{p} + \pa
$$

Note that approximation is rigid under condition:
$$
    p \frac{\partial^2 p}{\partial^2 \theta} \gg  (\frac{\partial p}{\partial \theta })^2
$$

As you would expect authors doesn't burden themself with  proper reasoning. 


## Actual steps of 

Before hessian inversion we actually need to calculate it:

![./static/]()