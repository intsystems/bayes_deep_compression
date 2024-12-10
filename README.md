![Tests](https://github.com/intsystems/bayes_deep_compression/actions/workflows/variational_tests.yml/badge.svg)
![Docs](https://github.com/intsystems/bayes_deep_compression/actions/workflows/gh-pages.yml/badge.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

# BayesComp

This python library is an extension of *pytorch*:fire: for transforming ordinary neural networks into **Bayesian**. Why? Check out our [post](https://intsystems.github.io/bayes_deep_compression/blog/) for motivation and approaches in this topic. You will find here foundamental ideas and practical considerations about Bayesian inference.

## Documantion

Our docs pages, blog and more are available on [Github Pages](https://intsystems.github.io/bayes_deep_compression/).

## Examples

See [`examples/`](examples/) for basic building of Bayesian nets and their training.

## Installation

Make sure you have [poetry](https://python-poetry.org/docs/) installed and run the following in the project's root

```
    poetry install
```

## Tests

To run tests with coverage

```
    poetry run pytest --cov=src tests/
```

## Authors

- Ilgam Latypov
- Alexander Terentyev
- Kirill Semkin
- Nikita Mashalov

## References

Here are the works upon which this library is built

1. Graves, A. (2011). Practical Variational Inference for Neural Networks. In Advances in Neural Information Processing Systems. Curran Associates, Inc.

2. Christos Louizos, Karen Ullrich, & Max Welling. (2017). Bayesian Compression for Deep Learning.

3. Hippolyt Ritter, Aleksandar Botev, & David Barber (2018). A Scalable Laplace Approximation for Neural Networks. In International Conference on Learning Representations.

4. Yingzhen Li, & Richard E. Turner. (2016). Renyi Divergence Variational Inference.
