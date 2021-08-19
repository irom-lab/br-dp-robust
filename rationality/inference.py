from functools import partial
from typing import Callable, Union

import jax
import jax.numpy as jnp
import jax.random as rnd
from jax.experimental import optimizers


Kernel = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], float]
Statistic = Union[Callable[[jnp.ndarray], jnp.ndarray], Callable[[jnp.ndarray], float]]


@partial(jax.jit, static_argnums=0)
def _distmat(func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the distance matrix fro two sets of points.

    :param func: The function used to compute the distance
    :param x: An n-by-m array of m points in n-dimensional space.
    :param y: An n-by-p array of p points in n-dimensional space.
    :return: An m-by-p array where the entry at (i, j) is the distance between the i-th column of x and the j-th
             column of y.
    """
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1))(y.T))(x.T)


@jax.jit
def _euclidean(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    The Euclidean distance between two vectors.

    :param x: An n-dimensional array.
    :param y: An n-dimensional array.
    :return: The Euclidean distance between x and y.
    """
    return jnp.sqrt(((x - y) ** 2).sum())


@jax.jit
def bw_median_rule(samples: jnp.ndarray, m: int) -> float:
    """
    A bandwidth selection heuristic chosen to make the sum of the kernel function over points equal 1, where one input
    to the kernel is fixed to any point in the data set. Specifically, the bandwidth for m data points is:

        bandwidth = (median pairwise distance) ^ 2 / log(m).

    :param samples: An n-by-m array containing m samples in n-dimensional space.
    :param m: The number of samples (columns in samples). This quantity is broken out as a separate parameter
              to allow for JIT.
    :return: The bandwidth according to the heuristic.
    """
    pairwise_dists = _distmat(_euclidean, samples, samples)
    return jnp.median(pairwise_dists) ** 2 / jnp.log(m)


@jax.jit
def rbf_kernel(x: jnp.ndarray, y: jnp.ndarray, _samples: jnp.array, bw: float) -> float:
    """
    Radial basis function (RBF) kernel. This kernel is defined as:

        k(x, y) = exp(-(euclidean distance between x and y) ^ 2 / bandwidth).

    :param x: An n-dimensional array.
    :param y: An n-dimensional array.
    :param _samples: Unused by this kernel.
    :param bw: The bandwidth for the RBF kernel. A larger bandwidth produces a flatter bump.
    :return: The kernel evaluated at x and y.
    """
    d = jnp.sum((x - y) ** 2, axis=0)

    return jnp.exp(-d / bw)


@jax.jit
def rbf_dyn_bw_kernel(x: jnp.ndarray, y: jnp.ndarray, samples: jnp.array, m: int) -> float:
    """
    Radial basis function (RBF) kernel. This kernel is defined as:

        k(x, y) = exp(-(euclidean distance between x and y) ^ 2 / bandwidth).

    where bandwidth is computed using `bw_median_rule`.

    :param x: An n-dimensional array.
    :param y: An n-dimensional array.
    :param samples: The whole batch of samples of which x and y are members. The shape of this array is n-by-m, and
                    each column is a sample.
    :param m: The number of samples (columns in samples). This quantity is broken out as a separate parameter
              to allow for JIT.
    :return: The kernel evaluated at x and y.
    """
    bw = bw_median_rule(samples, m)

    d = jnp.sum((x - y) ** 2, axis=0)

    return jnp.exp(-d / bw)


@partial(jax.jit, static_argnums=(0, 1))
def impsamp(weight: Callable[[jnp.ndarray], float],
            statistic: Statistic, samples: jnp.ndarray) -> Union[float, jnp.ndarray]:
    """
    Perform (self-normalized) importance sampling on a data set to compute a statistic of the data.

    See "Machine Learning: A Probabilistic Perspective" by Murphy for background on importance sampling.

    Given a target distribution p(x) and a sampling distribution q(x), we would like to evaluate the mean of a
    statistic f(X) for p(x) using samples from q(x). To do so, define the IS weight function w(x) = p(x)/q(x).
    Then the mean of interest is approximated

        mean of f(x) = sum of f(x)w(x) over x / sum of w(x) over x.

    Moreover, w(x) only needs to be proportional to the ratio p(x)/q(x). When p(x) is a Gibbs measure with Hamiltonian
    H(x) and inverse temperature beta, the weight function is proportional to w(x) = exp(-beta * H(x)).

    :param weight: The importance sampling weight function w(x) that maps an n-dimensional x to a float.
    :param statistic: The function defining the statistic to be computed (see `Statistic` type for details).
    :param samples: An n-by-m array of samples. Each of the m samples is a column of the array.

    :return: The statistic, which may be an array of arbitrary size.
    """
    weights = jax.vmap(weight, in_axes=-1)(samples)
    statistic = jax.vmap(statistic, in_axes=-1, out_axes=-1)(samples)

    return jnp.average(statistic, axis=-1, weights=weights)


@partial(jax.jit, static_argnums=(0,))
def sir(log_prob: Callable[[jnp.ndarray], float],
        samples: jnp.ndarray,
        key: jnp.ndarray,
        returned_samples: int = 1) -> jnp.ndarray:
    """
    Perform sampling importance resampling on a data set.

    See "Machine Learning: A Probabilistic Perspective" by Murphy for background on importance resampling.

    :param log_prob: A function that maps an n-dimensional sample array
                     to the log-probability of the target distribution.
    :param samples: An n-by-m array of samples. Each of the m samples is a column of the array.
    :param key: A PRNGKey used to generate the random sample(s). Can be None if a statistic is provided.
    :param returned_samples: The number of samples to return.
    :return: If `statistic` is None, then the requested number of samples from the (approximate) target distribution are
             returned. Otherwise the mean of statistic is computed using the samples from the final approximate
             distribution and this quantity is returned.
    """
    logits = jax.vmap(log_prob, in_axes=-1)(samples)
    idxs = rnd.categorical(key, logits=logits, shape=(returned_samples,))

    return jnp.take(samples, idxs, axis=-1)


# TODO: Change def so samples are along axis -1.
@partial(jax.jit, static_argnums=(0, 1, 2, 4))
def sgvd(log_prob: Callable[[jnp.ndarray], float],
         kernel: Kernel,
         opt: optimizers.Optimizer,
         samples: jnp.ndarray,
         iters: int = 100) -> jnp.ndarray:
    """
    Perform inferance using Stein Variational Gradient Descent.

    Stein variational gradient descent is an optimization based inference algorithm that
    combines the benefits of MCMC (optimization) and importance sampling (parallelization) methods.
    It performs a kind of gradient descent (like MCMC) that converges a number of particles (like
    importance sampling) to a target distribution, but unlike MCMC this is done in a deterministic
    manner.

    References:
    - "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm" by Liu, Wang 16.

    - Reference implementation: https://github.com/DartML/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py

    :param log_prob: A function that maps a n-dimensional array to a float that is the log-density of the target
                     distribution up to a constant. Note that this is the log-probability for the target
                     distribution, which includes the log-probability of the prior.
    :param kernel: The kernel to use for calculating the gradient of the KL divergence between the
                   current iteration's distribution and the target distribution. Should take in
                   the two sample elements followed by the n-by-num_samples matrix of samples.
                   The latter of which can be used to dynamically adjust kernel parameters.
    :param opt: The optimization procedure to use.
    :param samples: An n-by-N array where each column is a sample in n-dimensional space. These samples are used
                    as the initial condition for the algorithm.
    :param iters: The number of optimization steps to take.

    :return: All samples from the (approximate) target distribution are returned.
    """
    lp_grad = jax.jit(jax.grad(log_prob))
    kern_grad = jax.jit(jax.grad(kernel, argnums=0))

    opt_init, opt_update, get_params = opt

    @jax.jit
    def kl_grad(particle: jnp.ndarray, batch: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the gradient with respect to a single particle for SVMPC.

        :param particle: The n-dimensional particle whose gradient is computed.
        :param batch: The whole batch of particles as n-by-N array, where N is the number of particles in the batch.

        :return: The gradient as an n-dimensional array.
        """
        kern_values = jax.vmap(lambda x: kernel(x, particle, batch), in_axes=1)(batch)
        lp_grad_values = jax.vmap(lp_grad, in_axes=1, out_axes=1)(batch)
        kern_grad_values = jax.vmap(lambda x: kern_grad(x, particle, batch), in_axes=1, out_axes=1)(batch)

        return -(jnp.multiply(kern_values, lp_grad_values) + kern_grad_values).mean(axis=1)

    @jax.jit
    def step_scanner(opt_state: optimizers.OptimizerState, step_iter: int) -> tuple[optimizers.OptimizerState, float]:
        batch = get_params(opt_state)
        kl_grad_full = jax.vmap(lambda x: kl_grad(x, batch), in_axes=1, out_axes=1)
        grad = kl_grad_full(batch)

        return opt_update(step_iter, grad, opt_state), 0.0

    init_opt_state = opt_init(samples)

    final_opt_state, _ = jax.lax.scan(step_scanner, init_opt_state, jnp.arange(iters))

    samples = get_params(final_opt_state)

    return samples
