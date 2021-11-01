from __future__ import annotations

from functools import partial
from typing import NamedTuple, Callable, Any, Union

import jax
import jax.numpy as jnp
import jax.random as rnd
import jax.scipy as jsp


def isdistparams(d: Any) -> bool:
    """
    Check if variable is a distribution parameter instance (e.g. instance of GaussianParams).

    :param d: The variable to check.
    :returns: True if d is a distribution parameter instance otherwise False.
    """
    return isinstance(d, (GaussianParams,))


class DistributionPrototype(NamedTuple):
    """
    Abstract representation of a probability distribution.

    Attributes:
        sample: The prototype of the function that generates samples from the distribution.
        log_prob: The prototype of the function that computes the log probability (density) for a given sample.
        size: The dimension of the sample space.
    """
    sample: SamplePrototype
    log_prob: LogProbPrototype
    size: int


class Distribution(NamedTuple):
    """
    A structure that contains a distribution's prototype and parameters along with a simplified API for calling
    the distribution's methods.

    Attributes:
        prototype: The structure defining the nature of the distribution (e.g. probability law, number of variables).
        params: Numerical parameters of the distribution.
    """
    prototype: DistributionPrototype
    params: DistributionParams

    def sample(self, num_samples: int, key: jnp.ndarray) -> jnp.ndarray:
        """
        Generate a sample or samples from the distribution.

        :param num_samples: The (positive) number of samples to generate.
        :param key: The JAX RNG key for generating the samples.

        :returns: If num_samples is 1, then an n-dimensional array. Otherwise an n-by-num_samples array.
        """
        return self.prototype.sample(num_samples, key, self.params)

    def log_prob(self, x: jnp.ndarray) -> float:
        """
        Computes the log probability (density) for a given value.

        :param x: The point at which to evaluate the log probability.

        :returns: The log probability (may be infinite).
        """
        return self.prototype.log_prob(x, self.params)


class GaussianParams(NamedTuple):
    """
    Parameters of the Gaussian distribution.

    Attributes:
        mean: Either an n-dimensional vector or an m-by-n array representing m different Multivariate Gaussians.
        cov: Either an n-by-n array (positive-definite) representing a covariance matrix or an m-by-n-by-n array
             representing the covariance matrices for m Multivariate Gaussians.
    """
    mean: jnp.ndarray
    cov: jnp.ndarray


class Gaussian(Distribution):
    prototype: DistributionPrototype
    params: GaussianParams


@partial(jax.jit, static_argnums=0)
def sample_gaussian(num_samples: int, key: jnp.ndarray, params: GaussianParams) -> jnp.ndarray:
    mean, cov = params

    return jax.vmap(lambda k: rnd.multivariate_normal(k, mean, cov))(rnd.split(key, num_samples)).T


@jax.jit
def log_prob_gaussian(x: jnp.ndarray, params: GaussianParams) -> jnp.ndarray:
    mean, cov = params

    return jsp.stats.multivariate_normal.logpdf(x, mean, cov)


class GaussianPrototype(DistributionPrototype):
    sample: SamplePrototype
    log_prob: LogProbPrototype
    size: int

    def __new__(cls, size: int):
        return super().__new__(cls, sample_gaussian, log_prob_gaussian, size)


def gaussian(mean: jnp.ndarray, cov: jnp.ndarray) -> Gaussian:
    return Gaussian(GaussianPrototype(mean.shape[0]), GaussianParams(mean, cov))


DistributionParams = Union[GaussianParams]
SamplePrototype = Callable[[int, jnp.ndarray, DistributionParams], jnp.ndarray]  #: Maps (num_samples, rng_key, params) -> sample
LogProbPrototype = Callable[[jnp.ndarray, DistributionParams], float]  #: Maps (sample, params) -> log_prob