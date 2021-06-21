from functools import partial
from typing import NamedTuple, Callable, Any

import jax
import jax.numpy as jnp
import jax.random as rnd
import jax.scipy as jsp

SamplePrototype = Callable[[int, jnp.ndarray, Any], jnp.ndarray]
LogProbPrototype = Callable[[jnp.ndarray, Any], float]


class DistributionPrototype(NamedTuple):
    sample: SamplePrototype
    log_prob: LogProbPrototype
    size: int


class Distribution(NamedTuple):
    prototype: DistributionPrototype
    params: Any

    def sample(self, num_samples: int, key: jnp.ndarray) -> jnp.ndarray:
        return self.prototype.sample(num_samples, key, self.params)

    def log_prob(self, x: jnp.ndarray) -> float:
        return self.prototype.log_prob(x, self.params)


class GaussianParams(NamedTuple):
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