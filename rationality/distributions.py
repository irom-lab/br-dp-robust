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


class ProductPrototype(DistributionPrototype):
    sample: SamplePrototype
    log_prob: LogProbPrototype
    size: int


@partial(jax.jit, static_argnums=0)
def sample_gaussian(num_samples: int, key: jnp.ndarray, params: GaussianParams) -> jnp.ndarray:
    return jax.vmap(lambda k: rnd.multivariate_normal(k, params.mean, params.cov))(rnd.split(key, num_samples)).T


@jax.jit
def log_prob_gaussian(x: jnp.ndarray, params: GaussianParams) -> jnp.ndarray:
    return jsp.stats.multivariate_normal.logpdf(x, params.mean, params.cov)


class GaussianPrototype(DistributionPrototype):
    sample: SamplePrototype
    log_prob: LogProbPrototype
    size: int

    def __new__(cls, size: int):
        return super().__new__(cls, sample_gaussian, log_prob_gaussian, size)


def gaussian(mean: jnp.ndarray, cov: jnp.ndarray) -> Gaussian:
    return Gaussian(GaussianPrototype(mean.shape[0]), GaussianParams(mean, cov))


def sample_product(num_samples: int, key: jnp.ndarray, params: Any,
                   elem_proto: DistributionPrototype) -> jnp.ndarray:
    samples = jax.lax.scan(lambda _, p: (None, elem_proto.sample(num_samples, key, p)), None, params)

    return jnp.transpose(samples, axes=(1, 2, 0))


def log_prob_product(x: jnp.ndarray, params: GaussianParams, elem_proto: DistributionPrototype) -> jnp.ndarray:
    log_probs = jax.vmap(elem_proto.log_prob, in_axes=(-1, -1))(params)

    return jnp.sum(log_probs)


def product_prototype(elem_proto: DistributionPrototype) -> ProductPrototype:
    return ProductPrototype(
        jax.jit(lambda num_samples, key, params: sample_product(num_samples, key, params, elem_proto)),
        jax.jit(lambda x, params: log_prob_product(x, params, elem_proto)),
        elem_proto.size)
