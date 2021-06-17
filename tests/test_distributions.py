import unittest

import jax
import jax.random as rnd
import jax.numpy as jnp

import rationality.distributions as dst


class DistributionsTests(unittest.TestCase):
    def test_gaussian(self):
        key = rnd.PRNGKey(0)

        A = jnp.array([[1.0, 3.0], [2.0, 4.0]])
        cov = A @ A.T
        mean = jnp.array([6.0, 2.0])

        gaussian = dst.gaussian(mean, cov)
        samples = gaussian.sample(100000, key)

        jax.vmap(lambda x: gaussian.log_prob(x), in_axes=-1)(samples)

        self.assertTrue(jnp.allclose(mean, samples.mean(axis=1), 0.0, 0.1))
        self.assertTrue(jnp.allclose(cov, jnp.cov(samples), 0.0, 0.2))


if __name__ == '__main__':
    unittest.main()
