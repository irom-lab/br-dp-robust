import unittest

import jax.numpy as jnp

import rationality.dynamics as dyn


class DynamicsTests(unittest.TestCase):
    def test_linear_dynamics(self):
        x = jnp.array([1.0, 2.0, 1.0])
        u = jnp.array([-1.0, 1.0])

        A = 2 * jnp.eye(3)
        B = jnp.array([[1.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 0.0]])

        dynamics = dyn.linear(A, B)
        x_next = dynamics(x, u, 0)

        self.assertTrue(jnp.all(x_next == jnp.array([2.0 - 1.0, 4.0 + 1.0, 2.0 - 1.0])))

    def test_linearize(self):
        mass = 1.0
        gravity = 9.8
        inertia = 1.0
        dt = 1.0

        x0 = jnp.zeros(6)
        u0 = jnp.array([mass * gravity, 0.0])

        dynamics = dyn.quad2d(dt, mass, gravity, inertia)
        params = dyn.linearize(dynamics, x0, u0, 0)

        A_true = jnp.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, -gravity, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        B_true = jnp.array([[0.0, 0.0],
                            [0.0, 0.0],
                            [0.0, 0.0],
                            [0.0, 0.0],
                            [1 / mass, 0.0],
                            [0.0, 1 / inertia]])

        self.assertTrue(jnp.allclose(A_true, params.A))
        self.assertTrue(jnp.allclose(B_true, params.B))


if __name__ == '__main__':
    unittest.main()
