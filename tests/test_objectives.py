import unittest

import jax.numpy as jnp

import rationality.objectives as obj


class ObjectiveTests(unittest.TestCase):
    def test_quad_obj(self):
        Q = jnp.eye(2)
        R = jnp.eye(1)
        Qf = 10 * jnp.eye(2)

        x = jnp.array([1.0, 2.0])
        u = jnp.array([3.0])

        objective = obj.quadratic(Q, R, Qf)

        objective_evaluations = jnp.array([objective(x, u, 3), objective(x)])
        answers = jnp.array([x.T @ Q @ x + u.T @ R @ u] + [x.T @ Qf @ x])

        self.assertTrue(jnp.all(objective_evaluations == answers))


if __name__ == '__main__':
    unittest.main()
