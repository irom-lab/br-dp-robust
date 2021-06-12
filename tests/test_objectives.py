import unittest

import jax
import jax.numpy as jnp

import rationality.objectives as obj


class ObjectiveTests(unittest.TestCase):
    def test_quad_obj(self):
        Q = jnp.eye(2)
        R = jnp.eye(1)
        Qf = 10 * jnp.eye(2)
        horizon = 3

        x = jnp.array([1.0, 2.0])
        u = jnp.array([3.0])

        objective = obj.create_objective(obj.quad_obj_prototype, (Q, R, Qf, horizon))

        objective_evaluations = jnp.array([objective(x, u, t) for t in range(10)])
        answers = jnp.array([x.T @ Q @ x + u.T @ R @ u for t in range(3)] + [x.T @ Qf @ x] + [0.0] * 6)

        self.assertTrue(jnp.all(objective_evaluations == answers))


if __name__ == '__main__':
    unittest.main()
