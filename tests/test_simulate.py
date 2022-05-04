import unittest, os, sys

import jax
import jax.numpy as jnp
import jax.random as rnd

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

import rationality.controllers as ctl
import rationality.dynamics as dyn
import rationality.objectives as obj
import rationality.simulate as sim
import rationality.distributions as dst


class SimulateTests(unittest.TestCase):
    def test_stopping_condition(self):
        ic = jnp.array([0.5, 0.0])
        horizon = 5

        A = jnp.array([[1.0, 1.0], [0.0, 1.0]])
        B = jnp.array([[0.0], [1.0]])
        Q = jnp.eye(2)
        R = jnp.eye(1)
        Qf = 2 * jnp.eye(2)

        dynamics = dyn.linear(A, B)
        objective = obj.quadratic(Q, R, Qf)


        prob = ctl.problem(dynamics, objective, horizon, )
        olc = ctl.open_loop(jnp.array([0.25] + (horizon - 1) * [0.0]).reshape(-1, 1))

        simulation = sim.compile_simulation(prob, olc)
        states, inputs, costs, stopping_time = simulation(ic, jnp.zeros((2, horizon)))

        self.assertTrue(jnp.allclose(states[:, stopping_time], jnp.array([1.5, 0.25])))
        self.assertTrue(costs[-1] == 0.5 * states[:, stopping_time].T @ Qf @ states[:, stopping_time])

        simulation = sim.compile_simulation(prob, olc, jax.jit(lambda x, u, t, x_next: jnp.abs(x_next[0]) >= 1.0))
        states, inputs, costs, stopping_time = simulation(ic, jnp.zeros((2, horizon)))

        self.assertTrue(stopping_time == 2)
        self.assertTrue(costs[-1] == 0.5 * states[:, stopping_time].T @ Qf @ states[:, stopping_time])

        pass
