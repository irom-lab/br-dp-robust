import unittest

import jax.numpy as jnp

import rationality.controllers as ctl
import rationality.dynamics as dyn
import rationality.objectives as obj
import rationality.simulate as sim


class ControllerTests(unittest.TestCase):
    def test_lqr(self):
        ic = jnp.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        horizon = 50

        mass = 1.0
        gravity = 9.8
        inertia = 1.0
        dt = 1.0

        x0 = jnp.zeros(6)
        u0 = jnp.array([mass * gravity, 0.0])

        dynamics = dyn.quad2d(dt, mass, gravity, inertia)
        params = dyn.linearize(dynamics, x0, u0, 0)

        Q = jnp.eye(6)
        R = jnp.eye(2)
        Qf = 10 * jnp.eye(6)

        dynamics = dyn.linear(*params)
        objective = obj.quadratic(Q, R, Qf)
        prob = ctl.problem(dynamics, objective, horizon)

        lqr = ctl.lqr(prob)

        simulation = sim.compile_simulation(prob, lqr)
        states, inputs, costs = sim.run(ic, jnp.zeros((6, horizon)), simulation, prob, lqr)

        self.assertAlmostEqual(ctl.lqr_cost_to_go(ic, 0, prob), costs.sum(), places=4)

        for t in range(horizon):
            x = states[:, t]

            self.assertAlmostEqual(ctl.lqr_cost_to_go(x, t, prob),
                                   ctl.util.cost_of_control_sequence(x, t, inputs[:, t:], prob), places=4)

        for t in range(horizon):
            x = states[:, t]
            appended_inputs = jnp.concatenate((inputs[:, t:], jnp.ones((2, 100))), axis=1)

            self.assertAlmostEqual(ctl.lqr_cost_to_go(x, t, prob),
                                   ctl.util.cost_of_control_sequence(x, t, appended_inputs, prob), places=4)


if __name__ == '__main__':
    unittest.main()
