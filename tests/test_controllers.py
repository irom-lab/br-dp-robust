import unittest

import jax
import jax.numpy as jnp
import jax.random as rnd

import rationality.controllers as ctl
import rationality.dynamics as dyn
import rationality.objectives as obj
import rationality.simulate as sim
import rationality.distributions as dst


def make_lin_sys(ic: jnp.ndarray, horizon: int) -> ctl.Problem:
    mass = 1.0
    gravity = 9.8
    inertia = 1.0
    dt = 0.25

    x0 = jnp.zeros(6)
    u0 = jnp.array([mass * gravity, 0.0])

    dynamics = dyn.quad2d(dt, mass, gravity, inertia)
    params = dyn.linearize(dynamics, x0, u0, 0)

    Q = jnp.eye(6)
    R = jnp.eye(2)
    Qf = 10 * jnp.eye(6)

    dynamics = dyn.linear(*params)
    objective = obj.quadratic(Q, R, Qf)
    return ctl.problem(dynamics, objective, horizon)


class ControllerTests(unittest.TestCase):
    def test_lqr(self):
        ic = jnp.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        horizon = 50

        prob = make_lin_sys(ic, horizon)

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

    def test_isc(self):
        ic = jnp.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        horizon = 5

        prob = make_lin_sys(ic, horizon)

        lqr = ctl.lqr(prob)

        simulation = sim.compile_simulation(prob, lqr)
        _, lqr_inputs, lqr_costs = sim.run(ic, jnp.zeros((6, horizon)), simulation, prob, lqr)

        cov = jnp.diag(jnp.array([1e-0, 1e-0] * horizon) ** 2)

        prior_params = [dst.GaussianParams(jnp.pad(lqr_inputs[:, t:].flatten(order='F'),
                                                   (0, t * prob.num_inputs)),
                                           cov) for t in range(horizon)]

        isc = ctl.isc(prob, jnp.inf, 100000, rnd.PRNGKey(0), dst.GaussianPrototype(prob.num_inputs), prior_params)

        simulation = sim.compile_simulation(prob, isc)
        _, isc_inputs, isc_costs = sim.run(ic, jnp.zeros((6, horizon)), simulation, prob, isc)
        self.assertAlmostEqual(isc_costs.sum(), lqr_costs.sum(), places=0)



if __name__ == '__main__':
    unittest.main()
