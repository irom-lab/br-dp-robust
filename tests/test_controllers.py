import unittest

import jax
import jax.numpy as jnp
import jax.random as rnd
import jax.experimental.optimizers as opt

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


def make_nonlin_sys(horizon: int) -> ctl.Problem:
    dt = 0.3

    dynamics = dyn.crazyflie2d(dt)

    Q = jnp.zeros((6, 6))
    R = 0.001 * jnp.zeros((2, 2))
    Qf = 10 * jnp.eye(6)

    objective = obj.quadratic(Q, R, Qf)

    return ctl.problem(dynamics, objective, horizon)


class ControllerTests(unittest.TestCase):
    def test_lqr(self):
        ic = jnp.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        horizon = 25

        prob = make_lin_sys(ic, horizon)

        lqr = ctl.lqr.create(prob)

        simulation = sim.compile_simulation(prob, lqr)
        states, inputs, costs = sim.run(ic, jnp.zeros((6, horizon)), simulation, prob, lqr)

        self.assertAlmostEqual(ctl.lqr.cost_to_go(ic, 0, prob), costs.sum(), places=4)

        for t in range(horizon):
            x = states[:, t]

            self.assertAlmostEqual(ctl.lqr.cost_to_go(x, t, prob),
                                   ctl.util.cost_of_control_sequence(x, t, inputs[:, t:], prob), places=3)

        for t in range(horizon):
            x = states[:, t]
            appended_inputs = jnp.concatenate((inputs[:, t:], jnp.ones((2, 100))), axis=1)

            self.assertAlmostEqual(ctl.lqr.cost_to_go(x, t, prob),
                                   ctl.util.cost_of_control_sequence(x, t, appended_inputs, prob), places=3)

    def test_isc(self):
        ic = jnp.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        horizon = 5

        prob = make_lin_sys(ic, horizon)

        lqr = ctl.lqr.create(prob)

        simulation = sim.compile_simulation(prob, lqr)
        _, lqr_inputs, lqr_costs = sim.run(ic, jnp.zeros((6, horizon)), simulation, prob, lqr)

        cov = jnp.diag(jnp.array([1e-0, 1e-0] * horizon) ** 2)

        prior_params = [dst.GaussianParams(jnp.pad(lqr_inputs[:, t:].flatten(order='F'),
                                                   (0, t * prob.num_inputs)),
                                           cov) for t in range(horizon)]

        isc = ctl.isc.create(prob, jnp.inf, 100000, rnd.PRNGKey(0), dst.GaussianPrototype(prob.num_inputs),
                             prior_params)

        simulation = sim.compile_simulation(prob, isc)
        _, isc_inputs, isc_costs = sim.run(ic, jnp.zeros((6, horizon)), simulation, prob, isc)
        self.assertAlmostEqual(isc_costs.sum(), lqr_costs.sum(), places=0)

    def test_mpc(self):
        ic = jnp.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        horizon = 10

        prob = make_lin_sys(ic, horizon)

        lqr = ctl.lqr.create(prob)
        mpc = ctl.mpc.create(prob, opt.adam(1e-1), 1000)

        lqr_simulation = sim.compile_simulation(prob, lqr)
        mpc_simulation = sim.compile_simulation(prob, mpc)

        lqr_states, lqr_inputs, lqr_costs = sim.run(ic, jnp.zeros((6, horizon)), lqr_simulation, prob, lqr)
        mpc_states, mpc_inputs, mpc_costs = sim.run(ic, jnp.zeros((6, horizon)), mpc_simulation, prob, mpc)

        self.assertLess(jnp.abs(mpc_inputs.T - lqr_inputs.T).max(), 0.001)

    def test_mpc_nonlinear(self):
        ic = jnp.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        horizon = 12

        prob = make_nonlin_sys(horizon)

        mpc = ctl.mpc.create(prob, opt.adam(1e-0), 1000)

        mpc_simulation = sim.compile_simulation(prob, mpc)

        with jax.disable_jit():
            mpc_states, mpc_inputs, mpc_costs = sim.run(ic, jnp.zeros((6, horizon)), mpc_simulation, prob, mpc)

        self.assertLess(jnp.abs(mpc_states[:, -1].max()), 0.006)


if __name__ == '__main__':
    unittest.main()
