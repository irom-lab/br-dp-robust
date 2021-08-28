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
import rationality.inference as inf


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

        self.assertAlmostEqual(ctl.lqr.cost_to_go(prob, ic, t=0), costs.sum(), places=4)

        for t in range(horizon):
            x = states[:, t]

            self.assertAlmostEqual(ctl.lqr.cost_to_go(prob, x, t=t),
                                   ctl.util.cost_of_control_sequence(x, t, inputs[:, t:], prob), places=3)

        for t in range(horizon):
            x = states[:, t]
            appended_inputs = jnp.concatenate((inputs[:, t:], jnp.ones((2, 100))), axis=1)

            self.assertAlmostEqual(ctl.lqr.cost_to_go(prob, x, t=t),
                                   ctl.util.cost_of_control_sequence(x, t, appended_inputs, prob), places=3)

    def test_lqbr_vs_lqr(self):
        ic = jnp.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        horizon = 25
        inv_temp = 1000.0
        key = jax.random.PRNGKey(0)

        prob = make_lin_sys(ic, horizon)

        lqr = ctl.lqr.create(prob)
        lqr_sim = sim.compile_simulation(prob, lqr)

        lqr_states, lqr_inputs, lqr_costs = sim.run(ic, jnp.zeros((6, horizon)), lqr_sim, prob, lqr)

        key, subkey = jax.random.split(key)
        prior_params = [dst.GaussianParams(lqr_inputs[:, t], 10 * jnp.eye(2)) for t in range(horizon)]
        lqbr = ctl.lqbr.create(prob, prior_params, inv_temp, key)
        lqbr_sim = sim.compile_simulation(prob, lqbr)

        lqbr_states, lqbr_inputs, lqbr_costs = sim.run(ic, jnp.zeros((6, horizon)), lqbr_sim, prob, lqbr)

        self.assertLess(lqbr_costs.sum() - lqr_costs.sum(), 0.25)

    def test_lqbr_vs_is(self):
        ic = jnp.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        prior_ic = jnp.array([1.5, -1.5, 0.0, 0.0, 0.0, 0.0])
        horizon = 2
        inv_temp = 1.0
        is_samples = 1000000

        prob = make_lin_sys(ic, horizon)
        cost_of_control_sequence = ctl.util.compile_cost_of_control_sequence(prob)

        lqr = ctl.lqr.create(prob)
        lqr_sim = sim.compile_simulation(prob, lqr)
        lqr_states, lqr_inputs, lqr_costs = sim.run(prior_ic, jnp.zeros((6, horizon)), lqr_sim, prob, lqr)

        prior_params = [dst.GaussianParams(lqr_inputs[:, t], 1.0 * jnp.eye(2)) for t in range(horizon)]
        is_prior_params = dst.GaussianParams(lqr_inputs.flatten(order='F'), 1.0 * jnp.eye(2 * horizon))

        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        samples = dst.gaussian(*is_prior_params).sample(is_samples, subkey)

        approx_mean =  inf.impsamp(lambda u: jnp.exp(-inv_temp * cost_of_control_sequence(ic, 0, u.reshape((2, -1), order='F'))), lambda u: u[:2], samples)

        approx_cov = inf.impsamp(
            lambda u: jnp.exp(-inv_temp * cost_of_control_sequence(ic, 0, u.reshape((2, -1), order='F'))),
            lambda u: (u[:2].reshape((-1, 1)) @ u[:2].reshape((1, -1)) - approx_mean.reshape((-1, 1)) @ approx_mean.reshape((1, -1))), samples)

        key, subkey = jax.random.split(key)
        lqbr = ctl.lqbr.create(prob, prior_params, inv_temp, key)

        _, temp_info = lqbr.init(prob.params)
        exact_mean = temp_info[0][0, :, :] @ ic + temp_info[1][0][0, :]
        exact_cov = temp_info[1][1][0, :, :]

        lqbr_sim = sim.compile_simulation(prob, lqbr)
        lqbr_states, lqbr_inputs, lqbr_costs = sim.run(ic, jnp.zeros((6, horizon)), lqbr_sim, prob, lqbr)

        self.assertLess(jnp.linalg.norm(exact_mean - approx_mean), 0.18)
        self.assertLess(jnp.linalg.norm(exact_cov - approx_cov, ord='fro'), 0.15)

    def test_lqbr_cost_to_go(self):
        ic = jnp.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        prior_ic = jnp.array([1.5, -1.5, 0.0, 0.0, 0.0, 0.0])
        horizon = 50
        inv_temp = 1.0
        trials = 1000000

        prob = make_lin_sys(ic, horizon)

        lqr = ctl.lqr.create(prob)
        lqr_sim = sim.compile_simulation(prob, lqr)
        lqr_states, lqr_inputs, lqr_costs = sim.run(prior_ic, jnp.zeros((6, horizon)), lqr_sim, prob, lqr)

        key = jax.random.PRNGKey(0)
        prior_params = [dst.GaussianParams(lqr_inputs[:, t], 10 * jnp.eye(2)) for t in range(horizon)]
        lqbr = ctl.lqbr.create(prob, prior_params, inv_temp, key)
        lqbr_sim = sim.compile_simulation(prob, lqbr)


        expected_ctg = ctl.lqbr.cost_to_go(prob, lqbr.params, ic)

        states, inputs, costs = jax.vmap(lambda subkey: lqbr_sim(ic, jnp.zeros((6, horizon)),
                                                                         prob.params, ctl.lqbr.LQBRParams(inv_temp, subkey, lqbr.params.prior_params)),
                                         in_axes=0, out_axes=-1)(jax.random.split(key, trials))

        self.assertLess(jnp.abs(expected_ctg - costs.sum(axis=0).mean()), 0.005)


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
