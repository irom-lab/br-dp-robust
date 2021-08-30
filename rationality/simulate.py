from typing import Callable, Any, Iterable, NamedTuple, Optional, Any

import jax
import jax.numpy as jnp
import jax.random as rnd

import numpy as np

from rationality import controllers as ctl
from rationality.types import State, Input, Trajectory


def collect_trajectories(trajs: Iterable[Trajectory], stack=np.stack) -> Trajectory:
    return Trajectory(stack([t.states for t in trajs]),
                      stack([t.inputs for t in trajs]),
                      stack([t.costs for t in trajs]))


SimFun = Callable[[State, State, ctl.ProblemParams, Any, jnp.ndarray], Trajectory]


class Simulation(NamedTuple):
    simfun: SimFun
    problem: ctl.Problem
    controller: ctl.Controller

    def run(self, ic: State, est_noise: State, key: jnp.ndarray = rnd.PRNGKey(0)) -> Trajectory:
        return self.simfun(ic, est_noise, self.problem.params, self.controller.params, key)

    def run_with_params(self, ic: State, est_noise: State, params: Any, key: jnp.ndarray = rnd.PRNGKey(0)) -> Trajectory:
        return self.simfun(ic, est_noise, self.problem.params, params, key)

    def __call__(self, ic: State, est_noise: State, key: jnp.ndarray = rnd.PRNGKey(0),
                 with_params: Optional[Any] = None) -> Trajectory:
        if with_params is None:
            return self.run(ic, est_noise, key)
        else:
            return self.run_with_params(ic, est_noise, key, with_params)


def simulation_scanner_prototype(carry: tuple[State, ctl.ControllerState],
                                 temporal: tuple[int, State, ctl.ControllerTemporalInfo],
                                 prob_params: ctl.ProblemParams,
                                 controller_params: Any,
                                 prob_proto: ctl.ProblemPrototype,
                                 controller_prototype: ctl.ControllerPrototype) -> tuple[tuple[State, Input],
                                                                                         tuple[State, Input, float]]:
    state, controller_state = carry
    t, est_noise, controller_temporal_info = temporal

    input, controller_state = controller_prototype(state + est_noise, t, controller_state,
                                                   controller_temporal_info, controller_params)

    cost = prob_proto.trajectory_objective(state, input, t, prob_params.objective)

    return (prob_proto.dynamics(state, input, t, prob_params.dynamics), controller_state), (state, input, cost)


def compile_simulation(prob: ctl.Problem, controller: ctl.Controller) -> Simulation:
    _, _, terminal_objective_prototype, horizon = prob.prototype
    init_controller_prototype, controller_prototype, _ = controller

    @jax.jit
    def simulation_scanner(carry: tuple[State, ctl.ControllerState],
                           temporal: tuple[int, State, ctl.ControllerTemporalInfo],
                           prob_params: ctl.ProblemParams, controller_params: Any):
        return simulation_scanner_prototype(carry, temporal, prob_params, controller_params,
                                            prob.prototype, controller_prototype)

    def simulate(ic: State, est_noise: State,
                 prob_params: ctl.ProblemParams, controller_params: Any,
                 key: jnp.ndarray = rnd.PRNGKey(0)) -> Trajectory:
        init_controller_state, controller_temporal_info = init_controller_prototype(prob_params, controller_params, key)

        init_carry = (ic, init_controller_state)
        final_carry, traj = jax.lax.scan(lambda carry, temporal: simulation_scanner(carry, temporal, prob_params,
                                                                                    controller_params),
                                         init_carry, (jnp.arange(horizon), est_noise.T, controller_temporal_info),
                                         length=horizon)

        final_state, _ = final_carry
        states, inputs, costs = traj
        terminal_cost = terminal_objective_prototype(final_state, prob_params.objective)

        return Trajectory(jnp.append(states, final_state.reshape(1, -1), axis=0).T,
                          inputs.T, jnp.append(costs, terminal_cost))

    return Simulation(jax.jit(simulate), prob, controller)
