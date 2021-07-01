from typing import Callable, Any

import jax
import jax.numpy as jnp

from rationality import controllers as ctl
from rationality.types import State, Input

Trajectory = tuple[State, Input, jnp.ndarray]
Simulation = Callable[[State, State, ctl.ProblemParams, Any], Trajectory]


def simulation_scanner_prototype(carry: tuple[State, ctl.ControllerState],
                                 slice: tuple[int, ctl.ControllerTemporalInfo],
                                 prob_params: ctl.ProblemParams,
                                 controller_params: Any,
                                 prob_proto: ctl.ProblemPrototype,
                                 controller_prototype: ctl.ControllerPrototype) -> tuple[tuple[State, Input],
                                                                                         tuple[State, Input, float]]:
    state, controller_state = carry
    t, est_noise, controller_temporal_info = slice

    input, controller_state = controller_prototype(state + est_noise, t, controller_state,
                                                   controller_temporal_info, controller_params)

    cost = prob_proto.trajectory_objective(state, input, t, prob_params.objective)

    return (prob_proto.dynamics(state, input, t, prob_params.dynamics), controller_state), (state, input, cost)


def compile_simulation(prob: ctl.Problem, controller: ctl.Controller) -> Simulation:
    _, _, terminal_objective_prototype, horizon = prob.prototype
    init_controller_prototype, controller_prototype, _ = controller

    @jax.jit
    def simulation_scanner(carry: tuple[State, ctl.ControllerState],
                           slice: tuple[int, State, ctl.ControllerTemporalInfo],
                           prob_params: ctl.ProblemParams, controller_params: Any):
        return simulation_scanner_prototype(carry, slice, prob_params, controller_params,
                                            prob.prototype, controller_prototype)

    def simulate(ic: State, est_noise: State, prob_params: ctl.ProblemParams, controller_params: Any) -> Trajectory:
        init_controller_state, controller_temporal_info = init_controller_prototype(prob_params, controller_params)

        init_carry = (ic, init_controller_state)
        final_carry, traj = jax.lax.scan(lambda carry, slice: simulation_scanner(carry, slice, prob_params,
                                                                                 controller_params),
                                         init_carry, (jnp.arange(horizon), est_noise.T, controller_temporal_info),
                                         length=horizon)

        final_state, _ = final_carry
        states, inputs, costs = traj
        terminal_cost = terminal_objective_prototype(final_state, prob_params.objective)

        return jnp.append(states, final_state.reshape(1, -1), axis=0).T, inputs.T, \
               jnp.append(costs, terminal_cost)

    return jax.jit(simulate)


def run(ic: State, est_noise: State, sim: Simulation,
        prob: ctl.Problem, controller: ctl.Controller) -> Trajectory:
    return sim(ic, est_noise, prob.params, controller.params)
