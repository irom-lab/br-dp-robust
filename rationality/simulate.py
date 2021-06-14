from typing import Tuple, Callable, Any

import jax
import jax.numpy as jnp

from rationality import controllers as ctl, dynamics as dyn, objectives as obj
from rationality.types import State, Input

Trajectory = Tuple[State, Input, jnp.ndarray]
Simulation = Callable[[State, State, Any, Any, Any], Trajectory]


def simulation_scanner_prototype(carry: Tuple[State, ctl.ControllerState],
                                 slice: Tuple[int, ctl.ControllerTemporalInfo],
                                 dynamics_params: Any,
                                 objective_params: Any,
                                 controller_params: Any,
                                 dynamics_prototype: dyn.DynamicsPrototype,
                                 trajectory_objective_prototype: obj.TrajectoryObjectivePrototype,
                                 controller_prototype: ctl.ControllerPrototype) -> Tuple[Tuple[State, Input],
                                                                                         Tuple[State, Input, float]]:
    state, controller_state = carry
    t, est_noise, controller_temporal_info = slice

    input, controller_state = controller_prototype(state + est_noise, t, controller_state,
                                                   controller_temporal_info, controller_params)

    cost = trajectory_objective_prototype(state, input, t, objective_params)

    return (dynamics_prototype(state, input, t, dynamics_params), controller_state), (state, input, cost)


def compile_simulation(prob: ctl.ControlProblem, controller: ctl.Controller) -> Simulation:
    dynamics_prototype = prob.dynamics.prototype
    trajectory_objective_prototype = prob.objective.trajectory_prototype
    terminal_objective_prototype = prob.objective.terminal_prototype
    controller_prototype = controller.controller_prototype
    init_controller_prototype = controller.init_prototype

    @jax.jit
    def simulation_scanner(carry, slice, dynamics_params, objective_params, controller_params):
        return simulation_scanner_prototype(carry, slice, dynamics_params, objective_params, controller_params,
                                            dynamics_prototype, trajectory_objective_prototype, controller_prototype)

    def simulate(ic: State, est_noise: State, dynamics_params: Any,
                 objective_params: Any, controller_params: Any) -> Trajectory:
        init_controller_state, controller_temporal_info = init_controller_prototype(prob, controller_params)

        init_carry = (ic, init_controller_state)
        final_carry, traj = jax.lax.scan(lambda carry, slice: simulation_scanner(carry, slice, dynamics_params,
                                                                                 objective_params, controller_params),
                                         init_carry, (jnp.arange(prob.horizon), est_noise.T, controller_temporal_info),
                                         length=prob.horizon)

        final_state, _ = final_carry
        states, inputs, costs = traj
        terminal_cost = terminal_objective_prototype(final_state, objective_params)

        return jnp.append(states, final_state.reshape(1, -1), axis=0).T, inputs.T, \
               jnp.append(costs, terminal_cost)

    return jax.jit(simulate)


def run(ic: State, est_noise: State, sim: Simulation,
        prob: ctl.ControlProblem, controller: ctl.Controller) -> Trajectory:
    return sim(ic, est_noise, prob.dynamics.params, prob.objective.params, controller.params)


# def simulate(ic: State, prob: ctl.ControlProblem, controller: ctl.Controller,
#              est_noise: State) -> Trajectory:
#     init_controller_state, controller_temporal_info = controller.init()
#
#     init_carry = (ic, init_controller_state)
#     final_carry, traj = jax.lax.scan(lambda carry, slice: simulate_scanner(carry, slice, controller, prob),
#                                      init_carry, (jnp.arange(prob.horizon), est_noise.T, controller_temporal_info),
#                                      length=prob.horizon)
#
#     final_state, _ = final_carry
#     states, inputs, costs = traj
#     terminal_cost = prob.objective(final_state)
#
#     return jnp.append(states, final_state.reshape(1, -1), axis=0).T, inputs.T, \
#            jnp.append(costs, terminal_cost)
#
#
#
#
# def simulate_scanner(carry: Tuple[State, ctl.ControllerState],
#                      slice: Tuple[int, ctl.ControllerTemporalInfo],
#                      controller: ctl.Controller,
#                      prob: ctl.ControlProblem) -> Tuple[Tuple[State, Input], Tuple[State, Input, float]]:
#     state, controller_state = carry
#     t, est_noise, controller_temporal_info = slice
#     input, controller_state = controller(state + est_noise, t, controller_state, controller_temporal_info)
#     cost = prob.objective(state, input, t)
#
#     return (prob.dynamics(state, input, t), controller_state), (state, input, cost)
