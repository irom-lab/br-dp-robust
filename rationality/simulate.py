from typing import Callable, Any, Iterable, NamedTuple, Optional, Any

import jax
import jax.numpy as jnp
import jax.random as rnd

import numpy as np

from rationality import controllers as ctl
from rationality.types import State, Input, StoppingCondition, Trajectory


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


def _simulation_scanner_prototype(carry: tuple[State, ctl.ControllerState, bool, int, Optional[State]],
                                  temporal: tuple[int, State, ctl.ControllerTemporalInfo],
                                  prob_params: ctl.ProblemParams,
                                  controller_params: Any,
                                  prob_proto: ctl.ProblemPrototype,
                                  controller_prototype: ctl.ControllerPrototype,
                                  stop_cond: StoppingCondition) -> tuple[tuple[State, Input, bool, int, State],
                                                                         tuple[State, Input, float]]:
    state, controller_state, has_stopped, stopping_time, stopping_state = carry
    t, est_noise, controller_temporal_info = temporal

    input, controller_state = controller_prototype(state + est_noise, t, controller_state,
                                                   controller_temporal_info, controller_params)

    cost = prob_proto.trajectory_objective(state, input, t, prob_params.objective)
    next_state = prob_proto.dynamics(state, input, t, prob_params.dynamics)
    stop = stop_cond(state, input, t, next_state)
    
    stopping_state = jax.lax.cond(stop & ~has_stopped,
                                  lambda s: s[0],
                                  lambda s: s[1],
                                  (state, stopping_state))
    
    has_stopped = stop | has_stopped

    stopping_time = jax.lax.cond(has_stopped,
                                 lambda times: jnp.minimum(*times),
                                 lambda times: times[1],
                                 (t, stopping_time))

    return (next_state, controller_state, has_stopped, stopping_time, stopping_state), (state, input, cost)


def compile_simulation(prob: ctl.Problem, controller: ctl.Controller, stop_cond: StoppingCondition = jax.jit(lambda *_: False)) -> Simulation:
    _, _, terminal_objective_prototype, horizon = prob.prototype
    init_controller_prototype, controller_prototype, _ = controller

    @jax.jit
    def simulation_scanner(carry: tuple[State, ctl.ControllerState, bool, int, Optional[State]],
                           temporal: tuple[int, State, ctl.ControllerTemporalInfo],
                           prob_params: ctl.ProblemParams, controller_params: Any):
        return _simulation_scanner_prototype(carry, temporal, prob_params, controller_params,
                                             prob.prototype, controller_prototype, stop_cond)

    def simulate(ic: State, est_noise: State,
                 prob_params: ctl.ProblemParams, controller_params: Any,
                 key: jnp.ndarray = rnd.PRNGKey(0)) -> Trajectory:
        init_controller_state, controller_temporal_info = init_controller_prototype(prob_params, controller_params, key)

        init_carry = (ic, init_controller_state, False, horizon, jnp.nan * jnp.ones_like(ic))
        final_carry, traj = jax.lax.scan(lambda carry, temporal: simulation_scanner(carry, temporal, prob_params, controller_params),
                                         init_carry, (jnp.arange(horizon), est_noise.T, controller_temporal_info),
                                         length=horizon)

        final_state, final_time, has_stopped, stopping_time, stopping_state = final_carry
        states, inputs, costs = traj
        states = jnp.append(states, final_state.reshape(1, -1), axis=0)

        terminal_state = jax.lax.cond(has_stopped,
                                      lambda s: s[0],
                                      lambda s: s[1],
                                      (stopping_state, states[-1]))

        terminal_cost = terminal_objective_prototype(terminal_state, prob_params.objective)

        return Trajectory(states.T, inputs.T, jnp.append(costs, terminal_cost), stopping_time)

    return Simulation(jax.jit(simulate), prob, controller)
