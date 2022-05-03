from operator import inv
import jax
import jax.numpy as jnp
import jax.random as rnd

import rationality.distributions as dst
import rationality.inference as inf

from typing import NamedTuple, Union, Iterable

from rationality.controllers.types import *
from rationality.types import StoppingCondition

MPTemporalInfo = None
MPControllerState = jnp.ndarray

class MPParams(NamedTuple):
    inv_temp: float


def create(inv_temp: float, prob: Problem, prior_proto: dst.DistributionPrototype, stop_cond: StoppingCondition,
           prior_params: dst.DistributionParams, num_samples: int) -> Controller:
    dyn = jax.jit(lambda state, input, t: prob.prototype.dynamics(state, input, t, prob.params.dynamics))
    traj_cost = jax.jit(lambda state, input, t: prob.prototype.trajectory_objective(state, input, t, prob.params.objective))
    term_cost = jax.jit(lambda state: prob.prototype.terminal_objective(state, prob.params.objective))

    rollout = jax.jit(lambda ic, inputs, it: _rollout(ic, inputs, it, dyn))
    process_traj = jax.jit(lambda states, inputs, it: _process_traj(states, inputs, it, prob.prototype.horizon,
                                                                    traj_cost, term_cost, stop_cond))
    
    mp_prototype = jax.jit(lambda state, t, controller_state, temporal_info: _mp_prototype_finite_inv_temp(state, t, controller_state,
                                                                                                           temporal_info, num_samples,
                                                                                                           rollout, process_traj))

    return Controller(_init_mp_prototype, mp_prototype, MPParams(inv_temp))


def _rollout(ic: State, inputs: Input, it: int,
             dyn: Callable[[State, Input, State], State]) -> State:
    @jax.jit
    def _scanner(carry: tuple[State, int], input: Input) -> tuple[tuple[State, int], State]:
        state, t = carry
        new_state = dyn(state, input, t)

        return (new_state, t + 1), state
    
    carry, states = jax.lax.scan(_scanner, (ic, it), inputs)

    return jnp.concatenate([states, carry[0].reshape((1, -1))])           


def _process_traj(states: State, inputs: Input, it: int,
                  horizon: int,
                  traj_cost: Callable[[State, Input, int], float],
                  term_cost: Callable[[State], float],
                  stop_cond: StoppingCondition) -> tuple[float, int]:
    @jax.jit
    def _scanner(carry: tuple[int, int, bool], 
                 traj_data: tuple[tuple[State, State], Input]) -> tuple[tuple[int, int, float], tuple[float, int]]:
        t, stopping_time, already_stopped = carry
        state_pair, input = traj_data
        
        stop = stop_cond(state_pair[0], input, t, state_pair[1]) | t >= horizon
        cost = jax.lax.cond(stop | already_stopped,
                            lambda *_: 0.0,
                            lambda x, t, u: traj_cost(x, u, t),
                            (state_pair[0], t, input, state_pair[1]))
        
        stopping_time = jax.lax.cond(stop & ~already_stopped,
                                     lambda t, st: t,
                                     lambda t, st: st,
                                     (t, stopping_time))
        
        return (t + 1, stopping_time, stop | already_stopped), cost

    state_pairs = jax.vmap(lambda state, next_state: (state, next_state))(states[:-1], states[1:])
    carry, costs = jax.lax.scan(_scanner, (it, -1, False), (state_pairs, inputs))
    _, stopping_time, already_stopped = carry

    return costs.sum() + term_cost(states[stopping_time]), stopping_time


@jax.jit
def _init_mp_prototype(params: ProblemParams, mp_params: MPParams, key: jnp.ndarray) -> tuple[MPControllerState, MPTemporalInfo]:
    return key, None



def _mp_prototype_finite_inv_temp(state: State, t: int, controller_state: MPControllerState,
                                  temporal_info: MPTemporalInfo, params: MPParams,
                                  prob: Problem, rollout_horizon: int,
                                  prior_proto: dst.DistributionPrototype,
                                  num_samples: int,
                                  rollout: Callable[[State, Input, int], State],
                                  process_traj: Callable[[State, Input, int], tuple[float, int]]) -> tuple[Input, MPControllerState]:
    
    key, subkey = rnd.split(controller_state)    
    input_samples = prior_proto.sample(num_samples, subkey)
    states = jax.vmap(lambda x, u: rollout(x, u, t))(states, input_samples)
    cost, _ = jax.vmap(lambda x, u: process_traj(x, u, t))(states, input_samples)

    log_prob = jax.jit(lambda u, c: prior_proto.log_prob(u, temporal_info) - params.inv_temp * c)(cost, input_samples)

    key, subkey = rnd.split(key)
    input_traj = inf.sir(log_prob, input_samples.T, subkey)

    return input_traj.T[0], key

    




