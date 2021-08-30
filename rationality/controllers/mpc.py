from functools import partial
from typing import Union, Optional

import jax
import jax.numpy as jnp
from jax.experimental.optimizers import Optimizer, OptimizerState

import rationality.controllers.util as util
import rationality.distributions as dst
import rationality.inference as inf
from rationality.controllers.types import *

MPCTemporalInfo = None
MPCState = None


class MPCParams(NamedTuple):
    pass


def create(prob: Problem, opt: Optimizer, opt_iters: int, initial_inputs: Optional[jnp.ndarray] = None) -> Controller:
    if initial_inputs is None:
        initial_inputs = jnp.zeros(prob.prototype.horizon * prob.prototype.dynamics.num_inputs)
    else:
        initial_inputs = initial_inputs.flatten(order='F')

    cost_of_ctl_seq = util.compile_cost_of_control_sequence(prob)
    init_mpc = jax.jit(lambda prob_params, mpc_params, key: init_svmpc_prototype(prob_params, mpc_params, key))

    mpc = jax.jit(lambda state, t, controller_state, temporal_info, params:
                    mpc_prototype(state, t, controller_state, temporal_info, params,
                                  prob.prototype, cost_of_ctl_seq, opt, opt_iters, initial_inputs))

    return Controller(init_mpc, mpc, MPCParams())


def init_svmpc_prototype(params: ProblemParams, svmpc_params: MPCParams,
                         key: jnp.ndarray) -> tuple[MPCState, MPCTemporalInfo]:
    return None, None


@partial(jax.jit, static_argnums=(3, 4))
def objective(x: State, t: int, flattened_inputs: Input,
              cost_of_ctl_seq: Callable[[State, int, Input], float], prob_proto: ProblemPrototype) -> float:
    return util.hamiltonian(x, flattened_inputs, t, prob_proto, cost_of_ctl_seq)


@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def mpc_prototype(state: State,
                  t: int,
                  controller_state: MPCState,
                  temporal_info: Any,
                  params: MPCParams,
                  prob_proto: ProblemPrototype,
                  cost_of_ctl_seq: Callable[[State, int, Input], float],
                  opt: Optimizer,
                  opt_iters: int,
                  initial_inputs: jnp.ndarray) -> tuple[Input, MPCState]:
    opt_init, opt_update, get_params = opt
    obj = lambda u: objective(state, t, u, cost_of_ctl_seq, prob_proto)
    obj_grad = jax.jit(jax.grad(obj))

    @jax.jit
    def step_scanner(opt_state: OptimizerState, step_iter: int) -> tuple[OptimizerState, tuple[float, jnp.ndarray]]:
        flattened_inputs = get_params(opt_state)
        value = obj(flattened_inputs)
        grad = obj_grad(flattened_inputs)

        return opt_update(step_iter, grad, opt_state), (value, flattened_inputs)

    init_opt_state = opt_init(initial_inputs)

    _, opt_traj = jax.lax.scan(step_scanner, init_opt_state, jnp.arange(opt_iters))
    values, flattened_inputs = opt_traj

    best_idx = values.argmin()
    best_flattened_input = jnp.take(flattened_inputs, best_idx, axis=0)

    return best_flattened_input[:prob_proto.dynamics.num_inputs], None
