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


def create(prob: Problem, inv_temp: float, init_key: jnp.ndarray, bandwidth: Union[str, float], num_samples: int,
           prior_proto: dst.DistributionPrototype, prior_params: list[tuple],
           opt: Optimizer, opt_iters: int, sir_at_end: bool = False) -> Controller:
    num_prior_params = len(prior_params[0])
    prior_params_for_scanning = tuple(jnp.stack([p[i] for p in prior_params]) for i in range(num_prior_params))
    params = MPCParams(inv_temp, init_key, bandwidth)
    cost_of_ctl_seq = util.compile_cost_of_control_sequence(prob)

    init_svmpc = jax.jit(lambda prob_params, svmp_params: init_svmpc_prototype(prob_params, svmp_params,
                                                                               prior_params_for_scanning))

    if bandwidth == 'dynamic':
        kernel = jax.jit(lambda x, y, s: inf.rbf_dyn_bw_kernel(x, y, s, num_samples))
        bandwidth = jnp.nan
    else:
        kernel = jax.jit(lambda x, y, s: inf.rbf_kernel(x, y, s, bandwidth))

    svmpc = jax.jit(lambda state, t, controller_state, temporal_info, params:
                    svmpc_prototype(state, t, controller_state, temporal_info, params,
                                    prior_proto, num_samples, prob.prototype, cost_of_ctl_seq,
                                    kernel, opt, opt_iters, sir_at_end))

    return Controller(init_svmpc, svmpc, params)


def init_svmpc_prototype(params: ProblemParams, svmpc_params: MPCParams,
                         prior_params: Any) -> tuple[MPCState, MPCTemporalInfo]:
    return None, None


@partial(jax.jit, static_argnums=(3, 4))
def objective(x: State, t: int, flattened_inputs: Input,
              cost_of_ctl_seq: Callable[[State, int, Input], float], prob_proto: ProblemPrototype) -> float:
    return util.hamiltonian(x, flattened_inputs, t, prob_proto, cost_of_ctl_seq)


@partial(jax.jit, static_argnums=(5, 6, 7, 8, 9, 10, 11, 12))
def mpc_prototype(state: State, t: int, controller_state: MPCState,
                  temporal_info: Any, params: MPCParams,
                  prob_proto: ProblemPrototype,
                  cost_of_ctl_seq: Callable[[State, int, Input], float],
                  opt: Optimizer,
                  opt_iters: int) -> tuple[Input, MPCState]:
    opt_init, opt_update, get_params = opt
    initial_inputs = jnp.zeros(prob_proto.dynamics.num_inputs * prob_proto.horizon)
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
    best_flattened_input = jnp.take(best_idx, axis=0)

    return best_flattened_input[:prob_proto.dynamics.num_inputs], None
