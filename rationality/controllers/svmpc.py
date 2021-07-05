from functools import partial
from typing import Union, Optional

import jax
import jax.numpy as jnp
import jax.random as rnd
from jax.experimental.optimizers import Optimizer

import rationality.controllers.util as util
import rationality.distributions as dst
import rationality.inference as inf
from rationality.controllers.types import *

SVMPCTemporalInfo = Any
SVMPCState = jnp.ndarray


class SVMPCParams(NamedTuple):
    inv_temp: float
    init_key: jnp.ndarray
    bandwidth: float


@jax.jit
def dummy_kernel(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> float:
    return 1.0


def create(prob: Problem, inv_temp: float, init_key: jnp.ndarray, bandwidth: Union[str, float], num_samples: int,
           prior_proto: dst.DistributionPrototype, prior_params: list[tuple],
           opt: Optimizer, opt_iters: int, sir_at_end: bool = False) -> Controller:
    num_prior_params = len(prior_params[0])
    prior_params_for_scanning = tuple(jnp.stack([p[i] for p in prior_params]) for i in range(num_prior_params))
    params = SVMPCParams(inv_temp, init_key, bandwidth)
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


def init_svmpc_prototype(params: ProblemParams, svmpc_params: SVMPCParams,
                         prior_params: Any) -> tuple[SVMPCState, SVMPCTemporalInfo]:
    return svmpc_params.init_key, prior_params


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def svmpc_sample_finite(prior_input_samples: jnp.ndarray, key: Optional[jnp.ndarray],
                        log_prob_finite: Callable[[jnp.ndarray], float], kernel: inf.Kernel, opt: Optimizer,
                        opt_iters: int, sir_at_end: bool) -> jnp.ndarray:
    if sir_at_end:
        posterior_input_samples = inf.sgvd(log_prob_finite, kernel, opt, prior_input_samples, opt_iters)
        return inf.sir(log_prob_finite, posterior_input_samples, key)
    else:
        return inf.sgvd(log_prob_finite, kernel, opt, prior_input_samples, opt_iters)[:, 0]


@partial(jax.jit, static_argnums=(1, 2, 3))
def svmpc_sample_infinite(prior_input_samples: jnp.ndarray, log_prob_infinite: Callable[[jnp.ndarray], float],
                     opt: Optimizer, opt_iters: int) -> jnp.ndarray:
    posterior_input_samples = inf.sgvd(log_prob_infinite, dummy_kernel, opt, prior_input_samples, opt_iters)
    log_probs = jax.vmap(log_prob_infinite, in_axes=-1)(posterior_input_samples)

    return jnp.take(posterior_input_samples, jnp.argmax(log_probs), axis=-1)


@partial(jax.jit, static_argnums=(5, 6, 7, 8, 9, 10, 11, 12))
def svmpc_prototype(state: State, t: int, controller_state: SVMPCState,
                    temporal_info: Any, params: SVMPCParams,
                    prior_proto: dst.DistributionPrototype, num_samples: int, prob_proto: ProblemPrototype,
                    cost_of_ctl_seq: Callable[[State, int, Input], float],
                    kernel: inf.Kernel,
                    opt: Optimizer,
                    opt_iters: int,
                    sir_at_end: bool = False) -> tuple[Input, SVMPCState]:
    key, subkey1, subkey2 = rnd.split(controller_state, 3)
    prior_input_samples = prior_proto.sample(num_samples, subkey1, temporal_info)

    log_prob_finite = jax.jit(lambda u: prior_proto.log_prob(u, temporal_info)
                                        - params.inv_temp * util.hamiltonian(state, u, t, prob_proto, cost_of_ctl_seq))
    sample_finite = jax.jit(lambda u: svmpc_sample_finite(u, subkey2, log_prob_finite,
                                                          kernel, opt, opt_iters, sir_at_end))

    log_prob_infinite = jax.jit(lambda u: -util.hamiltonian(state, u, t, prob_proto, cost_of_ctl_seq))
    sample_infinite = jax.jit(lambda u: svmpc_sample_infinite(u, log_prob_infinite, opt, opt_iters))

    posterior_input_sample = jax.lax.cond(jnp.isinf(params.inv_temp),
                                          sample_infinite,
                                          sample_finite,
                                          prior_input_samples)

    return posterior_input_sample[:prob_proto.dynamics.num_inputs], key
