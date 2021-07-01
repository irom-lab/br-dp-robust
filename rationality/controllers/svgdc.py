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

SVGDCTemporalInfo = Any
SVGDCState = jnp.ndarray


class SVGDCParams(NamedTuple):
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
    params = SVGDCParams(inv_temp, init_key, bandwidth)
    cost_of_ctl_seq = util.compile_cost_of_control_sequence(prob)

    init_svgdc = jax.jit(lambda prob_params, svgd_params: init_svgdc_prototype(prob_params, svgd_params,
                                                                               prior_params_for_scanning))

    if bandwidth == 'dynamic':
        kernel = jax.jit(lambda x, y, s: inf.rbf_dyn_bw_kernel(x, y, s, num_samples))
        bandwidth = jnp.nan
    else:
        kernel = jax.jit(lambda x, y, s: inf.rbf_kernel(x, y, s, bandwidth))

    svgdc = jax.jit(lambda state, t, controller_state, temporal_info, params:
                    svgdc_prototype(state, t, controller_state, temporal_info, params,
                                    prior_proto, num_samples, prob.prototype, cost_of_ctl_seq,
                                    kernel, opt, opt_iters, sir_at_end))

    return Controller(init_svgdc, svgdc, params)


def init_svgdc_prototype(params: ProblemParams, svgdc_params: SVGDCParams,
                         prior_params: Any) -> tuple[SVGDCState, SVGDCTemporalInfo]:
    return svgdc_params.init_key, prior_params


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def svgdc_sample_finite(prior_input_samples: jnp.ndarray, key: Optional[jnp.ndarray],
                        log_prob_finite: Callable[[jnp.ndarray], float], kernel: inf.Kernel, opt: Optimizer,
                        opt_iters: int, sir_at_end: bool) -> jnp.ndarray:
    if sir_at_end:
        samples = inf.sgvd(log_prob_finite, kernel, opt, prior_input_samples, opt_iters)
        return inf.sir(log_prob_finite, samples, key)
    else:
        return inf.sgvd(log_prob_finite, kernel, opt, prior_input_samples, opt_iters)[:, 0]


@partial(jax.jit, static_argnums=(5, 6, 7, 8, 9, 10, 11, 12))
def svgdc_prototype(state: State, t: int, controller_state: SVGDCState,
                    temporal_info: Any, params: SVGDCParams,
                    prior_proto: dst.DistributionPrototype, num_samples: int, prob_proto: ProblemPrototype,
                    cost_of_ctl_seq: Callable[[State, int, Input], float],
                    kernel: inf.Kernel,
                    opt: Optimizer,
                    opt_iters: int,
                    sir_at_end: bool = False) -> tuple[Input, SVGDCState]:
    key, subkey1, subkey2 = rnd.split(controller_state, 3)
    prior_input_samples = prior_proto.sample(num_samples, subkey1, temporal_info)

    log_prob_finite = jax.jit(lambda u: prior_proto.log_prob(u, temporal_info)
                                        - params.inv_temp * util.hamiltonian(state, u, t, prob_proto, cost_of_ctl_seq))
    sample_finite = jax.jit(lambda u: svgdc_sample_finite(u, subkey2, log_prob_finite,
                                                          kernel, opt, opt_iters, sir_at_end))

    log_prob_inf = jax.jit(lambda u: -util.hamiltonian(state, u, t, prob_proto, cost_of_ctl_seq))
    sample_inf = jax.jit(lambda u: inf.sgvd(log_prob_inf, dummy_kernel, opt, u[:, 0].reshape((-1, 1)), opt_iters)[:, 0])

    posterior_input_sample = jax.lax.cond(jnp.isinf(params.inv_temp),
                                          sample_inf,
                                          sample_finite,
                                          prior_input_samples)

    return posterior_input_sample[:prob_proto.dynamics.num_inputs], key
