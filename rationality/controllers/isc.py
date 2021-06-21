from typing import List

import jax
import jax.numpy as jnp
import jax.random as rnd

import rationality.distributions as dst
import rationality.inference as inf
import rationality.controllers.util as util
from rationality.controllers.types import *

from functools import partial

ISTemporalInfo = Any
ISControllerState = jnp.ndarray


class ISCParams(NamedTuple):
    inv_temp: float
    init_key: jnp.ndarray


def isc(prob: Problem, inv_temp: float, num_samples: int, init_key: jnp.ndarray,
        prior_proto: dst.DistributionPrototype, prior_params: List[Tuple]) -> Controller:
    num_prior_params = len(prior_params[0])
    prior_params_for_scanning = tuple(jnp.stack([p[i] for p in prior_params]) for i in range(num_prior_params))
    params = ISCParams(inv_temp, init_key)
    cost_of_ctl_seq = util.compile_cost_of_control_sequence(prob)

    init_isc = jax.jit(lambda prob_params, isc_params: init_isc_prototype(prob_params, isc_params,
                                                                          prior_params_for_scanning))
    isc_controller = jax.jit(lambda state, t, controller_state, temporal_info, params:
                             isc_prototype(state, t, controller_state, temporal_info, params,
                                           prior_proto, num_samples, prob.prototype, cost_of_ctl_seq))

    return Controller(init_isc, isc_controller, params)


def init_isc_prototype(params: ProblemParams, isc_params: ISCParams,
                       prior_params: Any) -> Tuple[ISControllerState, ISTemporalInfo]:
    return isc_params.init_key, prior_params


def hamiltonian(state: State, input_seq: Input, t: int, proto: ProblemPrototype,
                cost_of_ctl_seq: Callable[[State, int, Input], float]) -> float:
    inputs = jnp.pad(input_seq.reshape((proto.dynamics.num_inputs, -1), order='F'),
                     [(0, 0), (0, 1)])

    return cost_of_ctl_seq(state, t, inputs)


@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def isc_prototype(state: State, t: int, controller_state: ISControllerState,
                  temporal_info: Any, params: ISCParams,
                  prior_proto: dst.DistributionPrototype, num_samples: int, prob_proto: ProblemPrototype,
                  cost_of_ctl_seq: Callable[[State, int, Input], float]) -> Tuple[Input, ISControllerState]:
    key, subkey = rnd.split(controller_state, 2)
    input_samples = prior_proto.sample(num_samples, subkey, temporal_info)

    log_prob = jax.jit(lambda u: prior_proto.log_prob(u, temporal_info) - params.inv_temp * hamiltonian(state, u, t,
                                                                                                        prob_proto,
                                                                                                        cost_of_ctl_seq))

    @jax.jit
    def zero_temp_case(input_sequences: jnp.ndarray) -> jnp.ndarray:
        costs = jax.vmap(lambda u: hamiltonian(state, u, t, prob_proto, cost_of_ctl_seq), in_axes=1)(input_sequences)

        return jnp.take(input_sequences, jnp.argmin(costs), axis=1)

    input = jax.lax.cond(jnp.isinf(params.inv_temp),
                         zero_temp_case,
                         lambda input_sequences: inf.sir(log_prob, input_sequences, subkey).flatten(),
                         input_samples)

    return input[:prob_proto.dynamics.num_inputs], key
