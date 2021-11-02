"""
Importance Sampling Controller (ISC).
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as rnd

import rationality.controllers.util as util
import rationality.distributions as dst
import rationality.inference as inf
from rationality.controllers.types import *

from typing import Iterable, Union

ISTemporalInfo = Any
ISControllerState = jnp.ndarray


class ISCParams(NamedTuple):
    """
    Parameters for the ISC

    inv_temp: The inverse temperature (beta) value of the controller.
    """
    inv_temp: float


def create(prob: Problem, inv_temp: float, num_samples: int,
           prior_proto: dst.DistributionPrototype,
           prior_params: Union[Iterable[dst.DistributionParams], dst.DistributionParams]) -> Controller:
    """
    Creates a Controller instance using the ISC prototype functions and parameters.

    :param prob: The control problem instance.
    :param inv_temp: The inverse temperature value (beta) to use for the controller.
    :param num_samples: The number of samples used in the importance sampling procedure.
    :param prior_proto: The prototype describing the prior distribution.
    :param prior_params: The sequence of parameters describing the prior distribution at each time step. These may
                         either be specified as a list of parameter object instances or a single instance where
                         the first axis of each parameter is the time index.

    :returns: The controller instance.
    """
    if not dst.isdistparams(prior_params):
        num_prior_params = len(prior_params[0])
        prior_params = tuple(jnp.stack([p[i] for p in prior_params]) for i in range(num_prior_params))

    params = ISCParams(inv_temp)
    cost_of_ctl_seq = util.compile_cost_of_control_sequence(prob)

    init_isc = jax.jit(lambda prob_params, isc_params, key: init_isc_prototype(prob_params, isc_params,
                                                                               key, prior_params))
    isc_controller = jax.jit(lambda state, t, controller_state, temporal_info, params:
                             isc_prototype(state, t, controller_state, temporal_info, params,
                                           prior_proto, num_samples, prob.prototype, cost_of_ctl_seq))

    return Controller(init_isc, isc_controller, params)


def init_isc_prototype(params: ProblemParams, isc_params: ISCParams, key: jnp.ndarray,
                       prior_params: Any) -> tuple[ISControllerState, ISTemporalInfo]:
    """
    Initialize the ISC by returning the controller state and temporal information.

    :param params: Unused
    :param isc_params: Unused
    :param key: The RNG key that will be used for the simulation.
    :param prior_params: The parameters describing the prior distribution. Each attribute of the object must be at least
                         2-dimensional with the first axis indexing the temporal sequence of parameters.

    :returns: The tuple of state and temporal information for the ISC.
    """
    return key, prior_params


@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def isc_prototype(state: State, t: int, controller_state: ISControllerState,
                  temporal_info: ISTemporalInfo, params: ISCParams,
                  prior_proto: dst.DistributionPrototype, num_samples: int, prob_proto: ProblemPrototype,
                  cost_of_ctl_seq: Callable[[State, int, Input], float]) -> tuple[Input, ISControllerState]:
    """
    The prototype for the ISC controller.

    :param state: The current system state.
    :param t: The current time step.
    :param controller_state: The current state of the controller.
    :param temporal_info: The state-independent temporal information provided to the controller at the current time.
    :param num_samples: The number of samples used for importance sampling.
    :param prob_proto: The prototype of the control problem.
    :param cost_of_ctl_seq: A function that maps (x, t, u) to a cost, where x is the current state (n-dimensional
                            vector), t is the current time step, and u (m-by-t_f array where t_f is the problem
                            horizon), is the sequence of control inputs.

    :returns: A tuple (u, isc_state) containing the control input and the ISC State respectively.
    """
    key, subkey = rnd.split(controller_state, 2)
    input_samples = prior_proto.sample(num_samples, subkey, temporal_info)

    log_prob = jax.jit(lambda u: prior_proto.log_prob(u, temporal_info)
                                 - params.inv_temp * util.hamiltonian_prototype(state, u, t, prob_proto, cost_of_ctl_seq))

    @jax.jit
    def zero_temp_case(input_sequences: jnp.ndarray) -> jnp.ndarray:
        """
        This function handles the limiting case where inv_temp is infinite. In this case, we simply take the best
        performing input sequence.
        """
        costs = jax.vmap(lambda u: util.hamiltonian_prototype(state, u, t, prob_proto, cost_of_ctl_seq), in_axes=1)(input_sequences)

        return jnp.take(input_sequences, jnp.argmin(costs), axis=1)

    input = jax.lax.cond(jnp.isinf(params.inv_temp),
                         zero_temp_case,
                         lambda input_sequences: inf.sir(log_prob, input_sequences, subkey).flatten(),
                         input_samples)

    return input[:prob_proto.dynamics.num_inputs], key
