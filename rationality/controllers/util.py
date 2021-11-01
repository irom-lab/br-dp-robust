from functools import partial

import jax
import jax.numpy as jnp

from rationality.types import ObjectiveParams, DynamicsParams
from rationality.controllers.types import *


@partial(jax.jit, static_argnums=4)
def objective_with_temporal_overflow(state: State, input: Input, t: int,
                                     params: ObjectiveParams, proto: ProblemPrototype) -> float:
    """
    Compute the MPC objective at time t with clipping for t greater than the problem horizon.

    :param state: The current system state (n-dimensional vector).
    :param input: The current system input (m-dimensional vector).
    :param t: The current time (positive integer).
    :param params: The objective parameters for the problem of interest.
    :param params: The problem prototype.

    :returns: The cost at the current time step or 0.0 if t is larger than the problem horizon.
    """
    time_to_end = t - proto.horizon

    branches = (lambda op: proto.trajectory_objective(*op, params),
                lambda op: proto.terminal_objective(op[0], params),
                lambda op: 0.0)

    return jax.lax.switch(time_to_end + 1, branches, (state, input, t))


@partial(jax.jit, static_argnums=3)
def _cost_of_control_sequence_scanner(carry: tuple[State, int], input: Input, params: ProblemParams,
                                      proto: ProblemPrototype) -> tuple[tuple[State, int], float]:
    state, t = carry
    cost = objective_with_temporal_overflow(state, input, t, params.objective, proto)
    next_state = proto.dynamics(state, input, t, params.dynamics)

    return (next_state, t + 1), cost


def cost_of_control_sequence_prototype(ic: State, it: int, inputs: Input, prob: Problem) -> float:
    """
    Prototype of the cost-of-control-sequence function that maps a sequence of control inputs applied from an initial
    time onward. Costs incurred after the end of the problem horizon are not included.

    :param ic: The initial state of the system.
    :param it: The initial time of the system (non-negative integer).
    :param inputs: An array whose columns represent m-dimensional control inputs to be applied in order.
    :param prob: The control problem instance.

    :returns: The cost of applying the input sequence for the steps starting from the initial time `it`.
    """
    init = (ic, it)
    scanner = jax.jit(lambda c, u: _cost_of_control_sequence_scanner(c, u, prob.params, prob.prototype))
    final, costs = jax.lax.scan(scanner, init, inputs.T)

    return jnp.sum(costs)


def compile_cost_of_control_sequence(prob: Problem) -> Callable[[State, int, Input], float]:
    return jax.jit(lambda ic, it, inputs: cost_of_control_sequence_prototype(ic, it, inputs, prob))


def cost_of_control_sequence(ic: State, it: int, inputs: Input, prob: Problem) -> float:
    return compile_cost_of_control_sequence(prob)(ic, it, inputs)


def hamiltonian(state: State, input_seq: Input, t: int, proto: ProblemPrototype,
                cost_of_ctl_seq: Callable[[State, int, Input], float]) -> float:
    inputs = jnp.pad(input_seq.reshape((proto.dynamics.num_inputs, -1), order='F'),
                     [(0, 0), (0, 1)])

    return cost_of_ctl_seq(state, t, inputs)


@partial(jax.jit, static_argnums=1)
def flat_inputs_to_sequence(inputs: Input, num_inputs: int) -> Input:
    return inputs.reshape((num_inputs, -1), order='F')