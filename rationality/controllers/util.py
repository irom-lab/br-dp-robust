from functools import partial

import jax
import jax.numpy as jnp

from rationality.controllers.types import *


@partial(jax.jit, static_argnums=4)
def objective_with_temporal_overflow(state: State, input: Input, t: int,
                                     params: Any, proto: ProblemPrototype) -> float:
    time_to_end = t - proto.horizon

    branches = (lambda op: proto.trajectory_objective(*op, params),
                lambda op: proto.terminal_objective(op[0], params),
                lambda op: 0.0)

    return jax.lax.switch(time_to_end + 1, branches, (state, input, t))


@partial(jax.jit, static_argnums=3)
def cost_of_control_sequence_scanner(carry: Tuple[State, int], input: Input, params: ProblemParams,
                                     proto: ProblemPrototype) -> Tuple[Tuple[State, int], float]:
    state, t = carry
    cost = objective_with_temporal_overflow(state, input, t, params.objective, proto)
    next_state = proto.dynamics(state, input, t, params.dynamics)

    return (next_state, t + 1), cost


def cost_of_control_sequence_prototype(ic: State, it: int, inputs: Input, prob: Problem) -> float:
    init = (ic, it)
    scanner = jax.jit(lambda c, u: cost_of_control_sequence_scanner(c, u, prob.params, prob.prototype))
    final, costs = jax.lax.scan(scanner, init, inputs.T)

    return jnp.sum(costs)


def compile_cost_of_control_sequence(prob: Problem) -> Callable[[State, int, Input], float]:
    return jax.jit(lambda ic, it, inputs: cost_of_control_sequence_prototype(ic, it, inputs, prob))


def cost_of_control_sequence(ic: State, it: int, inputs: Input, prob: Problem) -> float:
    return compile_cost_of_control_sequence(prob)(ic, it, inputs)
