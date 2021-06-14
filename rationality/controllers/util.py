import jax

from rationality.controllers.types import *


def objective_with_temporal_overflow(state: State, input: Input, t: int,
                                     horizon: int, objective: obj.Objective) -> float:
    time_to_end = t - horizon

    branches = (lambda op: objective(*op),
                lambda op: objective(op[0]),
                lambda op: 0.0)

    return jax.lax.switch(time_to_end + 1, branches, (state, input, t))


def cost_of_control_sequence_scanner(carry: Tuple[State, int], input: Input,
                                     prob: ControlProblem) -> Tuple[Tuple[State, int], float]:
    state, t = carry
    dynamics, objective, _ = prob
    cost = objective_with_temporal_overflow(state, input, t, prob.horizon, prob.objective)
    next_state = dynamics(state, input, t)

    return (next_state, t + 1), cost


def cost_of_control_sequence(ic: State, it: int, inputs: Input, prob: ControlProblem) -> float:
    init = (ic, it)

    final, costs = jax.lax.scan(lambda c, u: cost_of_control_sequence_scanner(c, u, prob), init, inputs.T)

    return costs.sum()
