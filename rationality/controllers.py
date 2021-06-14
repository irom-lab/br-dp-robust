from functools import partial
from typing import Any, Callable, Tuple, NamedTuple

import jax
import jax.numpy as jnp

import rationality.dynamics as dyn
import rationality.objectives as obj
from rationality.types import State, Input


class ControlProblem(NamedTuple):
    dynamics: dyn.Dynamics
    objective: obj.Objective
    horizon: int

ControllerState = Any
ControllerTemporalInfo = Any

ControllerPrototype = Callable[[State, int, ControllerState, ControllerTemporalInfo, Any],
                               Tuple[Input, ControllerState]]


ControllerInitPrototype = Callable[[ControlProblem, Any], ControllerTemporalInfo]


class Controller(NamedTuple):
    prob: ControlProblem

    init_prototype: ControllerInitPrototype
    controller_prototype: ControllerPrototype

    params: Any

    def init(self) -> Tuple[ControllerState, ControllerTemporalInfo]:
        return self.init_prototype(self.prob, self.params)

    def __call__(self, state: State, t: int, controller_state: ControllerState,
                 temporal_info: ControllerTemporalInfo) -> Tuple[Input, ControllerState]:
        return self.controller_prototype(state, t, controller_state, temporal_info, self.params)


LQRTemporalInfo = jnp.ndarray
LQRControllerState = None
LQRParams = None


def lqr(prob: ControlProblem) -> Controller:
    return Controller(prob, lqr_init_prototype, lqr_prototype, None)


@jax.jit
def lqr_scanner(P: jnp.ndarray, _, A: jnp.ndarray, B: jnp.ndarray,
                Q: jnp.ndarray, R: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    K = -jnp.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A
    P = Q + A.T @ P @ A + A.T @ P @ B @ K

    return P, K


def lqr_dynamic_programming(prob: ControlProblem) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dynamics, objective, horizon = prob
    A, B = dynamics.params
    Q, R, Qf = objective.params

    init = Qf

    P, K = jax.lax.scan(lambda c, _: lqr_scanner(c, _, A, B, Q, R), init, None, length=horizon, reverse=True)

    return K, P


def lqr_init_prototype(prob: ControlProblem,  params: LQRParams) -> Tuple[LQRControllerState, LQRTemporalInfo]:
    K, _ = lqr_dynamic_programming(prob)

    return None, K


@jax.jit
def lqr_prototype(state: State, t: int, controller_state: LQRControllerState,
           temporal_info: LQRTemporalInfo, params: LQRParams) -> Tuple[Input, LQRControllerState]:
    K = temporal_info

    return K @ state, None


def lqr_cost_to_go(state: State, t: int, prob: ControlProblem) -> float:
    dynamics, objective, horizon = prob
    _, P = lqr_dynamic_programming(ControlProblem(dynamics, objective, horizon - t))

    return state.T @ P @ state


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
