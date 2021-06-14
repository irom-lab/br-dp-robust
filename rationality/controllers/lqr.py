import jax
import jax.numpy as jnp

from rationality.controllers.types import *

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


def lqr_init_prototype(prob: ControlProblem, params: LQRParams) -> Tuple[LQRControllerState, LQRTemporalInfo]:
    K, _ = lqr_dynamic_programming(prob)

    return None, K


@jax.jit
def lqr_prototype(state: State, t: int, controller_state: LQRControllerState,
                  temporal_info: LQRTemporalInfo, params: LQRParams) -> Tuple[Input, LQRControllerState]:
    K = temporal_info

    return K @ state, None


def cost_to_go(state: State, t: int, prob: ControlProblem) -> float:
    dynamics, objective, horizon = prob
    _, P = lqr_dynamic_programming(ControlProblem(dynamics, objective, horizon - t))

    return state.T @ P @ state
