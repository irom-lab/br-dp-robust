import jax
import jax.numpy as jnp

from rationality.controllers.types import *

LQRTemporalInfo = jnp.ndarray
LQRControllerState = None
LQRParams = None


def create(prob: Problem) -> Controller:
    return Controller(jax.jit(lambda prob_params, params: lqr_init_prototype(prob_params, params,
                                                                             prob.prototype.horizon)),
                      lqr_prototype, None)


@jax.jit
def lqr_scanner(P: jnp.ndarray, _, A: jnp.ndarray, B: jnp.ndarray,
                Q: jnp.ndarray, R: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    K = jnp.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    P = Q + A.T @ P @ A - A.T @ P @ B @ K

    return P, K


def lqr_dynamic_programming(prob_params: ProblemParams, horizon: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    dynamics, objective = prob_params
    A, B = dynamics
    Q, R, Qf = objective
    init = Qf

    P, K = jax.lax.scan(lambda c, _: lqr_scanner(c, _, A, B, Q, R), init, None, length=horizon, reverse=True)

    return K, P


def lqr_init_prototype(prob_params: ProblemParams,
                       params: LQRParams, horizon: int) -> tuple[LQRControllerState, LQRTemporalInfo]:
    K, _ = lqr_dynamic_programming(prob_params, horizon)

    return None, K


@jax.jit
def lqr_prototype(state: State, t: int, controller_state: LQRControllerState,
                  temporal_info: LQRTemporalInfo, params: LQRParams) -> tuple[Input, LQRControllerState]:
    K = temporal_info

    return -K @ state, None


def cost_to_go(state: State, t: int, prob: Problem) -> float:
    _, P = lqr_dynamic_programming(prob.params, prob.prototype.horizon - t)

    return 0.5 * state.T @ P @ state
