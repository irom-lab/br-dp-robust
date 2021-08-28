import jax
import jax.numpy as jnp

import rationality.distributions as dst

from typing import Optional

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
    K = -jnp.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    P = Q + A.T @ P @ A + A.T @ P @ B @ K

    return P, K


def lqr_dynamic_programming(prob_params: ProblemParams, horizon: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    dynamics, objective = prob_params
    A, B = dynamics
    Q, R, Qf, _, _ = objective
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

    return K @ state, None


def rollout_scanner(carry: tuple[jnp.ndarray, jnp.ndarray],
                    temporal_info: tuple[jnp.ndarray, jnp.ndarray], prob: Problem) -> tuple[tuple[jnp.ndarray, jnp.ndarray],
                                                                                            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    Q, R, Qf, _, _ = prob.params.objective
    A, B = prob.params.dynamics

    x_bar, Sigma_x = carry
    Sigma_noise, K = temporal_info

    Sigma_u = K @ (Sigma_x + Sigma_noise) @ K.T
    u_bar = K @ x_bar

    cost = 0.5 * (x_bar.T @ Q @ x_bar + u_bar.T @ R @ u_bar + jnp.trace(Q @ Sigma_x) + jnp.trace(R @ Sigma_u))

    new_state_mean = A @ x_bar + B @ u_bar
    new_state_cov = (A + B @ K) @ Sigma_x @ (A + B @ K).T + B @ K @ Sigma_noise @ K.T @ B.T

    # Adding the diagonal element to Sigma_u to ensure it is positive-definite for use elsewhere.
    return (new_state_mean, new_state_cov), (cost, x_bar, Sigma_x, u_bar, Sigma_u + 1e-11 * jnp.eye(prob.prototype.dynamics.num_inputs))


def cost_to_go(prob: Problem, state: State,
               init_state_cov: Optional[jnp.ndarray] = None, noise_cov: Optional[jnp.ndarray] = None, t: int = 0) -> float:
    horizon = prob.prototype.horizon
    temporal_info, _ = lqr_dynamic_programming(prob.params, horizon - t)
    n = prob.prototype.dynamics.num_states
    _, _, Qf, _, _ = prob.params.objective

    if noise_cov is None:
        noise_cov = jnp.zeros((n, n, horizon - t))

    if init_state_cov is None:
        init_state_cov = jnp.zeros((n, n))

    augmented_temporal_info = (jnp.transpose(noise_cov, [2, 1, 0]), temporal_info)

    scanner = jax.jit(lambda c, t: rollout_scanner(c, t, prob))

    carry, temporal = jax.lax.scan(scanner, (state, init_state_cov), augmented_temporal_info)
    x_bar, Sigma_x = carry
    costs = temporal[0]

    terminal_cost = 0.5 * (x_bar.T @ Qf @ x_bar + jnp.trace(Qf @ Sigma_x))

    return costs.sum() + terminal_cost


def input_stats(prob: Problem, state: State,
                init_state_cov: Optional[jnp.ndarray] = None,
                noise_cov: Optional[jnp.ndarray] = None, t: int = 0) -> dst.GaussianParams:
    horizon = prob.prototype.horizon
    temporal_info, _ = lqr_dynamic_programming(prob.params, horizon - t)
    n = prob.prototype.dynamics.num_states
    _, _, Qf, _, _ = prob.params.objective

    if noise_cov is None:
        noise_cov = jnp.zeros((n, n, horizon - t))

    if init_state_cov is None:
        init_state_cov = jnp.zeros((n, n))

    augmented_temporal_info = (jnp.transpose(noise_cov, [2, 1, 0]), temporal_info)

    scanner = jax.jit(lambda c, t: rollout_scanner(c, t, prob))

    carry, temporal = jax.lax.scan(scanner, (state, init_state_cov), augmented_temporal_info)

    return dst.GaussianParams(temporal[3], temporal[4])
