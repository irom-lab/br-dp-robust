import jax
import jax.numpy as jnp

from rationality.controllers.types import *
from rationality.distributions import GaussianParams, gaussian

LQBRTemporalInfo = tuple[jnp.ndarray, GaussianParams]
LQBRControllerState = jnp.ndarray


class LQBRParams(NamedTuple):
    inv_temp: float
    init_key: jnp.ndarray
    prior_params: GaussianParams


def create(prob: Problem, prior_params: list[GaussianParams],
           inv_temp: float, init_key: jnp.ndarray) -> Controller:
    prior_params_for_scanning = (jnp.stack([p.mean for p in prior_params]), jnp.stack([p.cov for p in prior_params]))

    return Controller(jax.jit(lambda prob_params, params: lqbr_init_prototype(prob_params,
                                                                              LQBRParams(inv_temp, init_key, prior_params_for_scanning),
                                                                              prob.prototype.horizon)),
                      lqbr_prototype, None)


@jax.jit
def lqbr_scanner(carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                 prior_params: GaussianParams, A: jnp.ndarray, B: jnp.ndarray,
                 Q: jnp.ndarray, R: jnp.ndarray,
                 inv_temp: float) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], LQBRTemporalInfo]:
    P, b, d = carry
    prior_mean, prior_cov = prior_params
    prior_cov_inv = jnp.linalg.inv(prior_cov)

    Sigma_eta_inv = inv_temp * B.T @ P @ B + inv_temp * R + prior_cov_inv
    Sigma_eta = jnp.linalg.inv(Sigma_eta_inv)
    eta_bar = -Sigma_eta @ (inv_temp * B.T @ b - prior_cov_inv @ prior_mean)

    K = -inv_temp * Sigma_eta @ B.T @ P @ A

    P_next = Q + K.T @ R @ K + A.T @ P @ A + A.T @ P @ B @ K
    b_next = eta_bar.reshape((1, -1)) @ (R @ K + B.T @ P @ A + B.T @ P @ B @ K) + b.reshape((1, -1)) @ (A + B @ K)
    d_next = d + 0.5 * eta_bar @ (R + B.T @ P @ B) @ eta_bar + b.T @ B @ eta_bar + 0.5 * jnp.trace((R + B.T @ P @ B) @ Sigma_eta)

    return (P_next, b_next.flatten(), d_next), (K, GaussianParams(eta_bar, Sigma_eta))


def lqbr_dynamic_programming(prob_params: ProblemParams, params: LQBRParams,
                             horizon: int) -> tuple[LQBRTemporalInfo, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    dynamics, objective = prob_params
    A, B = dynamics
    Q, R, Qf, _, _ = objective

    n, m = B.shape

    init = (Qf, jnp.zeros(n), jnp.array(0.0))
    value_function, temporal_info = jax.lax.scan(lambda c, _: lqbr_scanner(c, _, A, B, Q, R, params.inv_temp), init,
                                                 params.prior_params, length=horizon, reverse=True)

    return temporal_info, value_function


def lqbr_init_prototype(prob_params: ProblemParams,
                        params: LQBRParams,
                        horizon: int) -> tuple[LQBRControllerState, LQBRTemporalInfo]:
    temporal_info, _ = lqbr_dynamic_programming(prob_params, params, horizon)

    return params.init_key, temporal_info


@jax.jit
def lqbr_prototype(state: State, t: int, controller_state: LQBRControllerState,
                   temporal_info: LQBRTemporalInfo, params: LQBRParams) -> tuple[Input, LQBRControllerState]:
    K, dist_params = temporal_info
    key, subkey = jax.random.split(controller_state)

    return K @ state + gaussian(dist_params.mean, dist_params.cov).sample(1, subkey).flatten(), key


def cost_to_go(state: State, t: int, prior_means: jnp.ndarray, prior_covs: jnp.ndarray, prob: Problem) -> float:
    temporal_info, _ = lqbr_dynamic_programming(prob.params, prior_means, prior_covs, prob.prototype.horizon)

    return jnp.nan
