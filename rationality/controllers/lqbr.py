import jax
import jax.numpy as jnp

from typing import Optional, Union

from rationality.controllers.types import *
from rationality.distributions import GaussianParams, gaussian

LQBRTemporalInfo = tuple[jnp.ndarray, GaussianParams]
LQBRControllerState = jnp.ndarray


class LQBRParams(NamedTuple):
    inv_temp: float
    init_key: jnp.ndarray
    prior_params: GaussianParams


def create(prob: Problem, prior_params: Union[list[GaussianParams], GaussianParams],
           inv_temp: float, init_key: jnp.ndarray) -> Controller:
    if type(prior_params) == list:
        prior_params_for_scanning = GaussianParams(jnp.stack([p.mean for p in prior_params]), jnp.stack([p.cov for p in prior_params]))
    elif type(prior_params) == GaussianParams:
        prior_params_for_scanning = prior_params
    else:
        raise TypeError('Expected `prior_params` to be either `list[GaussianParams]` or `GaussianParams`.')

    return Controller(jax.jit(lambda prob_params, params: lqbr_init_prototype(prob_params,
                                                                              LQBRParams(params.inv_temp, params.init_key, prior_params_for_scanning),
                                                                              prob.prototype.horizon)),
                      lqbr_prototype, LQBRParams(inv_temp, init_key, prior_params_for_scanning))


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

    value_function_params = (P_next, b_next.flatten(), d_next)

    return value_function_params, (K, GaussianParams(eta_bar, Sigma_eta), carry)


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

    return params.init_key, (temporal_info[0], temporal_info[1])


@jax.jit
def lqbr_prototype(state: State, t: int, controller_state: LQBRControllerState,
                   temporal_info: LQBRTemporalInfo, params: LQBRParams) -> tuple[Input, LQBRControllerState]:
    K, dist_params = temporal_info
    key, subkey = jax.random.split(controller_state)

    return K @ state + gaussian(dist_params.mean, dist_params.cov).sample(1, subkey).flatten(), key


def cost_to_go(prob: Problem, params: LQBRParams, state: State,
               init_state_cov: Optional[jnp.ndarray] = None, noise_cov: Optional[jnp.ndarray] = None, t: int = 0) -> float:
    temporal_info, _ = lqbr_dynamic_programming(prob.params, params, prob.prototype.horizon)
    Q, R, Qf, _, _ = prob.params.objective
    A, B = prob.params.dynamics
    n, m = B.shape
    horizon = prob.prototype.horizon

    if noise_cov is None:
        noise_cov = jnp.zeros((n, n, horizon - t))

    if init_state_cov is None:
        init_state_cov = jnp.zeros((n, n))

    augmented_temporal_info = (jnp.transpose(noise_cov, [2, 1, 0]), temporal_info[0], temporal_info[1][0], temporal_info[1][1])

    @jax.jit
    def cost_to_go_scanner(carry: tuple[jnp.ndarray, jnp.ndarray],
                           temporal_info: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> tuple[tuple[jnp.ndarray, jnp.ndarray], float]:
        x_bar, Sigma_x = carry
        Sigma_noise, K, eta_bar, Sigma_eta = temporal_info

        Sigma_u = K @ (Sigma_x + Sigma_noise) @ K.T + Sigma_eta
        u_bar = K @ x_bar + eta_bar

        cost = 0.5 * (x_bar.T @ Q @ x_bar + u_bar.T @ R @ u_bar + jnp.trace(Q @ Sigma_x) + jnp.trace(R @ Sigma_u))

        new_state_mean = A @ x_bar + B @ u_bar
        new_state_cov = (A + B @ K) @ Sigma_x @ (A + B @ K).T + B @ Sigma_eta @ B.T + B @ K @ Sigma_noise @ K.T @ B.T

        return (new_state_mean, new_state_cov), cost

    carry, costs = jax.lax.scan(cost_to_go_scanner, (state, init_state_cov), augmented_temporal_info)
    x_bar, Sigma_x = carry

    terminal_cost = 0.5 * (x_bar.T @ Qf @ x_bar + jnp.trace(Qf @ Sigma_x))


    return costs.sum() + terminal_cost


def state_distribution(prob: Problem, params: LQBRParams, state: State,
                       init_state_cov: Optional[jnp.ndarray] = None,
                       noise_cov: Optional[jnp.ndarray] = None, t: int = 0) -> GaussianParams:
    temporal_info, _ = lqbr_dynamic_programming(prob.params, params, prob.prototype.horizon)
    Q, R, Qf, _, _ = prob.params.objective
    A, B = prob.params.dynamics
    n, m = B.shape
    horizon = prob.prototype.horizon

    if noise_cov is None:
        noise_cov = jnp.zeros((n, n, horizon - t))

    if init_state_cov is None:
        init_state_cov = jnp.zeros((n, n))

    init_dist = GaussianParams(state, init_state_cov)

    augmented_temporal_info = (jnp.transpose(noise_cov, [2, 1, 0]), temporal_info[0], temporal_info[1][0], temporal_info[1][1])

    @jax.jit
    def cost_to_go_scanner(carry: GaussianParams, temporal_info: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> tuple[GaussianParams, GaussianParams]:
        x_bar, Sigma_x = carry
        Sigma_noise, K, eta_bar, Sigma_eta = temporal_info

        u_bar = K @ x_bar + eta_bar

        new_state_mean = A @ x_bar + B @ u_bar
        new_state_cov = (A + B @ K) @ Sigma_x @ (A + B @ K).T + B @ Sigma_eta @ B.T + B @ K @ Sigma_noise @ K.T @ B.T

        return GaussianParams(new_state_mean, new_state_cov), GaussianParams(x_bar, Sigma_x)

    terminal_dist, stagewise_dists = jax.lax.scan(cost_to_go_scanner, init_dist, augmented_temporal_info)

    return GaussianParams(jnp.append(stagewise_dists.mean, terminal_dist.mean.reshape((1, -1)), axis=0),
                          jnp.append(stagewise_dists.cov, jnp.expand_dims(terminal_dist.cov, 0), axis=0))


@jax.jit
def lipschitz_scanner(carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                      temporal_info: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                      A: jnp.ndarray, B: jnp.ndarray,
                      Q: jnp.ndarray, R: jnp.ndarray,
                      inv_temp: float) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], float]:
    P, b, d = carry

    prior_mean, prior_cov, input = temporal_info
    fro = jnp.linalg.norm(Q + A.T @ P @ A, ord='fro')
    affine = A.T @ P @ B @ input + A.T @ b
    lipschitz = jnp.maximum(fro, jnp.linalg.norm(affine, ord=2)).astype(float)

    return lqbr_scanner(carry, GaussianParams(prior_mean, prior_cov), A, B, Q, R, inv_temp)[0], lipschitz


def lipschitz_constants(prob: Problem, prior_params: GaussianParams,
                        inv_temp: float, inputs: jnp.ndarray) -> jnp.ndarray:
    dynamics, objective = prob.params
    A, B = dynamics
    Q, R, Qf, _, _ = objective
    n, m = B.shape

    init = (Qf, jnp.zeros(n), jnp.array(0.0))
    _, lipschitz = jax.lax.scan(lambda carry, temporal: lipschitz_scanner(carry, temporal, A, B, Q, R, inv_temp),
                                init, (prior_params.mean, prior_params.cov, inputs.T),
                                length=prob.prototype.horizon, reverse=True)

    return lipschitz