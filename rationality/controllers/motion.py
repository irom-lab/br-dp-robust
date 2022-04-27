from ast import Call
import jax
import jax.numpy as jnp

import rationality.distributions as dst

from typing import NamedTuple, Union, Iterable

from rationality.controllers.types import *

MPTemporalInfo = None
MPControllerState = None

class MPParams(NamedTuple):
    inv_temp: float


def create(prob: Problem, prior_proto: dst.DistributionPrototype,
           prior_params: dst.DistributionParams) -> Controller:
    return Controller(jax.jit(lambda prob_params, params, key: lqr_init_prototype(prob_params, params, key,
                                                                                  prob.prototype.horizon)),
                      lqr_prototype, None)


def rollout(ic: State, inputs: Input, it: int,
            dyn: Callable[[State, Input, State], State]) -> State:
    @jax.jit
    def _scanner(carry: tuple[State, int], input: Input) -> tuple[tuple[State, int], State]:
        state, t = carry
        new_state = dyn(state, input, t)

        return (new_state, t + 1), state
    
    carry, states = jax.lax.scan(_scanner, (ic, it), inputs)

    return jnp.concatenate([states, carry[0].reshape((1, -1))])           

def process_traj(states: State, inputs: Input, it: int,
                 traj_cost, term_cost, stop_cond):
    state_pairs = jax.vmap(lambda state, next_state: (state, next_state))(states[:-1], states[1:])

    @jax.jit
    def _scanner(carry: tuple[int, bool, float], traj_data: tuple[tuple[State, State], Input]) -> tuple[int, bool, float]:
        t, stop, term_cost_carry = carry
        state_pair, input = traj_data
        
        stop = stop | stop_cond(state_pair[0], input, state_pair[1])
        cost = jax.lax.cond(stop,
                            lambda _: 0.0,
                            lambda x, t, u, x_next: traj_cost(x, u, t, x_next),
                            (state_pair[0], t, input, state_pair[1]))
        
        term_cost_carry = jax.lax.cond(stop,
                                 lambda _:)

    


@jax.jit
def mp_prototype(state: State, t: int, controller_state: MPControllerState,
                 temporal_info: MPTemporalInfo, horizon: int, params: MPParams, prob: Problem) -> tuple[Input, MPControllerState]:
    prob_proto, prob_params = prob
    dyn = jax.jit(lambda x, u, t: prob_proto.dynamics(x, u, t))

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
