from typing import Callable, Tuple, NamedTuple

import jax
import jax.numpy as jnp

from rationality.types import State, Input


class Objective(NamedTuple):
    fn: Callable[[State, Input, int], float]
    params: Tuple

    def __call__(self, state: State, input: Input, t: int) -> float:
        return self.fn(state, input, t)


class Quadratic(NamedTuple):
    Q: jnp.ndarray
    R: jnp.ndarray
    Qf: jnp.ndarray
    horizon: int


def quadratic(Q: jnp.ndarray, R: jnp.ndarray, Qf: jnp.ndarray, horizon: int) -> Objective:
    params = Quadratic(Q, R, Qf, horizon)

    return Objective(jax.jit(lambda x, u, t: quad_obj_prototype(x, u, t, params)), params)


@jax.jit
def quad_obj_prototype(state: jnp.ndarray, input: jnp.ndarray, t: int, params: Quadratic) -> float:
    Q, R, Qf, horizon = params

    def trajectory_obj(xu: Tuple[jnp.ndarray, jnp.ndarray]) -> float:
        x, u = xu
        return x.T @ Q @ x + u.T @ R @ u

    def terminal_obj(xu: Tuple[jnp.ndarray, jnp.ndarray]) -> float:
        x, u = xu
        return x.T @ Qf @ x

    return jax.lax.switch(t - (horizon - 1), (trajectory_obj, terminal_obj, lambda _: jnp.array(0.0)), (state, input))
