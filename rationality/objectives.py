import jax
import jax.numpy as jnp

from typing import NewType, Callable, Tuple
from functools import partial

Objective = NewType('Objective', Callable[[jnp.ndarray, jnp.ndarray, int], float])


def create_objective(objective_prototype: Callable[..., jnp.ndarray], params: Tuple) -> Objective:
    return Objective(lambda x, u, t: objective_prototype(x, u, t, *params))


@jax.jit
def quad_obj_prototype(state: jnp.ndarray, input: jnp.ndarray, t: int,
                       Q: jnp.ndarray, R: jnp.ndarray, Qf: jnp.ndarray, horizon: int) -> float:

    def trajectory_obj(xu: Tuple[jnp.ndarray, jnp.ndarray]) -> float:
        x, u = xu
        return x.T @ Q @ x + u.T @ R @ u

    def terminal_obj(xu: Tuple[jnp.ndarray, jnp.ndarray]) -> float:
        x, u = xu
        return x.T @ Qf @ x

    return jax.lax.switch(t - (horizon - 1), (trajectory_obj, terminal_obj, lambda _: jnp.array(0.0)), (state, input))