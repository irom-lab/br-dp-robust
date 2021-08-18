from typing import Callable, NamedTuple, Union, Any, Optional

import jax
import jax.numpy as jnp

from rationality.types import State, Input


TrajectoryObjectivePrototype = Callable[[State, Input, int, Any], float]
TerminalObjectivePrototype = Callable[[State, Any], float]


class Objective(NamedTuple):
    trajectory_prototype: TrajectoryObjectivePrototype
    terminal_prototype: TerminalObjectivePrototype
    params: Any

    def __call__(self, *args: Union[tuple[State, Input, int], tuple[State]]) -> float:
        if len(args) == 3:
            return self.trajectory_prototype(*args, self.params)
        elif len(args) == 1:
            return self.terminal_prototype(*args, self.params)
        else:
            raise ValueError('Requires either 1 or 3 arguments.')


class Quadratic(NamedTuple):
    Q: jnp.ndarray
    R: jnp.ndarray
    Qf: jnp.ndarray
    state_offset: Optional[jnp.ndarray]
    input_offset: Optional[jnp.ndarray]


def quadratic(Q: jnp.ndarray, R: jnp.ndarray, Qf: jnp.ndarray,
              state_offset: Optional[jnp.ndarray] = None, input_offset: Optional[jnp.ndarray] = None) -> Objective:
    if state_offset is None:
        state_offset = jnp.zeros(Q.shape[0])

    if input_offset is None:
        input_offset = jnp.zeros(R.shape[0])

    @jax.jit
    def trajectory_obj(x: State, u: Input, t: int, params: Quadratic) -> float:
        Q, R, _, state_offset, input_offset = params

        return 0.5 * (x - state_offset).T @ Q @ (x - state_offset) + 0.5 * (u - input_offset).T @ R @ (u - input_offset)

    @jax.jit
    def terminal_obj(x: State, params: Quadratic) -> float:
        _, _, Qf, state_offset, input_offset = params

        return 0.5 * (x - state_offset).T @ Qf @ (x - state_offset)

    return Objective(trajectory_obj, terminal_obj, Quadratic(Q, R, Qf, state_offset, input_offset))