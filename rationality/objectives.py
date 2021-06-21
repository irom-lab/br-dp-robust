from typing import Callable, Tuple, NamedTuple, Union, Any

import jax
import jax.numpy as jnp

from rationality.types import State, Input


TrajectoryObjectivePrototype = Callable[[State, Input, int, Any], float]
TerminalObjectivePrototype = Callable[[State, Any], float]


class Objective(NamedTuple):
    trajectory_prototype: TrajectoryObjectivePrototype
    terminal_prototype: TerminalObjectivePrototype
    params: Any

    def __call__(self, *args: Union[Tuple[State, Input, int], Tuple[State]]) -> float:
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


def quadratic(Q: jnp.ndarray, R: jnp.ndarray, Qf: jnp.ndarray) -> Objective:
    @jax.jit
    def trajectory_obj(x: State, u: Input, t: int, params: Quadratic) -> float:
        Q, R, _ = params

        return 0.5 * x.T @ Q @ x + 0.5 * u.T @ R @ u

    @jax.jit
    def terminal_obj(x: State, params: Quadratic) -> float:
        _, _, Qf = params

        return 0.5 * x.T @ Qf @ x

    return Objective(trajectory_obj, terminal_obj, Quadratic(Q, R, Qf))