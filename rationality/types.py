from __future__ import annotations
from typing import TypeVar, NamedTuple, Union

import jax.numpy as jnp
import numpy as np

State = jnp.ndarray
Input = jnp.ndarray

Array = TypeVar('Array', jnp.ndarray, np.ndarray)


class Trajectory(NamedTuple):
    states: Array
    inputs: Array
    costs: Array

    def asnumpy(self) -> Trajectory:
        return Trajectory(np.asarray(self.states), np.asarray(self.inputs), np.asarray(self.costs))

    def structured(self) -> np.ndarray:
        s = self[0]

        dtype = np.dtype([('states', s.states.dtype, s.states.shape),
                          ('inputs', s.inputs.dtype, s.inputs.shape),
                          ('costs', s.costs.dtype, s.costs.shape)])

        return np.array([(x, u, c) for x, u, c in zip(self.states, self.inputs, self.costs)], dtype=dtype)

    def asjax(self) -> Trajectory:
        return Trajectory(jnp.asarray(self.states), jnp.asarray(self.inputs), jnp.asarray(self.costs))

    def __getitem__(self, s: Union[int, slice]) -> Trajectory:
        return Trajectory(self.states[s, :, :], self.inputs[s, :, :], self.costs[s, :])