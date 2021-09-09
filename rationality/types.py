from __future__ import annotations
from typing import TypeVar, NamedTuple

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

    def asjax(self) -> Trajectory:
        return Trajectory(jnp.asarray(self.states), jnp.asarray(self.inputs), jnp.asarray(self.costs))

    def __getitem__(self, s: slice) -> Trajectory:
        return Trajectory(self.states[s, :, :], self.inputs[s, :, :], self.costs[s, :, :])