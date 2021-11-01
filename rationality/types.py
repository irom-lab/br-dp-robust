from __future__ import annotations
from typing import TypeVar, NamedTuple, Union, NewType

import jax.numpy as jnp
import numpy as np

State = jnp.ndarray  #: Type alias to enhance readability of type signatures. Typically represents an n-dimensional system state
Input = jnp.ndarray  #: Type alias to enhance readability of type signatures. Typically represents an m-dimensional system input

Array = TypeVar('Array', jnp.ndarray, np.ndarray)  #: Generic type for handling either form of array.

ObjectiveParams = NewType('ObjectiveParams', NamedTuple)
DynamicsParams = NewType('DynamicsParams', NamedTuple)


class Trajectory(NamedTuple):
    """
    Structure that encapsulates all variables describing a system trajectory or multiple trajectories.

    If the structure represents multiple trajectories, the first array index is the trajectory index for each
    attribute and the remaining indices represent the value and time step for the variable at that trajectory.

    If the instance represents a single trajectory, the indices represent the value and temporal index respectively.

    Attributes:
        states: The system
    """
    states: State
    inputs: State
    costs: Array

    def asnumpy(self) -> Trajectory:
        """
        Converts the Trajectory instance to use NumPy arrays.
        """
        return Trajectory(np.asarray(self.states), np.asarray(self.inputs), np.asarray(self.costs))

    def asjax(self) -> Trajectory:
        """
        Converts the Trajectory instance to use JAX arrays.
        """
        return Trajectory(jnp.asarray(self.states), jnp.asarray(self.inputs), jnp.asarray(self.costs))

    def to_structured(self) -> np.ndarray:
        """
        Converts the Trajectory instance to a structures (JAX or NumPy) array.
        """
        s = self[0]

        dtype = np.dtype([('states', s.states.dtype, s.states.shape),
                          ('inputs', s.inputs.dtype, s.inputs.shape),
                          ('costs', s.costs.dtype, s.costs.shape)])

        return np.array([(x, u, c) for x, u, c in zip(self.states, self.inputs, self.costs)], dtype=dtype)

    def __getitem__(self, s: Union[int, slice]) -> Trajectory:
        """
        Returns a subset of trajectories.

        :param s: The trajectory index or slice identifying which subset of trajectories to index.

        :returns: A new Trajectory instance representing the selected trajectory or trajectories.
        """
        return Trajectory(self.states[s, :, :], self.inputs[s, :, :], self.costs[s, :])
