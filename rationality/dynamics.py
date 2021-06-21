from typing import Tuple, Callable, NamedTuple, Any

import jax
import jax.numpy as jnp

from rationality.types import State, Input

class DynamicsPrototype(NamedTuple):
    mapping: Callable[[State, Input, int, Any], State]
    num_states: int
    num_inputs: int

    def __call__(self, state: State, input: Input, t: int, params: Any) -> State:
        return self.mapping(state, input, t, params)


class Dynamics(NamedTuple):
    prototype: DynamicsPrototype
    params: Tuple

    def __call__(self, state: State, input: Input, t: int) -> State:
        return self.prototype(state, input, t, self.params)


class Linear(NamedTuple):
    A: jnp.ndarray
    B: jnp.ndarray


class Quad2D(NamedTuple):
    dt: float
    mass: float
    gravity: float
    inertia: float

    @property
    def hover_force(self) -> float:
        return self.mass * self.gravity


def linear(A: jnp.ndarray, B: jnp.ndarray) -> Dynamics:
    params = Linear(A, B)

    return Dynamics(DynamicsPrototype(linear_prototype, *B.shape), params)


def quad2d(dt: float, mass: float, gravity: float, inertia: float) -> Dynamics:
    params = Quad2D(dt, mass, gravity, inertia)

    return Dynamics(DynamicsPrototype(quad2d_prototype, 6, 2), params)


def crazyflie2d(dt: float) -> Dynamics:
    return quad2d(dt, 0.03, 9.82, 1.43e-5)


@jax.jit
def linear_prototype(state: State, input: State, t: int, params: Linear) -> jnp.ndarray:
    """
    The prototype for a linear dynamical system.

    :param state: The system state
    :param input: The system input
    :param t: The current time (unused)
    :param params: The parameters for the model. See `Linear` for details.

    :return: The next state: x' = Ax + Bu
    """
    A, B = params

    return A @ state + B @ input


@jax.jit
def quad2d_prototype(state: State, input: State, t: int, params: Quad2D) -> jnp.ndarray:
    """
    The prototype for a planar quadrotor dynamical system.

    The system state for the quadrotor is: (x, y, theta, vx, vy, vtheta)
    The system inputs for the quadrotor are: (thrust, moment)

    The rotation of the quadrotor is theta. The variables vx, vy, vtheta are the time derivatives of
    x, y, and theta respectively.

    :param state: The system state
    :param input: The system input
    :param t: The current time (unused)
    :param params: The parameters for the model. See `Quad2D` for details.

    :return: The next state.
    """
    dt, mass, gravity, inertia = params

    return jnp.array([state[0] + dt * state[3],
                      state[1] + dt * state[4],
                      state[2] + dt * state[5],
                      state[3] - dt * input[0] * jnp.sin(state[2]) / mass,
                      state[4] + dt * (input[0] * jnp.cos(state[2]) / mass - gravity),
                      state[5] + dt * input[1] / inertia]).flatten()


def linearize(dynamics: Dynamics, state: State, input: Input, t: int) -> Linear:
    """
    Linearize the dynamics at a fixed-point.

    :param dynamics: The dynamics to be linearized.
    :param state: The system state at the fixed-point.
    :param input: The system input at the fixed-point.
    :param t: The time at which to evaluate the dynamics.

    :return: A pair of matrices (A, B), representing the state and input matrices.
    """
    return Linear(jax.jacfwd(lambda x: dynamics(x, input, t))(state),
                  jax.jacfwd(lambda u: dynamics(state, u, t))(input))
