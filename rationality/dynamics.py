from typing import NewType, Tuple, Callable

import jax
import jax.numpy as jnp

Dynamics = NewType('Dynamics', Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray])


def create_dynamics(dynamics_prototype: Callable[..., jnp.ndarray], params: Tuple) -> Dynamics:
    return Dynamics(lambda x, u, t: dynamics_prototype(x, u, t, *params))


@jax.jit
def linear_prototype(state: jnp.ndarray, input: jnp.ndarray, t: int, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """
    The prototype for a linear dynamical system.

    :param state: The system state
    :param input: The system input
    :param t: The current time (unused)
    :param A: The state matrix
    :param B: The control matrix

    :return: The next state: x' = Ax + Bu
    """
    return A @ state + B @ input


@jax.jit
def quad2d_prototype(state: jnp.ndarray, input: jnp.ndarray, t: int,
                     dt: float, mass: float, gravity: float, inertia: float) -> jnp.ndarray:
    """
    The prototype for a planar quadrotor dynamical system.

    The system state for the quadrotor is: (x, y, theta, vx, vy, vtheta)
    The system inputs for the quadrotor are: (thrust, moment)

    The rotation of the quadrotor is theta. The variables vx, vy, vtheta are the time derivatives of
    x, y, and theta respectively.

    :param state: The system state
    :param input: The system input
    :param t: The current time (unused)

    :param dt: The timestep used for time discretization.
    :param mass: The mass of the quadrotor.
    :param gravity: The acceleration due to gravity.
    :param inertia: The inertia of the quadrotor.

    :return: The next state.
    """
    return jnp.array([state[0] + dt * state[3],
                      state[1] + dt * state[4],
                      state[2] + dt * state[5],
                      state[3] - dt * input[0] * jnp.sin(state[2]) / mass,
                      state[4] + dt * (input[0] * jnp.cos(state[2]) / mass - gravity),
                      state[5] + dt * input[1] / inertia]).flatten()


def linearize(dynamics: Dynamics, state: jnp.ndarray, input: jnp.ndarray, t: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Linearize the dynamics at a fixed-point.

    :param dynamics: The dynamics to be linearized.
    :param state: The system state at the fixed-point.
    :param input: The system input at the fixed-point.
    :param t: The time at which to evaluate the dynamics
    :return: A pair of matrices (A, B), representing the state and input matrices.
    """
    return jax.jacfwd(lambda x: dynamics(x, input, t))(state), jax.jacfwd(lambda u: dynamics(state, u, t))(input)
