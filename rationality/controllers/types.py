from typing import Any, Callable, NamedTuple

import rationality.dynamics as dyn
import rationality.objectives as obj
from rationality.types import State, Input, ObjectiveParams, DynamicsParams, StoppingCondition

import jax
import jax.numpy as jnp


class ProblemPrototype(NamedTuple):
    """
    The prototype for a control problem.

    This structure contains
    """
    dynamics: dyn.DynamicsPrototype
    trajectory_objective: obj.TrajectoryObjectivePrototype
    terminal_objective: obj.TerminalObjectivePrototype
    horizon: int


class ProblemParams(NamedTuple):
    dynamics: DynamicsParams
    objective: ObjectiveParams


class Problem(NamedTuple):
    prototype: ProblemPrototype
    params: ProblemParams

    @property
    def num_states(self) -> int:
        return self.prototype.dynamics.num_states

    @property
    def num_inputs(self) -> int:
        return self.prototype.dynamics.num_inputs


ControllerState = Any
ControllerTemporalInfo = Any

ControllerPrototype = Callable[[State, int, ControllerState, ControllerTemporalInfo, Any],
                               tuple[Input, ControllerState]]

ControllerInitPrototype = Callable[[ProblemParams, Any, jnp.ndarray], tuple[ControllerState, ControllerTemporalInfo]]


class Controller(NamedTuple):
    init_prototype: ControllerInitPrototype
    controller_prototype: ControllerPrototype
    params: Any

    def init(self, prob_params: ProblemParams, key: jnp.ndarray) -> tuple[ControllerState, ControllerTemporalInfo]:
        return self.init_prototype(prob_params, self.params, key)

    def __call__(self, state: State, t: int, controller_state: ControllerState,
                 temporal_info: ControllerTemporalInfo) -> tuple[Input, ControllerState]:
        return self.controller_prototype(state, t, controller_state, temporal_info, self.params)

def open_loop(inputs: Input) -> Controller:
    def _init_open_loop_prototype(*args) -> tuple[None, inputs]:
        return None, inputs
    
    def _open_loop_prototype(state: State, t: int, ctl_state: State, temp_info: Input, params: None) -> tuple[Input, None]:
        return temp_info, None

    return Controller(jax.jit(_init_open_loop_prototype), jax.jit(_open_loop_prototype), None)



def problem(dynamics: dyn.Dynamics, objective: obj.Objective, horizon: int) -> Problem:
    return Problem(ProblemPrototype(dynamics.prototype, objective.trajectory_prototype,
                                    objective.terminal_prototype, horizon),
                   ProblemParams(dynamics.params, objective.params))
