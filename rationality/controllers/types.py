from typing import Any, Callable, Tuple, NamedTuple

import rationality.dynamics as dyn
import rationality.objectives as obj
from rationality.types import State, Input


class ControlProblem(NamedTuple):
    dynamics: dyn.Dynamics
    objective: obj.Objective
    horizon: int


ControllerState = Any
ControllerTemporalInfo = Any

ControllerPrototype = Callable[[State, int, ControllerState, ControllerTemporalInfo, Any],
                               Tuple[Input, ControllerState]]

ControllerInitPrototype = Callable[[ControlProblem, Any], ControllerTemporalInfo]


class Controller(NamedTuple):
    prob: ControlProblem

    init_prototype: ControllerInitPrototype
    controller_prototype: ControllerPrototype

    params: Any

    def init(self) -> Tuple[ControllerState, ControllerTemporalInfo]:
        return self.init_prototype(self.prob, self.params)

    def __call__(self, state: State, t: int, controller_state: ControllerState,
                 temporal_info: ControllerTemporalInfo) -> Tuple[Input, ControllerState]:
        return self.controller_prototype(state, t, controller_state, temporal_info, self.params)