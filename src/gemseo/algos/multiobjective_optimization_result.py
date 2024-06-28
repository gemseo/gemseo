# Copyright 2022 Airbus SAS
# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Gabriel Max DE MENDONÇA ABRANTES
"""Multi-objective optimization result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from numpy import ndarray

from gemseo.algos.optimization_result import OptimizationResult
from gemseo.algos.pareto.pareto_front import ParetoFront
from gemseo.utils.string_tools import MultiLineString


@dataclass
class MultiObjectiveOptimizationResult(OptimizationResult):
    """The result of a multi-objective optimization."""

    pareto_front: ParetoFront | None = None
    """The Pareto front when the solution is feasible."""

    __PARETO_FRONT: Final[str] = "pareto_front"

    def __repr__(self) -> str:
        msg = MultiLineString()
        msg.add("Multi-objective optimization result:")
        msg.indent()
        msg.add("Design variables: {}", self.x_opt)
        msg.add("Objective function: {}", self.f_opt)
        msg.add("Feasible solution: {}", self.is_feasible)
        if self.pareto_front is not None:
            msg.add("Pareto front:")
            msg.indent()
            msg.add("Number of points: {}", self.pareto_front.f_optima.shape[0])
            msg.add("Distance from utopia: {}", self.pareto_front.distance_from_utopia)
            msg.dedent()
        return str(msg)

    def __str__(self) -> str:
        parent_string = super().__str__()
        if self.pareto_front is not None:
            msg = MultiLineString()
            msg.indent()
            msg.add("Pareto efficient solutions:")
            msg.indent()
            for line in str(self.pareto_front).split("\n"):
                msg.add("{}", line)
            return f"{parent_string}\n{msg}"
        return parent_string

    def to_dict(self):  # noqa: D102
        dict_ = super().to_dict()
        pareto_front = dict_.pop(self.__PARETO_FRONT)
        if pareto_front is not None:
            for attr, value in self.pareto_front.__dict__.items():
                if isinstance(value, (int, float, str, bool, list, tuple, ndarray)):
                    dict_[f"{self.__PARETO_FRONT}:{attr}"] = value

        return dict_

    @classmethod
    def _get_additional_fields(cls, problem):
        return {
            f"{cls.__PARETO_FRONT}": ParetoFront.from_optimization_problem(problem)
            if problem.optimum[2]
            else None
        }
