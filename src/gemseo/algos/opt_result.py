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
# Contributors:
#    Francois Gallard
#    Matthias De Lozzo
"""Optimization result."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Union

from numpy import ndarray

from gemseo.utils.string_tools import MultiLineString

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.algos.opt_problem import OptimizationProblem

Value = Union[str, int, bool, ndarray]


# TODO: API: Rename the module to optimization_result.
@dataclass
class OptimizationResult:
    """The result of an optimization."""

    x_0: ndarray | None = None
    """The initial values of the design variables."""

    x_0_as_dict: dict[str, ndarray] = field(default_factory=dict)
    """The design variable names bound to the initial design values."""

    x_opt: ndarray | None = None
    """The optimal values of the design variables, called the *optimum*."""

    x_opt_as_dict: dict[str, ndarray] = field(default_factory=dict)
    """The design variable names bound to the optimal design values."""

    f_opt: ndarray | None = None
    """The value of the objective function at the optimum."""

    objective_name: str = ""
    """The name of the objective function."""

    status: int | None = None
    """The status of the optimization."""

    optimizer_name: str | None = None
    """The name of the optimizer."""

    message: str | None = None
    """The message returned by the optimizer."""

    n_obj_call: int | None = None
    """The number of calls to the objective function."""

    n_grad_call: int | None = None
    """The number of calls to the gradient function."""

    n_constr_call: int | None = None
    """The number of calls to the constraints function."""

    is_feasible: bool = False
    """Whether the solution is feasible."""

    optimum_index: int | None = None
    """The zero-based position of the optimum in the optimization history."""

    constraint_values: Mapping[str, ndarray] | None = None
    """The values of the constraints at the optimum."""

    constraints_grad: Mapping[str, ndarray | None] | None = None
    """The values of the gradients of the constraints at the optimum."""

    __CGRAD_TAG = "constr_grad:"
    __CGRAD_TAG_LEN = len(__CGRAD_TAG)
    __C_TAG = "constr:"
    __C_TAG_LEN = len(__C_TAG)
    __CONSTRAINTS_VALUES = "constraint_values"
    __CONSTRAINTS_GRAD = "constraints_grad"
    __NOT_DICT_KEYS: ClassVar[list[str]] = [__CONSTRAINTS_VALUES, __CONSTRAINTS_GRAD]

    @property
    def __string_representation(self) -> MultiLineString:
        """The string representation of the optimization result."""
        mls = MultiLineString()
        mls.add("Optimization result:")
        mls.indent()
        mls.add("Design variables: {}", self.x_opt)
        mls.add("Objective function: {}", self.f_opt)
        mls.add("Feasible solution: {}", self.is_feasible)
        return mls

    def __repr__(self) -> str:
        return str(self.__string_representation)

    def _repr_html_(self) -> str:
        return self.__string_representation._repr_html_()

    @property
    def _strings(self) -> list[MultiLineString]:
        """The 3 multi-line strings used by __str__ and for logging.

        The second one can be logged with either an INFO or a WARNING level according to
        the feasibility of the solution.
        """
        strings = []
        msg = MultiLineString()
        msg.add("Optimization result:")
        msg.indent()
        msg.add("Optimizer info:")
        msg.indent()
        msg.add("Status: {}", self.status)
        msg.add("Message: {}", self.message)
        if self.n_obj_call is not None:
            msg.add(
                "Number of calls to the objective function by the optimizer: {}",
                self.n_obj_call,
            )
        msg.dedent()
        msg.add("Solution:")
        msg.indent()
        strings.append(msg)
        msg = MultiLineString()
        if self.constraint_values:
            not_ = "" if self.is_feasible else "not "
            msg.indent()
            msg.indent()
            msg.add("The solution is {}feasible.", not_)
        strings.append(msg)

        msg = MultiLineString()
        msg.indent()
        msg.indent()
        msg.add("Objective: {}", self.f_opt)
        if self.constraint_values and len(self.constraint_values) < 20:
            msg.add("Standardized constraints:")
            msg.indent()
            for name, value in sorted(self.constraint_values.items()):
                msg.add("{} = {}", name, value)
        strings.append(msg)
        return strings

    def __str__(self) -> str:
        return str(self._strings[0] + self._strings[1] + self._strings[2])

    def to_dict(self) -> dict[str, Value]:
        """Convert the optimization result to a dictionary.

        The keys are the names of the optimization result fields,
        except for the constraint values and gradients.
        The key ``"constr:y"`` maps to ``result.constraint_values["y"]``
        while ``"constr_grad:y"`` maps to ``result.constraints_grad["y"]``.

        Returns:
            A dictionary representation of the optimization result.
        """
        dict_ = {
            k: v for k, v in self.__dict__.items() if k not in self.__NOT_DICT_KEYS
        }
        for mapping, prefix in [
            (self.constraint_values, self.__C_TAG),
            (self.constraints_grad, self.__CGRAD_TAG),
        ]:
            if mapping is not None:
                for key, value in mapping.items():
                    dict_[f"{prefix}{key}"] = value

        return dict_

    @classmethod
    def from_dict(cls, dict_: Mapping[str, Value]) -> OptimizationResult:
        """Create an optimization result from a dictionary.

        Args:
            dict_: The dictionary representation of the optimization result.
                The keys are the names of the optimization result fields,
                except for the constraint values and gradients.
                The value associated with the key ``"constr:y"``
                will be stored in ``result.constraint_values["y"]``
                while the value associated with the key ``"constr_grad:y"``
                will be stored in ``result.constraints_grad["y"]``.

        Returns:
            An optimization result.
        """
        cstr = {}
        cstr_grad = {}
        for key, value in dict_.items():
            if key.startswith(cls.__C_TAG):
                cstr[key[cls.__C_TAG_LEN :]] = value

            if key.startswith(cls.__CGRAD_TAG):
                cstr_grad[key[cls.__CGRAD_TAG_LEN :]] = value

        optimization_result = {
            key.name: dict_[key.name] for key in fields(cls) if key.name in dict_
        }
        optimization_result.update({
            cls.__CONSTRAINTS_VALUES: cstr or None,
            cls.__CONSTRAINTS_GRAD: cstr_grad or None,
        })
        return cls(**optimization_result)

    @classmethod
    def from_optimization_problem(
        cls, problem: OptimizationProblem, **fields_
    ) -> OptimizationResult:
        """Create an optimization result from an optimization problem.

        Args:
            problem: The optimization problem.
            **fields_: The fields of the :class:`.OptimizationResult`
                that cannot be deduced from the optimization problem;
                e.g. ``"optimizer_name"``.

        Returns:
            The optimization result associated with the optimization problem.
        """
        if not problem.database:
            return cls(n_obj_call=0, **fields_)

        x_0 = problem.database.get_x_vect(1)
        # compute the best feasible or infeasible point
        f_opt, x_opt, is_feas, c_opt, c_opt_grad = problem.get_optimum()
        if (
            f_opt is not None
            and not problem.minimize_objective
            and not problem.use_standardized_objective
        ):
            f_opt = -f_opt
            objective_name = problem.objective.original_name
        else:
            objective_name = problem.objective.name

        if x_opt is None:
            optimum_index = None
        else:
            optimum_index = problem.database.get_iteration(x_opt) - 1

        fields_["objective_name"] = objective_name
        return cls(
            x_0=x_0,
            x_0_as_dict=problem.design_space.array_to_dict(x_0),
            x_opt=x_opt,
            x_opt_as_dict=problem.design_space.array_to_dict(x_opt),
            f_opt=f_opt,
            n_obj_call=problem.objective.n_calls,
            is_feasible=is_feas,
            constraint_values=c_opt,
            constraints_grad=c_opt_grad,
            optimum_index=optimum_index,
            **fields_,
        )
