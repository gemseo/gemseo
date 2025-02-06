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
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#       :author: Damien Guenot
#       :author: Francois Gallard, Charlie Vanaret, Benoit Pauwels
#       :author: Gabriel Max De Mendonça Abrantes
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""Optimization problem.

The :class:`.OptimizationProblem` class operates on a :class:`.DesignSpace` defining:

- an initial guess :math:`x_0` for the design variables,
- the bounds :math:`l_b \leq x \leq u_b` of the design variables.

A (possible vector) objective function with an :class:`.MDOFunction` type
is set using the ``objective`` attribute.
If the optimization problem looks for the maximum of this objective function,
the :meth:`.OptimizationProblem.minimize_objective` property
changes the objective function sign
because the optimization drivers seek to minimize this objective function.

Equality and inequality constraints are also :class:`.MDOFunction` instances
provided to the :class:`.OptimizationProblem`
by means of its :meth:`.OptimizationProblem.add_constraint` method.

The :class:`.OptimizationProblem` allows to evaluate the different functions
for a given design parameters vector
(see :meth:`.OptimizationProblem.evaluate_functions`).
Note that this evaluation step relies on an automated scaling of function wrt the bounds
so that optimizers and DOE algorithms work
with inputs scaled between 0 and 1 for all the variables.

The :class:`.OptimizationProblem`  has also a :class:`.Database`
that stores the calls to all the functions
so that no function is called twice with the same inputs.
Concerning the derivatives' computation,
the :class:`.OptimizationProblem` automates
the generation of the finite differences or complex step wrappers on functions,
when the analytical gradient is not available.

Lastly,
various getters and setters are available,
as well as methods to export the :class:`.Database`
to an HDF file or to a :class:`.Dataset` for future post-processing.
"""

from __future__ import annotations

import contextlib
import logging
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final
from typing import Literal
from typing import Optional
from typing import overload

import h5py
from numpy import atleast_1d
from numpy import bytes_
from numpy import eye as np_eye
from numpy import hstack
from numpy import ndarray
from numpy import where
from numpy import zeros
from strenum import StrEnum

from gemseo.algos.aggregation.aggregation_func import aggregate_positive_sum_square
from gemseo.algos.aggregation.aggregation_func import aggregate_sum_square
from gemseo.algos.constraint_tolerances import ConstraintTolerances
from gemseo.algos.database import Database
from gemseo.algos.evaluation_problem import EvaluationProblem
from gemseo.algos.multiobjective_optimization_result import (
    MultiObjectiveOptimizationResult,
)
from gemseo.algos.optimization_history import OptimizationHistory
from gemseo.algos.optimization_result import OptimizationResult
from gemseo.algos.pareto.pareto_front import ParetoFront
from gemseo.core.mdo_functions.collections.constraints import Constraints
from gemseo.core.mdo_functions.linear_composite_function import LinearCompositeFunction
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.core.mdo_functions.mdo_linear_function import MDOLinearFunction
from gemseo.datasets.dataset import Dataset
from gemseo.datasets.io_dataset import IODataset
from gemseo.datasets.optimization_dataset import OptimizationDataset
from gemseo.typing import RealArray
from gemseo.utils.hdf5 import convert_h5_group_to_dict
from gemseo.utils.hdf5 import get_hdf5_group
from gemseo.utils.hdf5 import store_attr_h5data
from gemseo.utils.hdf5 import store_h5data
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from gemseo.algos.design_space import DesignSpace


LOGGER = logging.getLogger(__name__)

BestInfeasiblePointType = tuple[
    Optional[RealArray], Optional[RealArray], bool, dict[str, RealArray]
]


class OptimizationProblem(EvaluationProblem):
    """An optimization problem."""

    __constraints: Constraints
    """The constraints."""

    __tolerances: ConstraintTolerances
    """The constraint tolerances."""

    __is_linear: bool
    """Whether the optimization problem is linear."""

    __minimize_objective: bool
    """Whether to minimize the objective."""

    _objective: MDOFunction | None
    """The objective if set."""

    solution: OptimizationResult | None
    """The solution of the optimization problem if solved; otherwise ``None``."""

    use_standardized_objective: bool
    """Whether to use standardized objective for logging and post-processing.

    The standardized objective corresponds to the original one expressed as a cost
    function to minimize. A :class:`.BaseDriverLibrary` works with this standardized
    objective and the :class:`.Database` stores its values. However, for convenience, it
    may be more relevant to log the expression and the values of the original objective.
    """

    # Enumerations
    AggregationFunction = Constraints.AggregationFunction
    DifferentiationMethod = EvaluationProblem.DifferentiationMethod

    class HistoryFileFormat(StrEnum):
        """The format of the history file."""

        HDF5 = "hdf5"
        GGOBI = "ggobi"

    # HDF5 group names
    _CONSTRAINTS_GROUP: Final[str] = "constraints"
    _OBJECTIVE_GROUP: Final[str] = "objective"
    _OBSERVABLES_GROUP: Final[str] = "observables"
    _OPT_DESCR_GROUP: Final[str] = "opt_description"
    _SOLUTION_GROUP: Final[str] = "solution"
    _OPTIM_DESCRIPTION: ClassVar[str] = [
        "minimize_objective",
        "differentiation_step",
        "differentiation_method",
        "is_linear",
        "ineq_tolerance",
        "eq_tolerance",
    ]
    _SLACK_VARIABLE: Final[str] = "slack_variable_{}"

    def __init__(
        self,
        design_space: DesignSpace,
        is_linear: bool = True,
        database: Database | None = None,
        differentiation_method: DifferentiationMethod = DifferentiationMethod.USER_GRAD,
        differentiation_step: float = 1e-7,
        parallel_differentiation: bool = False,
        use_standardized_objective: bool = True,
        **parallel_differentiation_options: int | bool,
    ) -> None:
        """
        Args:
            pb_type: The type of the optimization problem.
            use_standardized_objective: Whether to use standardized objective
                for logging and post-processing.
        """  # noqa: D205, D212, D415
        self.__tolerances = ConstraintTolerances()
        self._objective = None
        self.__constraints = Constraints(design_space, self.tolerances)
        self.__minimize_objective = True
        self.use_standardized_objective = use_standardized_objective
        self.solution = None
        super().__init__(
            design_space,
            database=database,
            differentiation_method=differentiation_method,
            differentiation_step=differentiation_step,
            parallel_differentiation=parallel_differentiation,
            **parallel_differentiation_options,
        )
        self._sequence_of_functions = [self.__constraints, *self._sequence_of_functions]
        self._function_names = ["_objective"]
        self.history = OptimizationHistory(
            self.constraints, self.database, self.design_space
        )
        self.__is_linear = is_linear

    @property
    def is_linear(self) -> bool:
        """Whether the optimization problem is linear."""
        if self.constraints.aggregated_constraint_indices:
            self.__is_linear = False

        return self.__is_linear

    @property
    def tolerances(self) -> ConstraintTolerances:
        """The constraint tolerances."""
        return self.__tolerances

    @property
    def constraints(self) -> Constraints:
        """The constraints."""
        return self.__constraints

    @constraints.setter
    def constraints(self, functions: Iterable[MDOFunction]) -> None:
        self.__constraints.clear()
        self.__constraints.extend(functions)

    @property
    def objective(self) -> MDOFunction:
        """The objective function."""
        return self._objective

    @objective.setter
    def objective(self, function: MDOFunction) -> None:
        if self.is_linear and not isinstance(function, MDOLinearFunction):
            self.__is_linear = False

        function.f_type = function.FunctionType.OBJ
        self._objective = function
        self.history.objective_name = function.name

    @property
    def minimize_objective(self) -> bool:
        """Whether to minimize the objective."""
        return self.__minimize_objective

    @minimize_objective.setter
    def minimize_objective(self, value: bool) -> None:
        if self.__minimize_objective != value:
            self.__minimize_objective = not self.__minimize_objective
            self.objective = -self.objective

    def add_constraint(
        self,
        function: MDOFunction,
        value: float = 0.0,
        constraint_type: MDOFunction.ConstraintType | None = None,
        positive: bool = False,
    ) -> None:
        r"""Add an equality or inequality constraint to the optimization problem.

        An equality constraint is written as :math:`c(x)=a`,
        a positive inequality constraint is written as :math:`c(x)\geq a`
        and a negative inequality constraint is written as :math:`c(x)\leq a`.

        Args:
            function: The function :math:`c`.
            value: The value :math:`a`.
            constraint_type: The type of the constraint.
            positive: Whether the inequality constraint is positive.

        Raises:
            TypeError: When the constraint of a linear optimization problem
                is not an :class:`.MDOLinearFunction`.
            ValueError: When the type of the constraint is missing.
        """
        if self.is_linear and not isinstance(function, MDOLinearFunction):
            self.__is_linear = False

        formatted_constraint = self.__constraints.format(
            function, value=value, constraint_type=constraint_type, positive=positive
        )
        self.__constraints.append(formatted_constraint)

    def apply_exterior_penalty(
        self,
        objective_scale: float = 1.0,
        scale_inequality: float | RealArray = 1.0,
        scale_equality: float | RealArray = 1.0,
    ) -> None:
        r"""Reformulate the optimization problem using exterior penalty.

        Given the optimization problem with equality and inequality constraints:

        .. math::

            min_x f(x)

            s.t.

            g(x)\leq 0

            h(x)=0

            l_b\leq x\leq u_b

        The exterior penalty approach consists in building a penalized objective
        function that takes into account constraints violations:

        .. math::

            min_x \tilde{f}(x) = \frac{f(x)}{o_s} + s[\sum{H(g(x))g(x)^2}+\sum{h(x)^2}]

            s.t.

            l_b\leq x\leq u_b

        Where :math:`H(x)` is the Heaviside function,
        :math:`o_s` is the ``objective_scale``
        parameter and :math:`s` is the scale parameter.
        The solution of the new problem approximate the one of the original problem.
        Increasing the values of ``objective_scale`` and scale,
        the solutions are closer but
        the optimization problem requires more and more iterations to be solved.

        Args:
            scale_equality: The equality constraint scaling constant.
            objective_scale: The objective scaling constant.
            scale_inequality: The inequality constraint scaling constant.
        """
        self.__is_linear = False
        penalized_objective = self._objective / objective_scale
        self.add_observable(self._objective)
        for constraint in self.__constraints:
            if constraint.f_type == MDOFunction.ConstraintType.INEQ:
                penalized_objective += aggregate_positive_sum_square(
                    constraint, scale=scale_inequality
                )
            else:
                penalized_objective += aggregate_sum_square(
                    constraint, scale=scale_equality
                )
            self.add_observable(constraint)

        self.objective = penalized_objective
        self.__constraints.clear()

    def get_reformulated_problem_with_slack_variables(self) -> OptimizationProblem:
        r"""Add slack variables and replace inequality constraints with equality ones.

        Given the original optimization problem,

        .. math::

            min_x f(x)

            s.t.

            g(x)\leq 0

            h(x)=0

            l_b\leq x\leq u_b

        Slack variables are introduced for all inequality constraints that are
        non-positive. An equality constraint for each slack variable is then defined.

        .. math::

            min_{x,s} F(x,s) = f(x)

            s.t.

            H(x,s) = h(x)=0

            G(x,s) = g(x)-s=0

            l_b\leq x\leq u_b

            s\leq 0

        Returns:
            An optimization problem without inequality constraints.
        """
        # Copy the original design space
        problem = OptimizationProblem(deepcopy(self.design_space))

        # Evaluate the MDOFunctions.
        self.evaluate_functions()

        # Add a slack variable to the copied design space for each
        # inequality constraint.
        for inequality_constraint in self.constraints.get_inequality_constraints():
            problem.design_space.add_variable(
                name=self._SLACK_VARIABLE.format(inequality_constraint.name),
                size=inequality_constraint.dim,
                value=0,
                upper_bound=0,
            )

        # Compute a restriction operator that goes from the new design space to the old
        # design space variables.
        dimension = self.design_space.dimension
        restriction_operator = hstack((
            np_eye(dimension),
            zeros((dimension, problem.design_space.dimension - dimension)),
        ))
        # Get the new problem objective function composing the initial objective
        # function with the restriction operator.
        problem.objective = LinearCompositeFunction(
            self._objective, restriction_operator
        )

        # Each constraint is passed to the new problem. Each inequality constraints is
        # modified first composing the initial constraint with the restriction operator
        # then subtracting s. Where s is the constraint slack variable previously built.
        # Each equality constraint is added composing the initial constraint with the
        # restriction operator.
        for constraint in self.__constraints:
            new_function = LinearCompositeFunction(constraint, restriction_operator)
            new_function.f_type = MDOFunction.ConstraintType.EQ
            if constraint.f_type == MDOFunction.ConstraintType.EQ:
                problem.add_constraint(new_function)
                continue

            coefficients = where(
                [
                    i
                    in problem.design_space.get_variables_indexes(
                        self._SLACK_VARIABLE.format(constraint.name)
                    )
                    for i in range(problem.design_space.dimension)
                ],
                -1,
                0,
            )
            correction_term = MDOLinearFunction(
                coefficients=coefficients,
                name=f"offset_{constraint.name}",
                input_names=problem.design_space.get_indexed_variable_names(),
            )
            problem.add_constraint(new_function + correction_term)

        return problem

    @property
    def functions(self) -> list[MDOFunction]:  # noqa: D102
        return [self._objective, *self.__constraints, *super().functions]

    @property
    def original_functions(self) -> list[MDOFunction]:  # noqa: D102
        return [
            self.objective.original,
            *self.__constraints.get_originals(),
            *super().original_functions,
        ]

    @property
    def standardized_objective_name(self) -> str:
        """The name of the standardized objective.

        Given an objective named ``"f"``,
        the name of the standardized objective is ``"f"`` in the case of minimization
        and "-f" in the case of maximization.
        """
        return self._objective.name

    @property
    def objective_name(self) -> str:
        """The name of the objective."""
        if self.minimize_objective:
            return self._objective.name

        return self._objective.name[1:]

    def get_function_names(self, names: Iterable[str]) -> list[str]:
        """Return the names of the functions stored in the database.

        Args:
            names: The names of the outputs or constraints specified by the user.

        Returns:
            The names of the constraints stored in the database.
        """
        original_to_current_names = self.constraints.original_to_current_names
        function_names = []
        for name in names:
            if name in original_to_current_names:
                function_names.extend(original_to_current_names[name])
            else:
                function_names.append(name)

        return function_names

    def get_functions(
        self,
        no_db_no_norm: bool = False,
        observable_names: Iterable[str] | None = (),
        jacobian_names: Iterable[str] | None = None,
        evaluate_objective: bool = True,
        constraint_names: Iterable[str] | None = (),
    ) -> tuple[list[MDOFunction], list[MDOFunction]]:
        """
        Args:
            evaluate_objective: Whether to evaluate the objective.
            constraint_names: The names of the constraints to evaluate.
                If empty,
                then all the constraints are returned.
                If ``None``,
                then no constraint is returned.
        """  # noqa: D205, D212
        return super().get_functions(
            no_db_no_norm=no_db_no_norm,
            observable_names=observable_names,
            jacobian_names=jacobian_names,
            return_objective=evaluate_objective,
            constraint_names=constraint_names,
        )

    def _get_options_for_get_functions(
        self, jacobian_names: list[str]
    ) -> dict[str, bool | list[str]]:
        return {
            "return_objective": self._objective.name in jacobian_names,
            "constraint_names": [
                name
                for name in jacobian_names
                if name in self.constraints.original_to_current_names
            ],
        }

    def _get_functions(
        self,
        observable_names: Iterable[str] | None,
        from_original_functions: bool,
        return_objective: bool,
        constraint_names: Iterable[str] | None,
    ) -> list[MDOFunction]:
        """
        Args:
            return_objective: Whether to return the objective function.
            constraint_names: The names of the constraints to return.
                If empty,
                then all the constraints are returned.
                If ``None``,
                then no constraint is returned.
        """  # noqa: D205, D212
        if not return_objective:
            functions = []
        elif from_original_functions:
            functions = [self.objective.original]
        else:
            functions = [self._objective]

        if constraint_names:
            for name in constraint_names:
                functions.append(
                    self.constraints.get_from_name(name, from_original_functions)
                )
        elif constraint_names is not None and from_original_functions:
            functions += self.__constraints.get_originals()
        elif constraint_names is not None:
            functions += self.__constraints

        functions.extend(
            super()._get_functions(observable_names, from_original_functions)
        )
        return functions

    # TODO: only one public use in DOELibrary and it seems to be a duplicate.
    def check(self) -> None:
        """Check if the optimization problem is ready for run.

        Raises:
            ValueError: If the objective function is missing.
        """
        if self._objective is None:
            msg = "Missing objective function in OptimizationProblem"
            raise ValueError(msg)
        super().check()

    @property
    def optimum(self) -> OptimizationHistory.Solution:
        """The optimum solution within a given feasibility tolerance.

        This solution is defined by:

        - the value of the objective function,
        - the value of the design variables,
        - the indicator of feasibility of the optimal solution,
        - the value of the constraints,
        - the value of the gradients of the constraints.
        """
        return self.history.optimum

    def _get_string_representation(self) -> MultiLineString:
        mls = MultiLineString()
        mls.add("Optimization problem:")
        mls.indent()

        # objective representation
        if self.minimize_objective or self.use_standardized_objective:
            optimize_verb = "minimize "
            start = 0
        else:
            optimize_verb = "maximize "
            start = 1

        objective_function = [
            line for line in repr(self._objective).split("\n") if line
        ]
        mls.add(optimize_verb + objective_function[0][start:])
        for line in objective_function[1:]:
            mls.add(" " * len(optimize_verb) + line)

        # variables representation
        mls.add("with respect to {}", pretty_str(self.design_space))
        if self.__constraints:
            mls.add("subject to constraints:")
            mls.indent()
            for functions in [
                self.constraints.get_inequality_constraints(),
                self.constraints.get_equality_constraints(),
            ]:
                for constraint in functions:
                    constraint = [c_i for c_i in str(constraint).split("\n") if c_i]
                    for constraint_i in constraint:
                        mls.add(constraint_i)

        return mls

    def to_hdf(
        self,
        file_path: str | Path,
        append: bool = False,
        hdf_node_path: str = "",
    ) -> None:
        """Export the optimization problem to an HDF file.

        Args:
            file_path: The HDF file path.
            append: Whether to append the data to the file if not empty.
                Otherwise,
                overwrite data.
            hdf_node_path: The path of the HDF node
                in which the optimization problem should be exported.
                If empty, the root node is considered.
        """
        msg = "Exporting the optimization problem to the file %s"
        if hdf_node_path:
            msg += " at node %s"
            LOGGER.info(msg, file_path, hdf_node_path)
        else:
            LOGGER.info(msg, file_path)

        with h5py.File(file_path, "a" if append else "w") as h5file:
            if hdf_node_path:
                h5file = h5file.require_group(hdf_node_path)

            if not append or self._OPT_DESCR_GROUP not in h5file:
                opt_group = h5file.require_group(self._OPT_DESCR_GROUP)
                for attr_name in self._OPTIM_DESCRIPTION:
                    if attr_name == "ineq_tolerance":
                        attr = self.tolerances.inequality
                    elif attr_name == "eq_tolerance":
                        attr = self.tolerances.equality
                    elif attr_name == "fd_step":
                        attr = self.differentiation_step
                    else:
                        attr = getattr(self, attr_name)

                    store_h5data(opt_group, attr, attr_name)

                store_attr_h5data(
                    self._objective, h5file.require_group(self._OBJECTIVE_GROUP)
                )

                for functions, group in zip(
                    [self.__constraints, self.observables],
                    [self._CONSTRAINTS_GROUP, self._OBSERVABLES_GROUP],
                ):
                    if functions:
                        function_group = h5file.require_group(group)
                        for function in functions:
                            store_attr_h5data(
                                function, function_group.require_group(function.name)
                            )

                if self.solution is not None:
                    sol_group = h5file.require_group(self._SOLUTION_GROUP)
                    store_attr_h5data(self.solution, sol_group)

        self.database.to_hdf(file_path, append=True, hdf_node_path=hdf_node_path)

    @classmethod
    def from_hdf(
        cls,
        file_path: str | Path,
        x_tolerance: float = 0.0,
        hdf_node_path: str = "",
    ) -> OptimizationProblem:
        """Import an optimization history from an HDF file.

        Args:
            file_path: The file containing the optimization history.
            x_tolerance: The tolerance on the design variables when reading the file.
            hdf_node_path: The path of the HDF node from which
                the database should be imported.
                If empty, the root node is considered.

        Returns:
            The read optimization problem.
        """
        msg = "Importing the optimization problem from the file %s"
        if hdf_node_path:
            msg += " at node %s"
            LOGGER.info(msg, file_path, hdf_node_path)
        else:
            LOGGER.info(msg, file_path)

        database = Database.from_hdf(file_path, hdf_node_path=hdf_node_path, log=False)
        design_space = database.input_space
        problem = OptimizationProblem(design_space, database=database)

        with h5py.File(file_path) as h5file:
            h5file = get_hdf5_group(h5file, hdf_node_path)
            if problem._SOLUTION_GROUP in h5file:
                solution_data = convert_h5_group_to_dict(
                    h5file, problem._SOLUTION_GROUP
                )
                for name in ["x_0_as_dict", "x_opt_as_dict"]:
                    if name in h5file:
                        solution_data[name] = convert_h5_group_to_dict(h5file, name)

                problem.solution = OptimizationResult.from_dict(solution_data)

            objective = MDOFunction.init_from_dict_repr(
                **convert_h5_group_to_dict(h5file, problem._OBJECTIVE_GROUP)
            )

            # The generated functions can be called at the x stored in
            # the database
            objective.set_pt_from_database(
                problem.database, design_space, x_tolerance=x_tolerance
            )
            problem.objective = objective

            group = get_hdf5_group(h5file, problem._OPT_DESCR_GROUP)
            for attr_name, attr in group.items():
                val = attr[()]
                if isinstance(val, bytes):
                    val = val.decode()

                if isinstance(val, ndarray) and isinstance(val[0], bytes_):
                    val = val[0].decode()

                if attr_name == "minimize_objective":
                    attr_name = "_OptimizationProblem__minimize_objective"

                if attr_name == "is_linear":
                    attr_name = "_OptimizationProblem__is_linear"

                if attr_name == "pb_type":
                    attr_name = "_OptimizationProblem__is_linear"
                    val = val == "linear"

                setattr(problem, attr_name, val)

            for name, functions in zip(
                [problem._CONSTRAINTS_GROUP, problem._OBSERVABLES_GROUP],
                [problem.constraints, problem.observables],
            ):
                if name in h5file:
                    group = get_hdf5_group(h5file, name)
                    for function_name in group:
                        functions.append(
                            MDOFunction.init_from_dict_repr(
                                **convert_h5_group_to_dict(group, function_name)
                            )
                        )

            is_mono_objective = False
            with contextlib.suppress(ValueError):
                # Sometimes the dimension of the problem cannot be determined.
                is_mono_objective = problem.is_mono_objective

            if not is_mono_objective and problem._SOLUTION_GROUP in h5file:
                pareto_front = (
                    ParetoFront.from_optimization_problem(problem)
                    if problem.solution.is_feasible
                    else None
                )
                problem.solution = MultiObjectiveOptimizationResult(
                    **problem.solution.__dict__, pareto_front=pareto_front
                )

        return problem

    @overload
    def to_dataset(
        self,
        name: str = "",
        categorize: Literal[True] = ...,
        export_gradients: bool = ...,
        input_values: Iterable[RealArray] = ...,
        opt_naming: Literal[False] = ...,
    ) -> IODataset: ...

    @overload
    def to_dataset(
        self,
        name: str = ...,
        categorize: Literal[True] = ...,
        export_gradients: bool = ...,
        input_values: Iterable[RealArray] = ...,
        opt_naming: Literal[True] = ...,
    ) -> OptimizationDataset: ...

    def to_dataset(
        self,
        name: str = "",
        categorize: bool = True,
        export_gradients: bool = False,
        input_values: Iterable[RealArray] = (),
        opt_naming: bool = True,
    ) -> Dataset:
        """
        Args:
            categorize: Whether to distinguish
                between the different groups of variables.
            opt_naming: Whether to
                put the design variables
                in the :attr:`.OptimizationDataset.DESIGN_GROUP`
                and the functions and their derivatives in the
                :attr:`.OptimizationDataset.FUNCTION_GROUP`.
                Otherwise,
                put the design variables in the :attr:`.IODataset.INPUT_GROUP`
                and the functions and their derivatives in the
                :attr:`.IODataset.OUTPUT_GROUP`.
        """  # noqa: D205, D212
        if categorize:
            gradient_group = Dataset.GRADIENT_GROUP
            if opt_naming:
                dataset_class = OptimizationDataset
                input_group = OptimizationDataset.DESIGN_GROUP
                output_group = OptimizationDataset.FUNCTION_GROUP
            else:
                dataset_class = IODataset
                input_group = IODataset.INPUT_GROUP
                output_group = IODataset.OUTPUT_GROUP
        else:
            dataset_class = Dataset
            input_group = output_group = gradient_group = Dataset.DEFAULT_GROUP

        return self.database.to_dataset(
            name=name,
            export_gradients=export_gradients,
            input_values=input_values,
            dataset_class=dataset_class,
            input_group=input_group,
            output_group=output_group,
            gradient_group=gradient_group,
        )

    @property
    def is_mono_objective(self) -> bool:
        """Whether the optimization problem is mono-objective.

        Raises:
            ValueError: When the dimension of the objective cannot be determined.
        """
        dimension = self._objective.dim
        if dimension != 0:
            return dimension == 1

        n_output_names = len(self._objective.output_names)
        if n_output_names == 0:
            msg = "Cannot determine the dimension of the objective."
            raise ValueError(msg)

        return n_output_names == 1

    def get_functions_dimensions(
        self, names: Iterable[str] | None = None
    ) -> dict[str, int]:
        """Return the dimensions of the outputs of the problem functions.

        Args:
            names: The names of the functions.
                If ``None``, then the objective and all the constraints are considered.

        Returns:
            The output dimensions of the functions associated with their names.
        """
        if names is None:
            names = [self._objective.name, *self.constraints.get_names()]

        return {name: self.get_function_dimension(name) for name in names}

    def get_function_dimension(self, name: str) -> int:
        """Return the dimension of a function of the problem (e.g. a constraint).

        Args:
            name: The name of the function.

        Returns:
            The dimension of the function.

        Raises:
            ValueError: If the function name is unknown to the problem.
            RuntimeError: If the function dimension is not available.
        """
        # Check that the required function belongs to the problem and get it
        for function in self.functions:
            if function.name == name:
                break
        else:
            msg = f"The problem has no function named {name}."
            raise ValueError(msg)

        # Get the dimension of the function output
        if function.dim:
            return function.dim

        if self.design_space.has_current_value:
            get_current_value = self.design_space.get_current_value
            if function.expects_normalized_inputs:
                current_variables = get_current_value(normalize=True)
            else:
                current_variables = get_current_value()

            return atleast_1d(function.evaluate(current_variables)).size

        msg = f"The output dimension of function {name} is not available."
        raise RuntimeError(msg)

    @property
    def scalar_constraint_names(self) -> list[str]:
        """The names of the scalar constraints.

        A scalar constraint is a constraint whose output is of dimension 1.
        """
        constraint_names = []
        for constraint in self.__constraints:
            dimension = self.get_function_dimension(constraint.name)
            if dimension == 1:
                constraint_names.append(constraint.name)
            else:
                constraint_names.extend([
                    constraint.get_indexed_name(index) for index in range(dimension)
                ])
        return constraint_names

    def reset(  # noqa: D102
        self,
        database: bool = True,
        current_iter: bool = True,
        design_space: bool = True,
        function_calls: bool = True,
        preprocessing: bool = True,
    ) -> None:
        if preprocessing and self._functions_are_preprocessed:
            n_obj_calls = self._objective.n_calls
            n_constraint_calls = [c.n_calls for c in self.__constraints]
            self._objective = self._objective.original
            self.__constraints.reset()
            if not function_calls:
                self._objective.n_calls = n_obj_calls
                for constraint, n_calls in zip(self.__constraints, n_constraint_calls):
                    constraint.n_calls = n_calls

        super().reset(
            database=database,
            current_iter=current_iter,
            design_space=design_space,
            function_calls=function_calls,
            preprocessing=preprocessing,
        )
