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
the :meth:`.OptimizationProblem.change_objective_sign`
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

import logging
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from copy import deepcopy
from functools import reduce
from numbers import Number
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Final
from typing import Optional

import h5py
import numpy
from h5py import Group
from numpy import abs as np_abs
from numpy import all as np_all
from numpy import any as np_any
from numpy import argmin
from numpy import array
from numpy import array_equal
from numpy import atleast_1d
from numpy import bytes_
from numpy import eye as np_eye
from numpy import hstack
from numpy import inf
from numpy import insert
from numpy import isnan
from numpy import issubdtype
from numpy import multiply
from numpy import nan
from numpy import ndarray
from numpy import number as np_number
from numpy import where
from numpy import zeros
from numpy.linalg import norm
from pandas import MultiIndex
from strenum import StrEnum

from gemseo.algos.aggregation.aggregation_func import aggregate_iks
from gemseo.algos.aggregation.aggregation_func import aggregate_lower_bound_ks
from gemseo.algos.aggregation.aggregation_func import aggregate_max
from gemseo.algos.aggregation.aggregation_func import aggregate_positive_sum_square
from gemseo.algos.aggregation.aggregation_func import aggregate_sum_square
from gemseo.algos.aggregation.aggregation_func import aggregate_upper_bound_ks
from gemseo.algos.base_problem import BaseProblem
from gemseo.algos.database import Database
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.mdofunctions.dense_jacobian_function import DenseJacobianFunction
from gemseo.core.mdofunctions.func_operations import LinearComposition
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction
from gemseo.core.mdofunctions.mdo_quadratic_function import MDOQuadraticFunction
from gemseo.core.mdofunctions.norm_db_function import NormDBFunction
from gemseo.core.mdofunctions.norm_function import NormFunction
from gemseo.datasets.dataset import Dataset
from gemseo.datasets.io_dataset import IODataset
from gemseo.datasets.optimization_dataset import OptimizationDataset
from gemseo.disciplines.constraint_aggregation import ConstraintAggregation
from gemseo.utils.compatibility.scipy import sparse_classes
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from gemseo.utils.derivatives.gradient_approximator_factory import (
    GradientApproximatorFactory,
)
from gemseo.utils.enumeration import merge_enums
from gemseo.utils.hdf5 import get_hdf5_group
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

BestInfeasiblePointType = tuple[
    Optional[ndarray], Optional[ndarray], bool, dict[str, ndarray]
]
OptimumType = tuple[ndarray, ndarray, bool, dict[str, ndarray], dict[str, ndarray]]
OptimumSolutionType = tuple[
    Optional[Sequence[ndarray]], ndarray, dict[str, ndarray], dict[str, ndarray]
]


class OptimizationProblem(BaseProblem):
    """An optimization problem.

    Create an optimization problem from:

    - a :class:`.DesignSpace` specifying the design variables
      in terms of names, lower bounds, upper bounds and initial guesses,
    - the objective function as an :class:`.MDOFunction`,
      which can be a vector,

    execute it from an algorithm provided by a :class:`.DriverLibrary`,
    and store some execution data in a :class:`.Database`.

    In particular,
    this :class:`.Database` stores the calls to all the functions
    so that no function is called twice with the same inputs.

    An :class:`.OptimizationProblem` also has an automated scaling of function
    with respect to the bounds of the design variables
    so that the driving algorithms work with inputs scaled between 0 and 1.

    Lastly, :class:`.OptimizationProblem` automates the generation
    of finite differences or complex step wrappers on functions,
    when analytical gradient is not available.
    """

    current_iter: int
    """The current iteration."""

    max_iter: int
    """The maximum iteration."""

    nonproc_objective: MDOFunction
    """The non-processed objective function."""

    constraints: list[MDOFunction]
    """The constraints."""

    nonproc_constraints: list[MDOFunction]
    """The non-processed constraints."""

    observables: list[MDOFunction]
    """The observables."""

    new_iter_observables: list[MDOFunction]
    """The observables to be called at each new iterate."""

    nonproc_observables: list[MDOFunction]
    """The non-processed observables."""

    nonproc_new_iter_observables: list[MDOFunction]
    """The non-processed observables to be called at each new iterate."""

    __minimize_objective: bool
    """Whether to minimize the objective."""

    fd_step: float
    """The finite differences step."""

    pb_type: ProblemType
    """The type of optimization problem."""

    ineq_tolerance: float
    """The tolerance for the inequality constraints."""

    eq_tolerance: float
    """The tolerance for the equality constraints."""

    database: Database
    """The database to store the optimization problem data."""

    solution: OptimizationResult | None
    """The solution of the optimization problem if solved; otherwise ``None``."""

    design_space: DesignSpace
    """The design space on which the optimization problem is solved."""

    stop_if_nan: bool
    """Whether the optimization stops when a function returns ``NaN``."""

    preprocess_options: dict
    """The options to pre-process the functions."""

    use_standardized_objective: bool
    """Whether to use standardized objective for logging and post-processing.

    The standardized objective corresponds to the original one expressed as a cost
    function to minimize. A :class:`.DriverLibrary` works with this standardized
    objective and the :class:`.Database` stores its values. However, for convenience, it
    may be more relevant to log the expression and the values of the original objective.
    """

    AggregationFunction = ConstraintAggregation.EvaluationFunction

    _AGGREGATION_FUNCTION_MAP: Final[str] = {
        AggregationFunction.IKS: aggregate_iks,
        AggregationFunction.LOWER_BOUND_KS: aggregate_lower_bound_ks,
        AggregationFunction.UPPER_BOUND_KS: aggregate_upper_bound_ks,
        AggregationFunction.POS_SUM: aggregate_positive_sum_square,
        AggregationFunction.MAX: aggregate_max,
        AggregationFunction.SUM: aggregate_sum_square,
    }

    class ProblemType(StrEnum):
        """The type of problem."""

        LINEAR = "linear"
        NON_LINEAR = "non-linear"

    ApproximationMode = ApproximationMode

    class _DifferentiationMethod(StrEnum):
        """The additional differentiation methods."""

        USER_GRAD = "user"
        NO_DERIVATIVE = "no_derivative"

    DifferentiationMethod = merge_enums(
        "DifferentiationMethod",
        StrEnum,
        ApproximationMode,
        _DifferentiationMethod,
        doc="The differentiation methods.",
    )

    DESIGN_VAR_NAMES: Final[str] = "x_names"
    DESIGN_VAR_SIZE: Final[str] = "x_size"
    DESIGN_SPACE_ATTRS: Final[str] = [
        "u_bounds",
        "l_bounds",
        "x_0",
        DESIGN_VAR_NAMES,
        "dimension",
    ]
    FUNCTIONS_ATTRS: ClassVar[str] = ["objective", "constraints"]
    OPTIM_DESCRIPTION: ClassVar[str] = [
        "minimize_objective",
        "fd_step",
        "differentiation_method",
        "pb_type",
        "ineq_tolerance",
        "eq_tolerance",
    ]

    OPT_DESCR_GROUP: Final[str] = "opt_description"
    DESIGN_SPACE_GROUP: Final[str] = "design_space"
    OBJECTIVE_GROUP: Final[str] = "objective"
    SOLUTION_GROUP: Final[str] = "solution"
    CONSTRAINTS_GROUP: Final[str] = "constraints"
    OBSERVABLES_GROUP: Final[str] = "observables"

    activate_bound_check: ClassVar[bool] = True
    """Whether to check if a point is in the design space before calling functions."""

    HDF5_FORMAT: Final[str] = "hdf5"
    GGOBI_FORMAT: Final[str] = "ggobi"
    KKT_RESIDUAL_NORM: Final[str] = "KKT residual norm"

    def __init__(
        self,
        design_space: DesignSpace,
        pb_type: ProblemType = ProblemType.LINEAR,
        input_database: str | Path | Database | None = None,
        differentiation_method: DifferentiationMethod = DifferentiationMethod.USER_GRAD,
        fd_step: float = 1e-7,
        parallel_differentiation: bool = False,
        use_standardized_objective: bool = True,
        **parallel_differentiation_options: int | bool,
    ) -> None:
        """
        Args:
            design_space: The design space on which the functions are evaluated.
            pb_type: The type of the optimization problem.
            input_database: A database to initialize that of the optimization problem.
                If ``None``, the optimization problem starts from an empty database.
            differentiation_method: The default differentiation method to be applied
                to the functions of the optimization problem.
            fd_step: The step to be used by the step-based differentiation methods.
            parallel_differentiation: Whether to approximate the derivatives in
            parallel.
            use_standardized_objective: Whether to use standardized objective
                for logging and post-processing.
            **parallel_differentiation_options: The options
                to approximate the derivatives in parallel.
        """  # noqa: D205, D212, D415
        self._objective = None
        self.nonproc_objective = None
        self.constraints = []
        self.nonproc_constraints = []
        self.observables = []
        self.new_iter_observables = []
        self.nonproc_observables = []
        self.nonproc_new_iter_observables = []
        self.__minimize_objective = True
        self.fd_step = fd_step
        self.__differentiation_method = None
        self.differentiation_method = differentiation_method
        self.pb_type = pb_type
        self.ineq_tolerance = 1e-4
        self.eq_tolerance = 1e-2
        self.max_iter = 0
        self.current_iter = 0
        self.use_standardized_objective = use_standardized_objective
        self.__functions_are_preprocessed = False
        if input_database is None:
            self.database = Database()
        elif isinstance(input_database, Database):
            self.database = input_database
        else:
            self.database = Database.from_hdf(input_database)
        self.solution = None
        self.design_space = design_space
        self.__initial_current_x = deepcopy(
            design_space.get_current_value(as_dict=True)
        )
        self.__x0 = None
        self.stop_if_nan = True
        self.preprocess_options = {}
        self.__parallel_differentiation = parallel_differentiation
        self.__parallel_differentiation_options = parallel_differentiation_options
        self.__eval_obs_jac = False
        self.__observable_names = set()

    def __raise_exception_if_functions_are_already_preprocessed(self):
        """Raise an exception if the function have already been pre-processed."""
        if self.__functions_are_preprocessed:
            raise RuntimeError(
                "The parallel differentiation cannot be changed "
                "because the functions have already been pre-processed."
            )

    def is_max_iter_reached(self) -> bool:
        """Check if the maximum amount of iterations has been reached.

        Returns:
            Whether the maximum amount of iterations has been reached.
        """
        if self.max_iter in {None, 0} or self.current_iter in {None, 0}:
            return False
        return self.current_iter >= self.max_iter

    @property
    def constraint_names(self) -> dict[str, list[str]]:
        """The standardized constraint names bound to the original ones."""
        names = {}
        for constraint in self.constraints:
            if constraint.original_name not in names:
                names[constraint.original_name] = []
            names[constraint.original_name].append(constraint.name)
        return names

    @property
    def parallel_differentiation(self) -> bool:
        """Whether to approximate the derivatives in parallel."""
        return self.__parallel_differentiation

    @parallel_differentiation.setter
    def parallel_differentiation(
        self,
        value: bool,
    ) -> None:
        self.__raise_exception_if_functions_are_already_preprocessed()
        self.__parallel_differentiation = value

    @property
    def parallel_differentiation_options(self) -> dict[str, int | bool]:
        """The options to approximate the derivatives in parallel."""
        return self.__parallel_differentiation_options

    @parallel_differentiation_options.setter
    def parallel_differentiation_options(self, value: dict[str, int | bool]) -> None:
        self.__raise_exception_if_functions_are_already_preprocessed()
        self.__parallel_differentiation_options = value

    @property
    def objective(self) -> MDOFunction:
        """The objective function."""
        return self._objective

    @objective.setter
    def objective(
        self,
        func: MDOFunction,
    ) -> None:
        if self.pb_type == self.ProblemType.LINEAR and not isinstance(
            func, MDOLinearFunction
        ):
            self.pb_type = self.ProblemType.NON_LINEAR
        self._objective = func

    @property
    def minimize_objective(self) -> bool:
        """Whether to minimize the objective."""
        return self.__minimize_objective

    @minimize_objective.setter
    def minimize_objective(self, value: bool) -> None:
        if self.__minimize_objective != value:
            self.change_objective_sign()

    @staticmethod
    def repr_constraint(
        func: MDOFunction,
        cstr_type: MDOFunction.ConstraintType,
        value: float | None = None,
        positive: bool = False,
    ) -> str:
        """Express a constraint as a string expression.

        Args:
            func: The constraint function.
            cstr_type: The type of the constraint.
            value: The value for which the constraint is active.
                If ``None``, this value is 0.
            positive: If ``True``, then the inequality constraint is positive.

        Returns:
            A string representation of the constraint.
        """
        if value is None:
            value = 0.0
        str_repr = func.name
        if func.input_names:
            arguments = ", ".join(func.input_names)
            str_repr += f"({arguments})"

        if cstr_type == MDOFunction.ConstraintType.EQ:
            sign = " == "
        elif positive:
            sign = " >= "
        else:
            sign = " <= "

        if func.expr:
            str_repr += ": "
            expr = func.expr
            n_char = len(str_repr)
            # Remove empty lines with filter
            expr_spl = [_f for _f in expr.split("\n") if _f]
            str_repr = str_repr + expr_spl[0] + sign + str(value)
            if isinstance(func, (MDOLinearFunction, MDOQuadraticFunction)):
                for repre in expr_spl[1:]:
                    str_repr += "\n" + " " * n_char + repre
            else:
                for repre in expr_spl[1:]:
                    str_repr += "\n" + " " * n_char + repre + sign + str(value)
        else:
            str_repr += sign + str(value)
        return str_repr

    def add_constraint(
        self,
        cstr_func: MDOFunction,
        value: float | None = None,
        cstr_type: MDOFunction.ConstraintType | None = None,
        positive: bool = False,
    ) -> None:
        """Add a constraint (equality and inequality) to the optimization problem.

        Args:
            cstr_func: The constraint.
            value: The value for which the constraint is active.
                If ``None``, this value is 0.
            cstr_type: The type of the constraint.
            positive: If ``True``, then the inequality constraint is positive.

        Raises:
            TypeError: When the constraint of a linear optimization problem
                is not an :class:`.MDOLinearFunction`.
            ValueError: When the type of the constraint is missing.
        """
        func_name = cstr_func.name
        has_default_name = cstr_func.has_default_name
        self.check_format(cstr_func)
        if self.pb_type == OptimizationProblem.ProblemType.LINEAR and not isinstance(
            cstr_func, MDOLinearFunction
        ):
            self.pb_type = OptimizationProblem.ProblemType.NON_LINEAR
        ctype = cstr_type or cstr_func.f_type
        cstr_repr = self.repr_constraint(cstr_func, ctype, value, positive)
        if value is not None:
            cstr_func = cstr_func.offset(-value)
        if positive:
            cstr_func = -cstr_func

        if cstr_type is not None:
            cstr_func.f_type = cstr_type
        elif not cstr_func.is_constraint():
            msg = (
                "Constraint type must be provided, "
                "either in the function attributes or to the add_constraint method."
            )
            raise ValueError(msg)

        cstr_func.special_repr = cstr_repr
        self.constraints.append(cstr_func)
        if not has_default_name:
            cstr_func.name = func_name
            if cstr_func.output_names:
                output_names = "#".join(cstr_func.output_names)
                cstr_repr = cstr_repr.replace(func_name, output_names)
                cstr_func.expr = cstr_func.expr.replace(func_name, output_names)
                cstr_func.special_repr = f"{func_name}: {cstr_repr}"

    def add_eq_constraint(
        self,
        cstr_func: MDOFunction,
        value: float | None = None,
    ) -> None:
        """Add an equality constraint to the optimization problem.

        Args:
            cstr_func: The constraint.
            value: The value for which the constraint is active.
                If ``None``, this value is 0.
        """
        self.add_constraint(cstr_func, value, cstr_type=MDOFunction.ConstraintType.EQ)

    def add_ineq_constraint(
        self,
        cstr_func: MDOFunction,
        value: float | None = None,
        positive: bool = False,
    ) -> None:
        """Add an inequality constraint to the optimization problem.

        Args:
            cstr_func: The constraint.
            value: The value for which the constraint is active.
                If ``None``, this value is 0.
            positive: If ``True``, then the inequality constraint is positive.
        """
        self.add_constraint(
            cstr_func,
            value,
            cstr_type=MDOFunction.ConstraintType.INEQ,
            positive=positive,
        )

    def aggregate_constraint(
        self,
        constraint_index: int,
        method: Callable[[NDArray[float]], float]
        | AggregationFunction = AggregationFunction.MAX,
        groups: Iterable[Sequence[int]] | None = None,
        **options: Any,
    ) -> None:
        """Aggregate a constraint to generate a reduced dimension constraint.

        Args:
            constraint_index: The index of the constraint in :attr:`.constraints`.
            method: The aggregation method, e.g. ``"max"``, ``"lower_bound_KS"``,
                ``"upper_bound_KS"``or ``"IKS"``.
            groups: The groups of components of the constraint to aggregate
                to produce one aggregation constraint per group of components;
                if ``None``, a single aggregation constraint is produced.
            **options: The options of the aggregation method.

        Raises:
            ValueError: When the given index is greater or equal
                than the number of constraints
                or when the constraint aggregation method is unknown.
        """
        n_constraints = len(self.constraints)
        self.pb_type = OptimizationProblem.ProblemType.NON_LINEAR
        if constraint_index >= n_constraints:
            raise KeyError(
                f"The index of the constraint ({constraint_index}) must be lower "
                f"than the number of constraints ({n_constraints})."
            )

        constraint = self.constraints[constraint_index]
        if callable(method):
            aggregate_constraints = method
        else:
            aggregate_constraints = self._AGGREGATION_FUNCTION_MAP[method]

        del self.constraints[constraint_index]
        if groups is None:
            self.constraints.insert(
                constraint_index, aggregate_constraints(constraint, **options)
            )
        else:
            self.constraints[constraint_index:constraint_index] = [
                aggregate_constraints(constraint, indices, **options)
                for indices in groups
            ]

    def apply_exterior_penalty(
        self,
        objective_scale: float = 1.0,
        scale_inequality: float | ndarray = 1.0,
        scale_equality: float | ndarray = 1.0,
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
        self.pb_type = OptimizationProblem.ProblemType.NON_LINEAR
        penalized_objective = self.objective / objective_scale
        self.add_observable(self.objective)
        for constr in self.constraints:
            if constr.f_type == MDOFunction.ConstraintType.INEQ:
                penalized_objective += aggregate_positive_sum_square(
                    constr, scale=scale_inequality
                )
            else:
                penalized_objective += aggregate_sum_square(
                    constr, scale=scale_equality
                )
            self.add_observable(constr)
        self.objective = penalized_objective
        self.constraints = []

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
        for constr in self.get_ineq_constraints():
            problem.design_space.add_variable(
                name=f"slack_variable_{constr.name}",
                size=constr.dim,
                value=0,
                u_b=0,
            )

        # Compute a restriction operator that goes from the new design space to the old
        # design space variables.
        restriction_operator = hstack((
            np_eye(self.dimension),
            zeros((self.dimension, problem.dimension - self.dimension)),
        ))
        # Get the new problem objective function composing the initial objective
        # function with the restriction operator.
        problem.objective = LinearComposition(self.objective, restriction_operator)

        # Each constraint is passed to the new problem. Each inequality constraints is
        # modified first composing the initial constraint with the restriction operator
        # then subtracting s. Where s is the constraint slack variable previously built.
        # Each equality constraint is added composing the initial constraint with the
        # restriction operator.
        for constr in self.constraints:
            new_function = LinearComposition(constr, restriction_operator)
            if constr.f_type == MDOFunction.ConstraintType.EQ:
                problem.add_eq_constraint(new_function)
                continue

            coefficients = where(
                [
                    i
                    in problem.design_space.get_variables_indexes(
                        f"slack_variable_{constr.name}"
                    )
                    for i in range(problem.dimension)
                ],
                -1,
                0,
            )
            correction_term = MDOLinearFunction(
                coefficients=coefficients,
                name=f"offset_{constr.name}",
                input_names=problem.design_space.get_indexed_variable_names(),
            )
            problem.add_eq_constraint(new_function + correction_term)

        return problem

    def add_observable(
        self,
        obs_func: MDOFunction,
        new_iter: bool = True,
    ) -> None:
        """Add a function to be observed.

        When the :class:`.OptimizationProblem` is executed, the observables are called
        following this sequence:

            - The optimization algorithm calls the objective function with a normalized
              ``x_vect``.
            - The :meth:`.OptimizationProblem.preprocess_functions` wraps the function
              as a :class:`.NormDBFunction`, which unnormalizes the ``x_vect`` before
              evaluation.
            - The unnormalized ``x_vect`` and the result of the evaluation are stored in
              the :attr:`.OptimizationProblem.database`.
            - The previous step triggers the
              :attr:`.OptimizationProblem.new_iter_listeners`, which calls the
              observables with the unnormalized ``x_vect``.
            - The observables themselves are wrapped as a :class:`.NormDBFunction` by
              :meth:`.OptimizationProblem.preprocess_functions`, but in this case the
              input is always expected as unnormalized to avoid an additional
              normalizing-unnormalizing step.
            - Finally, the output is stored in the
              :attr:`.OptimizationProblem.database`.

        Args:
            obs_func: An observable to be observed.
            new_iter: If ``True``,
                then the observable will be called at each new iterate.
        """
        name = obs_func.name
        if name in self.__observable_names:
            LOGGER.warning('The optimization problem already observes "%s".', name)
            return

        self.check_format(obs_func)
        obs_func.f_type = MDOFunction.FunctionType.OBS
        self.observables.append(obs_func)
        self.__observable_names.add(name)
        if new_iter:
            self.new_iter_observables.append(obs_func)

    def get_eq_constraints(self) -> list[MDOFunction]:
        """Retrieve all the equality constraints.

        Returns:
            The equality constraints.
        """

        def is_equality_constraint(
            func: MDOFunction,
        ) -> bool:
            """Check if a function is an equality constraint.

            Args:
                func: A function.

            Returns:
                True if the function is an equality constraint.
            """
            return func.f_type == MDOFunction.ConstraintType.EQ

        return list(filter(is_equality_constraint, self.constraints))

    def get_ineq_constraints(self) -> list[MDOFunction]:
        """Retrieve all the inequality constraints.

        Returns:
            The inequality constraints.
        """

        def is_inequality_constraint(
            func: MDOFunction,
        ) -> bool:
            """Check if a function is an inequality constraint.

            Args:
                func: A function.

            Returns:
                True if the function is an inequality constraint.
            """
            return func.f_type == MDOFunction.ConstraintType.INEQ

        return list(filter(is_inequality_constraint, self.constraints))

    def get_observable(self, name: str) -> MDOFunction:
        """Return an observable of the problem.

        Args:
            name: The name of the observable.

        Returns:
            The pre-processed observable if the functions of the problem have already
            been pre-processed, otherwise the original one.
        """
        return self.__get_observable(name, not self.__functions_are_preprocessed)

    def __get_observable(
        self, name: str, from_original_observables: bool
    ) -> MDOFunction:
        """Return an observable of the problem.

        Args:
            name: The name of the observable.
            from_original_observables: Whether to get the observable from the original
                observables; otherwise return the observable from the pre-processed
                observables.

        Returns:
            The observable.
        """
        return self.__get_function(
            name,
            from_original_observables,
            self.__observable_names,
            self.nonproc_observables,
            self.observables,
            self.OBSERVABLES_GROUP,
        )

    def get_ineq_constraints_number(self) -> int:
        """Retrieve the number of inequality constraints.

        Returns:
            The number of inequality constraints.
        """
        return len(self.get_ineq_constraints())

    def get_eq_constraints_number(self) -> int:
        """Retrieve the number of equality constraints.

        Returns:
            The number of equality constraints.
        """
        return len(self.get_eq_constraints())

    def get_constraints_number(self) -> int:
        """Retrieve the number of constraints.

        Returns:
            The number of constraints.
        """
        return len(self.constraints)

    def get_constraint_names(self) -> list[str]:
        """Retrieve the names of the constraints.

        Returns:
            The names of the constraints.
        """
        return [constraint.name for constraint in self.constraints]

    def get_nonproc_constraints(self) -> list[MDOFunction]:
        """Retrieve the non-processed constraints.

        Returns:
            The non-processed constraints.
        """
        return self.nonproc_constraints

    def get_design_variable_names(self) -> list[str]:
        """Retrieve the names of the design variables.

        Returns:
            The names of the design variables.
        """
        return self.design_space.variable_names

    def get_all_functions(self) -> list[MDOFunction]:
        """Retrieve all the functions of the optimization problem.

        These functions are the constraints, the objective function and the observables.

        Returns:
            All the functions of the optimization problem.
        """
        return [self.objective, *self.constraints, *self.observables]

    def get_all_function_name(self) -> list[str]:
        """Retrieve the names of all the function of the optimization problem.

        These functions are the constraints, the objective function and the observables.

        Returns:
            The names of all the functions of the optimization problem.
        """
        return [func.name for func in self.get_all_functions()]

    def get_objective_name(self, standardize: bool = True) -> str:
        """Retrieve the name of the objective function.

        Args:
            standardize: Whether to use the name of the objective expressed as a cost,
                e.g. ``"-f"`` when the user seeks to maximize ``"f"``.

        Returns:
            The name of the objective function.
        """
        if standardize or self.minimize_objective:
            return self.objective.name

        return self.objective.name[1:]

    def get_function_names(self, names: Iterable[str]) -> list[str]:
        """Return the names of the functions stored in the database.

        Args:
            names: The names of the outputs or constraints specified by the user.

        Returns:
            The names of the constraints stored in the database.
        """
        user_constraint_names = self.constraint_names.keys()
        function_names = []
        for name in names:
            if name in user_constraint_names:
                function_names.extend(self.constraint_names[name])
            else:
                function_names.append(name)

        return function_names

    def get_nonproc_objective(self) -> MDOFunction:
        """Retrieve the non-processed objective function."""
        return self.nonproc_objective

    def has_nonlinear_constraints(self) -> bool:
        """Check if the problem has non-linear constraints.

        Returns:
            True if the problem has equality or inequality constraints.
        """
        return len(self.constraints) > 0

    def has_constraints(self) -> bool:
        """Check if the problem has equality or inequality constraints.

        Returns:
            True if the problem has equality or inequality constraints.
        """
        return self.has_eq_constraints() or self.has_ineq_constraints()

    def has_eq_constraints(self) -> bool:
        """Check if the problem has equality constraints.

        Returns:
            True if the problem has equality constraints.
        """
        return len(self.get_eq_constraints()) > 0

    def has_ineq_constraints(self) -> bool:
        """Check if the problem has inequality constraints.

        Returns:
            True if the problem has inequality constraints.
        """
        return len(self.get_ineq_constraints()) > 0

    def get_x0_normalized(
        self, cast_to_real: bool = False, as_dict: bool = False
    ) -> ndarray | dict[str, ndarray]:
        """Return the initial values of the design variables after normalization.

        Args:
            cast_to_real: Whether to return the real part of the initial values.
            as_dict: Whether to return the values
                as a dictionary of the form ``{variable_name: variable_value}``.


        Returns:
            The current values of the design variables
            normalized between 0 and 1 from their lower and upper bounds.
        """
        return self.design_space.get_current_value(None, cast_to_real, as_dict, True)

    def get_dimension(self) -> int:
        """Retrieve the total number of design variables.

        Returns:
            The dimension of the design space.
        """
        return self.design_space.dimension

    @property
    def dimension(self) -> int:
        """The dimension of the design space."""
        return self.design_space.dimension

    @staticmethod
    def check_format(input_function: Any) -> None:
        """Check that a function is an instance of :class:`.MDOFunction`.

        Args:
            input_function: The function to be tested.

        Raises:
            TypeError: If the function is not an :class:`.MDOFunction`.
        """
        if not isinstance(input_function, MDOFunction):
            raise TypeError(
                "Optimization problem functions must be instances of MDOFunction"
            )

    def get_eq_cstr_total_dim(self) -> int:
        """Retrieve the total dimension of the equality constraints.

        This dimension is the sum
        of all the outputs dimensions
        of all the equality constraints.

        Returns:
            The total dimension of the equality constraints.
        """
        return self.__count_cstr_total_dim(MDOFunction.ConstraintType.EQ)

    def get_ineq_cstr_total_dim(self) -> int:
        """Retrieve the total dimension of the inequality constraints.

        This dimension is the sum
        of all the outputs dimensions
        of all the inequality constraints.

        Returns:
            The total dimension of the inequality constraints.
        """
        return self.__count_cstr_total_dim(MDOFunction.ConstraintType.INEQ)

    def __count_cstr_total_dim(
        self,
        cstr_type: str,
    ) -> int:
        """Retrieve the total dimension of the constraints.

        This dimension is the sum
        of all the outputs dimensions
        of all the constraints.
        of equality or inequality constraints dimensions
        that is the sum of all outputs dimensions of all constraints.

        Returns:
            The total dimension of the constraints.
        """
        n_cstr = 0
        for constraint in self.constraints:
            if not constraint.dim:
                raise ValueError(
                    "Constraint dimension not available yet, "
                    f"please call function {constraint} once"
                )
            if constraint.f_type == cstr_type:
                n_cstr += constraint.dim
        return n_cstr

    def get_active_ineq_constraints(
        self,
        x_vect: ndarray,
        tol: float = 1e-6,
    ) -> dict[MDOFunction, ndarray]:
        """For each constraint, indicate if its different components are active.

        Args:
            x_vect: The vector of design variables.
            tol: The tolerance for deciding whether a constraint is active.

        Returns:
            For each constraint,
            a boolean indicator of activation of its different components.
        """
        self.design_space.check_membership(x_vect)
        if self.preprocess_options.get("is_function_input_normalized", False):
            x_vect = self.design_space.normalize_vect(x_vect)

        return {
            func: atleast_1d((func(x_vect)) >= -tol)
            for func in self.get_ineq_constraints()
        }

    def add_callback(
        self,
        callback_func: Callable,
        each_new_iter: bool = True,
        each_store: bool = False,
    ) -> None:
        """Add a callback function after each store operation or new iteration.

        Args:
            callback_func: A function to be called after some event.
            each_new_iter: If ``True``, then callback at every iteration.
            each_store: If ``True``,
                then callback at every call to :meth:`.Database.store`.
        """
        if each_store:
            self.database.add_store_listener(callback_func)
        if each_new_iter:
            self.database.add_new_iter_listener(callback_func)

    def clear_listeners(self) -> None:
        """Clear all the listeners."""
        self.database.clear_listeners()

    def evaluate_functions(
        self,
        x_vect: ndarray = None,
        eval_jac: bool = False,
        eval_obj: bool = True,
        eval_observables: bool = True,
        normalize: bool = True,
        no_db_no_norm: bool = False,
        constraint_names: Iterable[str] | None = None,
        observable_names: Iterable[str] | None = None,
        jacobian_names: Iterable[str] | None = None,
    ) -> tuple[dict[str, float | ndarray], dict[str, ndarray]]:
        """Compute the functions of interest, and possibly their derivatives.

        These functions of interest are the constraints, and possibly the objective.

        Some optimization libraries require the number of constraints
        as an input parameter which is unknown by the formulation or the scenario.
        Evaluation of initial point allows to get this mandatory information.
        This is also used for design of experiments to evaluate samples.

        Args:
            x_vect: The input vector at which the functions must be evaluated;
                if None, the initial point x_0 is used.
            eval_jac: Whether to compute the Jacobian matrices
                of the functions of interest.
                If ``True`` and ``jacobian_names`` is ``None`` then
                compute the Jacobian matrices (or gradients) of the functions that are
                selected for evaluation (with ``eval_obj``, ``constraint_names``,
                ``eval_observables`` and``observable_names``).
                If ``False`` and ``jacobian_names`` is ``None`` then no Jacobian
                matrix is evaluated.
                If ``jacobian_names`` is not ``None`` then the value of
                ``eval_jac`` is ignored.
            eval_obj: Whether to consider the objective function
                as a function of interest.
            eval_observables: Whether to evaluate the observables.
                If ``True`` and ``observable_names`` is ``None`` then all the
                observables are evaluated.
                If ``False`` and ``observable_names`` is ``None`` then no observable
                is evaluated.
                If ``observable_names`` is not ``None`` then the value of
                ``eval_observables`` is ignored.
            normalize: Whether to consider the input vector ``x_vect`` normalized.
            no_db_no_norm: If ``True``, then do not use the pre-processed functions,
                so we have no database, nor normalization.
            constraint_names: The names of the constraints to evaluate.
                If ``None`` then all the constraints are evaluated.
            observable_names: The names of the observables to evaluate.
                If ``None`` and ``eval_observables`` is ``True`` then all the
                observables are evaluated.
                If ``None`` and ``eval_observables`` is ``False`` then no observable is
                evaluated.
            jacobian_names: The names of the functions whose Jacobian matrices
                (or gradients) to compute.
                If ``None`` and ``eval_jac`` is ``True`` then
                compute the Jacobian matrices (or gradients) of the functions that are
                selected for evaluation (with ``eval_obj``, ``constraint_names``,
                ``eval_observables`` and``observable_names``).
                If ``None`` and ``eval_jac`` is ``False`` then no Jacobian matrix is
                computed.

        Returns:
            The output values of the functions of interest,
            as well as their Jacobian matrices if ``eval_jac`` is ``True``.

        Raises:
            ValueError: If a name in ``jacobian_names`` is not the name of
                a function of the problem.
        """
        # Get the functions to be evaluated
        from_original_functions = not self.__functions_are_preprocessed or no_db_no_norm
        functions = self.__get_functions(
            eval_obj,
            constraint_names,
            observable_names,
            eval_observables,
            from_original_functions,
        )

        # Evaluate the functions
        outputs = {}
        if functions:
            # N.B. either all functions expect normalized inputs or none of them do.
            preprocessed_x_vect = self.__preprocess_inputs(
                x_vect, normalize, functions[0].expects_normalized_inputs
            )

            for function in functions:
                try:  # Calling function.evaluate is faster than function()
                    outputs[function.name] = function.evaluate(preprocessed_x_vect)
                except ValueError:  # noqa: PERF203
                    LOGGER.exception("Failed to evaluate function %s", function.name)
                    raise

        if not eval_jac and jacobian_names is None:
            return outputs, {}

        # Evaluate the Jacobians
        if jacobian_names is not None:
            unknown_names = set(jacobian_names) - set(self.get_all_function_name())
            if unknown_names:
                if len(unknown_names) > 1:
                    message = "These names are"
                else:
                    message = "This name is"

                raise ValueError(
                    f"{message} not among the names of the functions: "
                    f"{pretty_str(unknown_names)}."
                )

            functions = self.__get_functions(
                self.objective.name in jacobian_names,
                [name for name in jacobian_names if name in self.constraint_names],
                [name for name in jacobian_names if name in self.__observable_names],
                True,
                from_original_functions,
            )

            if functions:
                # N.B. either all functions expect normalized inputs or none of them do.
                preprocessed_x_vect = self.__preprocess_inputs(
                    x_vect, normalize, functions[0].expects_normalized_inputs
                )

        jacobians = {}
        for function in functions:
            try:
                jacobians[function.name] = function.jac(preprocessed_x_vect)
            except ValueError:  # noqa: PERF203
                LOGGER.exception("Failed to evaluate Jacobian of %s.", function.name)
                raise

        return outputs, jacobians

    def __preprocess_inputs(
        self,
        input_value: ndarray | None,
        normalized: bool,
        normalization_expected: bool,
    ) -> ndarray:
        """Prepare the design variables for the evaluation of functions.

        Args:
            input_value: The design variables.
            normalized: Whether the design variables are normalized.
            normalization_expected: Whether the functions expect normalized variables.

        Returns:
            The prepared variables.
        """
        if input_value is None:
            input_value = self.design_space.get_current_value(normalize=normalized)
        elif self.activate_bound_check:
            if normalized:
                non_normalized_variables = self.design_space.unnormalize_vect(
                    input_value, no_check=True
                )
            else:
                non_normalized_variables = input_value

            self.design_space.check_membership(non_normalized_variables)

        if normalized and not normalization_expected:
            return self.design_space.unnormalize_vect(input_value, no_check=True)

        if not normalized and normalization_expected:
            return self.design_space.normalize_vect(input_value)

        return input_value

    def __get_functions(
        self,
        eval_obj: bool,
        constraint_names: Iterable[str] | None,
        observable_names: Iterable[str] | None,
        eval_observables: bool,
        from_original_functions: bool,
    ) -> list[MDOFunction]:
        """Return functions.

        Args:
            eval_obj: Whether to return the objective function.
            constraint_names: The names of the constraints to return.
                If ``None`` then all the constraints are evaluated.
            observable_names: The names of the observables to return.
                If ``None`` and ``eval_observables`` is True then all the observables
                are returned.
                If ``None`` and ``eval_observables`` is False then no observable is
                returned.
            eval_observables: Whether to return the observables.
            from_original_functions: Whether to get the functions from the original
                ones; otherwise get the functions from the pre-processed ones.

        Returns:
            The functions to be evaluated or differentiated.
        """
        use_nonproc_functions = (
            self.__functions_are_preprocessed and from_original_functions
        )
        if not eval_obj:
            functions = []
        elif use_nonproc_functions:
            functions = [self.nonproc_objective]
        else:
            functions = [self.objective]

        if constraint_names is not None:
            for name in constraint_names:
                functions.append(self.__get_constraint(name, from_original_functions))

        elif use_nonproc_functions:
            functions += self.nonproc_constraints
        else:
            functions += self.constraints

        if observable_names is not None:
            for name in observable_names:
                functions.append(self.__get_observable(name, from_original_functions))

        elif eval_observables and use_nonproc_functions:
            functions += self.nonproc_observables
        elif eval_observables:
            functions += self.observables

        return functions

    def preprocess_functions(
        self,
        is_function_input_normalized: bool = True,
        use_database: bool = True,
        round_ints: bool = True,
        eval_obs_jac: bool = False,
        support_sparse_jacobian: bool = False,
    ) -> None:
        """Pre-process all the functions and eventually the gradient.

        Required to wrap the objective function and constraints with the database
        and eventually the gradients by complex step or finite differences.

        Args:
            is_function_input_normalized: Whether to consider the function input as
                normalized and unnormalize it before the evaluation takes place.
            use_database: Whether to wrap the functions in the database.
            round_ints: Whether to round the integer variables.
            eval_obs_jac: Whether to evaluate the Jacobian of the observables.
            support_sparse_jacobian: Whether the driver support sparse Jacobian.
        """
        if round_ints:
            # Keep the rounding option only if there is an integer design variable
            round_ints = any(
                np_any(var_type == DesignSpace.DesignVariableType.INTEGER)
                for var_type in self.design_space.variable_types.values()
            )
        # Avoids multiple wrappings of functions when multiple executions
        # are performed, in bi-level scenarios for instance
        if not self.__functions_are_preprocessed:
            self.preprocess_options = {
                "is_function_input_normalized": is_function_input_normalized,
                "use_database": use_database,
                "round_ints": round_ints,
            }
            # Preprocess the constraints
            for icstr, cstr in enumerate(self.constraints):
                self.nonproc_constraints.append(cstr)
                p_cstr = self.__preprocess_func(
                    cstr,
                    is_function_input_normalized=is_function_input_normalized,
                    use_database=use_database,
                    round_ints=round_ints,
                    support_sparse_jacobian=support_sparse_jacobian,
                )
                p_cstr.special_repr = cstr.special_repr
                self.constraints[icstr] = p_cstr

            # Preprocess the observables
            for iobs, obs in enumerate(self.observables):
                self.nonproc_observables.append(obs)
                p_obs = self.__preprocess_func(
                    obs,
                    is_function_input_normalized=is_function_input_normalized,
                    use_database=use_database,
                    round_ints=round_ints,
                    is_observable=True,
                    support_sparse_jacobian=support_sparse_jacobian,
                )
                p_obs.special_repr = obs.special_repr
                self.observables[iobs] = p_obs

            for iobs, obs in enumerate(self.new_iter_observables):
                self.nonproc_new_iter_observables.append(obs)
                p_obs = self.__preprocess_func(
                    obs,
                    is_function_input_normalized=False,
                    use_database=use_database,
                    round_ints=round_ints,
                    is_observable=True,
                    support_sparse_jacobian=support_sparse_jacobian,
                )

                p_obs.special_repr = obs.special_repr
                self.new_iter_observables[iobs] = p_obs

            # Preprocess the objective
            self.nonproc_objective = self.objective
            self._objective = self.__preprocess_func(
                self.objective,
                is_function_input_normalized=is_function_input_normalized,
                use_database=use_database,
                round_ints=round_ints,
                support_sparse_jacobian=support_sparse_jacobian,
            )
            self.objective.special_repr = self.objective.special_repr
            self.objective.f_type = MDOFunction.FunctionType.OBJ
            self.__functions_are_preprocessed = True
            self.check()
            self.__eval_obs_jac = eval_obs_jac

    def execute_observables_callback(self, last_x: ndarray) -> None:
        """The callback function to be passed to the database.

        Call all the observables with the last design variables values as argument.

        Args:
            last_x: The design variables values from the last evaluation.
        """
        if not self.new_iter_observables:
            return

        for func in self.new_iter_observables:
            func(last_x)
            if self.__eval_obs_jac:
                func.jac(last_x)

    def __preprocess_func(
        self,
        func: MDOFunction,
        is_function_input_normalized: bool = True,
        use_database: bool = True,
        round_ints: bool = True,
        is_observable: bool = False,
        support_sparse_jacobian: bool = False,
    ) -> MDOFunction:
        """Wrap the function to differentiate it and store its call in the database.

        Only the computed gradients are stored in the database,
        not the eventual finite differences or complex step perturbed evaluations.

        Args:
            func: The scaled and derived function to be pre-processed.
            is_function_input_normalized: Whether to consider the function input as
                normalized and unnormalize it before the evaluation takes place.
            use_database: If ``True``, then the function is wrapped in the database.
            round_ints: If ``True``, then round the integer variables.
            is_observable: If ``True``, new_iter_listeners are not called
                when function is called (avoid recursive call)
            support_sparse_jacobian: Whether the driver support sparse Jacobian.

        Returns:
            The pre-processed function.
        """
        self.check_format(func)
        # First differentiate it so that the finite differences evaluations
        # are not stored in the database, which would be the case in the other
        # way round
        # Also, store non normalized values in the database for further
        # exploitation
        # Convert Jacobian in dense format if needed
        if not support_sparse_jacobian:
            func = DenseJacobianFunction(func)
        if (
            isinstance(func, MDOLinearFunction)
            and not round_ints
            and is_function_input_normalized
        ):
            func = self.__normalize_linear_function(func)
        else:
            func = NormFunction(func, is_function_input_normalized, round_ints, self)

        if self.differentiation_method in set(self.ApproximationMode):
            self.__add_fd_jac(func, is_function_input_normalized)

        # Cast to real value, the results can be a complex number (ComplexStep)
        if use_database:
            func = NormDBFunction(
                func, is_function_input_normalized, is_observable, self
            )

        return func

    def __normalize_linear_function(
        self,
        orig_func: MDOLinearFunction,
    ) -> MDOLinearFunction:
        """Create a linear function using a scaled input vector.

        Args:
            orig_func: The original linear function

        Returns:
            The scaled linear function.

        Raises:
            TypeError: If the original function is not an :class:`.MDOLinearFunction`.
        """
        if not isinstance(orig_func, MDOLinearFunction):
            raise TypeError("Original function must be linear")
        design_space = self.design_space

        # Get normalization factors and shift
        norm_policies = design_space.dict_to_array(design_space.normalize)
        norm_factors = where(
            norm_policies,
            design_space.get_upper_bounds() - design_space.get_lower_bounds(),
            1.0,
        )
        shift = where(norm_policies, design_space.get_lower_bounds(), 0.0)

        # Build the normalized linear function
        if isinstance(orig_func.coefficients, sparse_classes):
            coefficients = deepcopy(orig_func.coefficients)
            coefficients.data *= norm_factors[coefficients.indices]
        else:
            coefficients = multiply(orig_func.coefficients, norm_factors)

        value_at_zero = orig_func(shift)
        return MDOLinearFunction(
            coefficients,
            orig_func.name,
            orig_func.f_type,
            orig_func.input_names,
            value_at_zero,
        )

    def __add_fd_jac(
        self,
        func: MDOFunction,
        normalize: bool,
    ) -> None:
        """Add a pointer to the approached Jacobian of the function.

        This Jacobian matrix is generated according :attr:`.differentiation_method`.

        Args:
            func: The function to be derivated.
            normalize: Whether to unnormalize the input vector of the function
                before evaluate it.

        Raises:
            ValueError: When the current value is not defined.
        """
        if not self.design_space.has_current_value():
            raise ValueError("The design space has no current value.")

        if self.differentiation_method not in set(self.ApproximationMode):
            return

        differentiation_object = GradientApproximatorFactory().create(
            self.differentiation_method,
            func.evaluate,
            step=self.fd_step,
            design_space=self.design_space,
            normalize=normalize,
            parallel=self.__parallel_differentiation,
            **self.__parallel_differentiation_options,
        )
        func.jac = differentiation_object.f_gradient

    def check(self) -> None:
        """Check if the optimization problem is ready for run.

        Raises:
            ValueError: If the objective function is missing.
        """
        if self.objective is None:
            raise ValueError("Missing objective function in OptimizationProblem")
        self.design_space.check()
        self.__check_differentiation_method()
        self.check_format(self.objective)
        self.__check_functions()

    def __check_functions(self) -> None:
        """Check that the constraints are well declared.

        Raises:
            ValueError: If a function declared as a constraint has the wrong type.
        """
        for cstr in self.constraints:
            self.check_format(cstr)
            if not cstr.is_constraint():
                raise ValueError(
                    f"Constraint type is not eq or ineq !, got {cstr.f_type}"
                    " instead "
                )
        self.check_format(self.objective)

    def __check_differentiation_method(self):
        """Check that the differentiation method is in allowed ones.

        Available ones are: :attr:`.OptimizationProblem.DifferentiationMethod`.

        Raises:
            ValueError: If either
                the differentiation method is unknown,
                the complex step is null or
                the finite differences' step is null.
        """
        if self.differentiation_method == self.ApproximationMode.COMPLEX_STEP:
            if self.fd_step == 0:
                raise ValueError("ComplexStep step is null!")
            if self.fd_step.imag != 0:
                LOGGER.warning(
                    "Complex step method has an imaginary "
                    "step while required a pure real one."
                    " Auto setting the real part"
                )
                self.fd_step = self.fd_step.imag
        elif self.differentiation_method == self.ApproximationMode.FINITE_DIFFERENCES:
            if self.fd_step == 0:
                raise ValueError("Finite differences step is null!")
            if self.fd_step.imag != 0:
                LOGGER.warning(
                    "Finite differences method has a complex "
                    "step while required a pure real one."
                    " Auto setting the imaginary part to 0"
                )
                self.fd_step = self.fd_step.real

    # TODO: API: to be deprecated in favor of self.minimize_objective
    def change_objective_sign(self) -> None:
        """Change the objective function sign in order to minimize its opposite.

        The :class:`.OptimizationProblem` expresses any optimization problem as a
        minimization problem. Then, an objective function originally expressed as a
        performance function to maximize must be converted into a cost function to
        minimize, by means of this method.
        """
        self.__minimize_objective = not self.__minimize_objective
        self.objective = -self.objective

    def _satisfied_constraint(
        self,
        cstr_type: MDOFunction.ConstraintType,
        value: ndarray,
    ) -> bool:
        """Determine if an evaluation satisfies a constraint within a given tolerance.

        Args:
            cstr_type: The type of the constraint.
            value: The value of the constraint.

        Returns:
            Whether a value satisfies a constraint.
        """
        if cstr_type == MDOFunction.ConstraintType.EQ:
            return np_all(np_abs(value) <= self.eq_tolerance)
        return np_all(value <= self.ineq_tolerance)

    def is_point_feasible(
        self,
        out_val: dict[str, ndarray],
        constraints: Iterable[MDOFunction] | None = None,
    ) -> bool:
        """Check if a point is feasible.

        Notes:
            If the value of a constraint is absent from this point,
            then this constraint will be considered satisfied.

        Args:
            out_val: The values of the objective function, and eventually constraints.
            constraints: The constraints whose values are to be tested.
                If ``None``, then take all constraints of the problem.

        Returns:
            The feasibility of the point.
        """
        if constraints is None:
            constraints = self.get_ineq_constraints() + self.get_eq_constraints()
        feasible = True
        for constraint in constraints:
            # look for the evaluation of the constraint
            eval_cstr = out_val.get(constraint.name, None)
            # if evaluation exists, check if it is satisfied
            if eval_cstr is None or not self._satisfied_constraint(
                constraint.f_type, eval_cstr
            ):
                feasible = False
                break
        return feasible

    def get_feasible_points(
        self,
    ) -> tuple[list[ndarray], list[dict[str, float | list[int]]]]:
        """Retrieve the feasible points within a given tolerance.

        This tolerance is defined by
        :attr:`.OptimizationProblem.eq_tolerance` for equality constraints and
        :attr:`.OptimizationProblem.ineq_tolerance` for inequality ones.

        Returns:
            The values of the design variables and objective function
            for the feasible points.
        """
        x_history = []
        f_history = []
        constraints = self.get_ineq_constraints() + self.get_eq_constraints()

        for x_vect, out_val in self.database.items():
            feasible = self.is_point_feasible(out_val, constraints=constraints)

            # if all constraints are satisfied, store the vector
            if feasible:
                x_history.append(x_vect.unwrap())
                f_history.append(out_val)
        return x_history, f_history

    # TODO: API: rename to check_design_point_is_feasible
    def get_violation_criteria(
        self,
        x_vect: ndarray,
    ) -> tuple[bool, float]:
        r"""Check if a design point is feasible and measure its constraint violation.

        The constraint violation measure at a design point :math:`x` is

        .. math::

           \lVert\max(g(x)-\varepsilon_{\text{ineq}},0)\rVert_2^2
           +\lVert|\max(|h(x)|-\varepsilon_{\text{eq}},0)\rVert_2^2

        where :math:`\|.\|_2` is the Euclidean norm,
        :math:`g(x)` is the inequality constraint vector,
        :math:`h(x)` is the equality constraint vector,
        :math:`\varepsilon_{\text{ineq}}` is the tolerance
        for the inequality constraints
        and
        :math:`\varepsilon_{\text{eq}}` is the tolerance for the equality constraints.

        If the design point is feasible, the constraint violation measure is 0.

        Args:
            x_vect: The design point :math:`x`.

        Returns:
            Whether the design point is feasible,
            and its constraint violation measure.
        """
        violation = 0.0
        x_vect_is_feasible = True
        output_names_to_values = self.database.get(x_vect)
        for constraint in self.constraints:
            constraint_value = output_names_to_values.get(constraint.name)
            if constraint_value is None:
                break

            f_type = constraint.f_type
            if self._satisfied_constraint(f_type, constraint_value):
                continue

            x_vect_is_feasible = False
            if isnan(constraint_value).any():
                return x_vect_is_feasible, inf

            if f_type == MDOFunction.ConstraintType.INEQ:
                tolerance = self.ineq_tolerance
            else:
                tolerance = self.eq_tolerance
                constraint_value = abs(constraint_value)

            if isinstance(constraint_value, ndarray):
                violated_components = (constraint_value > tolerance).nonzero()
                constraint_value = constraint_value[violated_components]

            violation += norm(constraint_value - tolerance) ** 2

        return x_vect_is_feasible, violation

    def get_best_infeasible_point(
        self,
    ) -> BestInfeasiblePointType:
        """Retrieve the best infeasible point within a given tolerance.

        Returns:
            The best infeasible point expressed as
            the design variables values,
            the objective function value,
            the feasibility of the point
            and the functions values.
        """
        x_history = []
        f_history = []
        is_feasible = []
        viol_criteria = []
        for x_vect, out_val in self.database.items():
            is_pt_feasible, f_violation = self.get_violation_criteria(x_vect)
            is_feasible.append(is_pt_feasible)
            viol_criteria.append(f_violation)
            x_history.append(x_vect.unwrap())
            f_history.append(out_val)

        is_opt_feasible = False
        if viol_criteria:
            best_i = int(argmin(array(viol_criteria)))
            is_opt_feasible = is_feasible[best_i]
        else:
            best_i = 0

        if len(f_history) <= best_i:
            outputs_opt = {}
            x_opt = None
            f_opt = None
        else:
            outputs_opt = f_history[best_i]
            x_opt = x_history[best_i]
            f_opt = outputs_opt.get(self.objective.name)
        if isinstance(f_opt, ndarray) and len(f_opt) == 1:
            f_opt = f_opt[0]

        return x_opt, f_opt, is_opt_feasible, outputs_opt

    def __get_optimum_infeas(
        self,
    ) -> OptimumSolutionType:
        """Retrieve the optimum solution.

        Use a feasibility tolerance, when there is no feasible point.

        Returns:
            The optimum solution expressed by
            the design variables values,
            the objective function value,
            the constraints values
            and the constraints gradients values.
        """
        msg = (
            "Optimization found no feasible point ! "
            " The least infeasible point is selected."
        )
        LOGGER.warning(msg)
        x_opt, f_opt, _, f_history = self.get_best_infeasible_point()
        c_opt = {}
        c_opt_grad = {}
        constraints = self.get_ineq_constraints() + self.get_eq_constraints()
        for constraint in constraints:
            c_opt[constraint.name] = f_history.get(constraint.name)
            f_key = Database.get_gradient_name(constraint.name)
            c_opt_grad[constraint.name] = f_history.get(f_key)
        return x_opt, f_opt, c_opt, c_opt_grad

    def __get_optimum_feas(
        self,
        feas_x: Sequence[ndarray],
        feas_f: Sequence[dict[str, float | list[int]]],
    ) -> OptimumSolutionType:
        """Retrieve the optimum solution.

        Use a feasibility tolerances, when there is a feasible point.

        Args:
            feas_x: The values of the design parameters for the feasible evaluations.
            feas_f: The values of the functions for the feasible evaluations.

        Returns:
            The optimum solution expressed by
            the design variables values,
            the objective function value,
            the constraints values
            and the constraints gradients values.
        """
        f_opt, x_opt = inf, None
        c_opt = {}
        c_opt_grad = {}
        obj_name = self.objective.name
        constraints = self.get_ineq_constraints() + self.get_eq_constraints()
        for i, out_val in enumerate(feas_f):
            obj_eval = out_val.get(obj_name)
            if obj_eval is None or isinstance(obj_eval, Number) or obj_eval.size == 1:
                tmp_objeval = obj_eval
            else:
                tmp_objeval = norm(obj_eval)
            if tmp_objeval is not None and tmp_objeval < f_opt:
                f_opt = tmp_objeval
                x_opt = feas_x[i]
                for constraint in constraints:
                    c_name = constraint.name
                    c_opt[c_name] = feas_f[i].get(c_name)
                    c_key = Database.get_gradient_name(c_name)
                    c_opt_grad[constraint.name] = feas_f[i].get(c_key)
        if isinstance(f_opt, ndarray) and len(f_opt) == 1:
            f_opt = f_opt[0]
        return x_opt, f_opt, c_opt, c_opt_grad

    def get_optimum(self) -> OptimumType:
        """Return the optimum solution within a given feasibility tolerances.

        Returns:
            The optimum result,
            defined by:

            - the value of the objective function,
            - the value of the design variables,
            - the indicator of feasibility of the optimal solution,
            - the value of the constraints,
            - the value of the gradients of the constraints.

        Raises:
            ValueError: When the optimization database is empty.
        """
        if not self.database:
            raise ValueError("Optimization history is empty")
        feas_x, feas_f = self.get_feasible_points()

        if not feas_x:
            is_feas = False
            x_opt, f_opt, c_opt, c_opt_d = self.__get_optimum_infeas()
        else:
            is_feas = True
            x_opt, f_opt, c_opt, c_opt_d = self.__get_optimum_feas(feas_x, feas_f)
        return f_opt, x_opt, is_feas, c_opt, c_opt_d

    def get_last_point(self) -> OptimumType:
        """Return the last design point.

        Returns:
            The last point result,
            defined by:

            - the value of the objective function,
            - the value of the design variables,
            - the indicator of feasibility of the last point,
            - the value of the constraints,
            - the value of the gradients of the constraints.

        Raises:
            ValueError: When the optimization database is empty.
        """
        if not self.database:
            raise ValueError("Optimization history is empty")
        x_last = self.database.get_x_vect(-1)
        f_last = self.database.get_function_value(self.objective.name, -1)
        is_feas = self.is_point_feasible(self.database[x_last], self.constraints)
        c_last = {}
        c_last_grad = {}
        for constraint in self.constraints:
            c_last[constraint.name] = self.database[x_last].get(constraint.name)
            f_key = Database.get_gradient_name(constraint.name)
            c_last_grad[constraint.name] = self.database[x_last].get(f_key)

        return f_last, x_last, is_feas, c_last, c_last_grad

    @property
    def __string_representation(self) -> MultiLineString:
        """The string representation of the optimization problem."""
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

        objective_function = [line for line in repr(self.objective).split("\n") if line]
        mls.add(optimize_verb + objective_function[0][start:])
        for line in objective_function[1:]:
            mls.add(" " * len(optimize_verb) + line)

        # variables representation
        mls.add("with respect to {}", pretty_str(self.design_space.variable_names))
        if self.has_constraints():
            mls.add("subject to constraints:")
            mls.indent()
            for constraints in self.get_ineq_constraints():
                constraints = [cstr for cstr in str(constraints).split("\n") if cstr]
                for constraint in constraints:
                    mls.add(constraint)
            for constraints in self.get_eq_constraints():
                constraints = [cstr for cstr in str(constraints).split("\n") if cstr]
                for constraint in constraints:
                    mls.add(constraint)
        return mls

    def __repr__(self) -> str:
        return str(self.__string_representation)

    def _repr_html_(self) -> str:
        return self.__string_representation._repr_html_()

    @staticmethod
    def __store_h5data(
        group: Any,
        data_array: ndarray[Number] | str | list[str | Number],
        dataset_name: str,
        dtype: str | None = None,
    ) -> None:
        """Store an array in a hdf5 file group.

        Args:
            group: The group pointer.
            data_array: The data to be stored.
            dataset_name: The name of the dataset to store the array.
            dtype: Numpy dtype or string. If ``None``, dtype('f') will be used.
        """
        if data_array is None or (
            isinstance(data_array, Iterable) and not len(data_array)
        ):
            return
        if isinstance(data_array, ndarray):
            data_array = data_array.real
        if isinstance(data_array, str):
            data_array = array([data_array], dtype="bytes")
        if isinstance(data_array, list):
            all_str = reduce(
                lambda x, y: x or y,
                (isinstance(data, str) for data in data_array),
            )
            if all_str:
                data_array = array([data_array], dtype="bytes")
                dtype = data_array.dtype
        group.create_dataset(dataset_name, data=data_array, dtype=dtype)

    @classmethod
    def __store_attr_h5data(cls, obj: Any, group: Group) -> None:
        """Store an object in the HDF5 dataset.

        The object shall be a mapping or have a method to_dict().

        Args:
            obj: The object to store
            group: The hdf5 group.
        """
        data = obj if isinstance(obj, Mapping) else obj.to_dict()
        for name, value in data.items():
            dtype = None
            if isinstance(value, str):
                value = value.encode("ascii", "ignore")
            elif isinstance(value, bytes):
                value = value.decode()
            elif isinstance(value, Mapping) and not isinstance(value, DesignSpace):
                cls.__store_attr_h5data(value, group.require_group(f"/{name}"))
                continue
            elif hasattr(value, "__iter__") and not (
                isinstance(value, ndarray) and issubdtype(value.dtype, np_number)
            ):
                value = [
                    att.encode("ascii", "ignore") if isinstance(att, str) else att
                    for att in value
                ]
                dtype = h5py.special_dtype(vlen=str)

            cls.__store_h5data(group, value, name, dtype)

    def to_hdf(self, file_path: str | Path, append: bool = False) -> None:
        """Export the optimization problem to an HDF file.

        Args:
            file_path: The path of the file to store the data.
            append: If ``True``, then the data are appended to the file if not empty.
        """
        LOGGER.info("Export optimization problem to file: %s", str(file_path))

        mode = "a" if append else "w"

        with h5py.File(file_path, mode) as h5file:
            no_design_space = DesignSpace.DESIGN_SPACE_GROUP not in h5file

            if not append or self.OPT_DESCR_GROUP not in h5file:
                opt_group = h5file.require_group(self.OPT_DESCR_GROUP)

                for attr_name in self.OPTIM_DESCRIPTION:
                    attr = getattr(self, attr_name)
                    self.__store_h5data(opt_group, attr, attr_name)

                obj_group = h5file.require_group(self.OBJECTIVE_GROUP)
                self.__store_attr_h5data(self.objective, obj_group)

                if self.constraints:
                    constraint_group = h5file.require_group(self.CONSTRAINTS_GROUP)
                    for constraint in self.constraints:
                        c_subgroup = constraint_group.require_group(constraint.name)
                        self.__store_attr_h5data(constraint, c_subgroup)

                if self.observables:
                    observables_group = h5file.require_group(self.OBSERVABLES_GROUP)
                    for observable in self.observables:
                        o_subgroup = observables_group.require_group(observable.name)
                        self.__store_attr_h5data(observable, o_subgroup)

                if hasattr(self.solution, "to_dict"):
                    # TODO: replace by "if self.solution is None"
                    sol_group = h5file.require_group(self.SOLUTION_GROUP)
                    self.__store_attr_h5data(self.solution, sol_group)

        self.database.to_hdf(file_path, append=True)

        # Design space shall remain the same in append mode
        if not append or no_design_space:
            self.design_space.to_hdf(file_path, append=True)

    @classmethod
    def from_hdf(
        cls, file_path: str | Path, x_tolerance: float = 0.0
    ) -> OptimizationProblem:
        """Import an optimization history from an HDF file.

        Args:
            file_path: The file containing the optimization history.
            x_tolerance: The tolerance on the design variables when reading the file.

        Returns:
            The read optimization problem.
        """
        LOGGER.info("Import optimization problem from file: %s", file_path)

        design_space = DesignSpace.from_file(file_path)
        opt_pb = OptimizationProblem(design_space, input_database=file_path)

        with h5py.File(file_path) as h5file:
            if opt_pb.SOLUTION_GROUP in h5file:
                group_data = cls.__h5_group_to_dict(h5file, opt_pb.SOLUTION_GROUP)
                if "x_0_as_dict" in h5file:
                    group_data["x_0_as_dict"] = cls.__h5_group_to_dict(
                        h5file, "x_0_as_dict"
                    )
                if "x_opt_as_dict" in h5file:
                    group_data["x_opt_as_dict"] = cls.__h5_group_to_dict(
                        h5file, "x_opt_as_dict"
                    )
                attr = OptimizationResult.from_dict(group_data)
                opt_pb.solution = attr

            group_data = cls.__h5_group_to_dict(h5file, opt_pb.OBJECTIVE_GROUP)
            attr = MDOFunction.init_from_dict_repr(**group_data)

            # The generated functions can be called at the x stored in
            # the database
            attr.set_pt_from_database(
                opt_pb.database, design_space, x_tolerance=x_tolerance
            )
            opt_pb.objective = attr

            group = get_hdf5_group(h5file, opt_pb.OPT_DESCR_GROUP)
            for attr_name, attr in group.items():
                val = attr[()]
                if isinstance(val, ndarray) and isinstance(val[0], bytes_):
                    val = val[0].decode()

                # Set the private attribute __minimize_objective instead of the property
                # to avoid an unnecessary change of sign of the objective function.
                if attr_name == "minimize_objective":
                    attr_name = "_OptimizationProblem__minimize_objective"
                setattr(opt_pb, attr_name, val)

            if opt_pb.CONSTRAINTS_GROUP in h5file:
                group = get_hdf5_group(h5file, opt_pb.CONSTRAINTS_GROUP)

                for cstr_name in group:
                    group_data = cls.__h5_group_to_dict(group, cstr_name)
                    attr = MDOFunction.init_from_dict_repr(**group_data)
                    opt_pb.constraints.append(attr)

            if opt_pb.OBSERVABLES_GROUP in h5file:
                group = get_hdf5_group(h5file, opt_pb.OBSERVABLES_GROUP)

                for observable_name in group:
                    group_data = cls.__h5_group_to_dict(group, observable_name)
                    attr = MDOFunction.init_from_dict_repr(**group_data)
                    opt_pb.observables.append(attr)

        return opt_pb

    def to_dataset(
        self,
        name: str = "",
        categorize: bool = True,
        opt_naming: bool = True,
        export_gradients: bool = False,
        input_values: Iterable[ndarray] = (),
    ) -> Dataset:
        """Export the database of the optimization problem to a :class:`.Dataset`.

        The variables can be classified into groups:
        :attr:`.Dataset.DESIGN_GROUP` or :attr:`.Dataset.INPUT_GROUP`
        for the design variables
        and :attr:`.Dataset.FUNCTION_GROUP` or :attr:`.Dataset.OUTPUT_GROUP`
        for the functions
        (objective, constraints and observables).

        Args:
            name: The name to be given to the dataset.
                If empty, use the name of the :attr:`.OptimizationProblem.database`.
            categorize: Whether to distinguish
                between the different groups of variables.
                Otherwise, group all the variables in :attr:`.Dataset.PARAMETER_GROUP``.
            opt_naming: Whether to use
                :attr:`.Dataset.DESIGN_GROUP` and :attr:`.Dataset.FUNCTION_GROUP`
                as groups.
                Otherwise,
                use :attr:`.Dataset.INPUT_GROUP` and :attr:`.Dataset.OUTPUT_GROUP`.
            export_gradients: Whether to export the gradients of the functions
                (objective function, constraints and observables)
                if the latter are available in the database of the optimization problem.
            input_values: The input values to be considered.
                If empty, consider all the input values of the database.

        Returns:
            A dataset built from the database of the optimization problem.
        """
        dataset_name = name or self.database.name
        # Set the different groups
        if categorize:
            if opt_naming:
                dataset_class = OptimizationDataset
                input_group = OptimizationDataset.DESIGN_GROUP
                output_group = OptimizationDataset.FUNCTION_GROUP
                gradient_group = OptimizationDataset.GRADIENT_GROUP
            else:
                dataset_class = IODataset
                input_group = IODataset.INPUT_GROUP
                output_group = IODataset.OUTPUT_GROUP
                gradient_group = IODataset.GRADIENT_GROUP
        else:
            dataset_class = Dataset
            input_group = output_group = gradient_group = Dataset.DEFAULT_GROUP

        # Add database inputs
        input_names = self.design_space.variable_names
        names_to_sizes = self.design_space.variable_sizes
        input_history = array(self.database.get_x_vect_history())
        n_samples = len(input_history)
        positions = []
        offset = int(categorize & opt_naming)
        for input_value in input_values:
            _positions = ((input_history == input_value).all(axis=1)).nonzero()[0]
            positions.extend((_positions + offset).tolist())

        data = [input_history.real]
        columns = [
            (input_group, name, index)
            for name in input_names
            for index in range(names_to_sizes[name])
        ]

        # Add database outputs
        variable_names = self.database.get_function_names()
        output_names = [name for name in variable_names if name not in input_names]
        self.__add_data_to_database(
            data, columns, output_names, n_samples, output_group, False
        )

        # Add database output gradients
        if export_gradients:
            self.__add_data_to_database(
                data, columns, output_names, n_samples, gradient_group, True
            )

        return dataset_class(
            hstack(data),
            dataset_name=dataset_name,
            columns=MultiIndex.from_tuples(
                columns,
                names=dataset_class.COLUMN_LEVEL_NAMES,
            ),
        ).get_view(indices=positions)

    def __add_data_to_database(
        self,
        data: list[NDArray[float]],
        columns: list[tuple[str, str, int]],
        output_names: Iterable[str],
        n_samples: int,
        group: str,
        store_gradient: bool,
    ) -> None:
        """Add the database output gradients to the dataset.

        Args:
            data: The sequence of data arrays to be augmented with the output data.
            columns: The multi-index columns to be augmented with the output names.
            output_names: The names of the outputs in the database.
            n_samples: The total number of samples,
                including possible points where the evaluation failed.
            group: The dataset group where the variables will be added.
            store_gradient: Whether the variable of interest
                is the gradient of the output.
        """
        x_vect_history = array(self.database.get_x_vect_history())
        for output_name in output_names:
            if store_gradient:
                function_name = Database.get_gradient_name(output_name)
                if self.database.check_output_history_is_empty(function_name):
                    continue
            else:
                function_name = output_name

            history, input_history = self.database.get_function_history(
                function_name=function_name, with_x_vect=True
            )
            history = (
                self.__replace_missing_values(
                    history,
                    input_history,
                    x_vect_history,
                )
                .reshape((n_samples, -1))
                .real
            )
            data.append(history)
            columns.extend([(group, function_name, i) for i in range(history.shape[1])])

    @staticmethod
    def __replace_missing_values(
        output_history: ndarray,
        input_history: ndarray,
        full_input_history: ndarray,
    ) -> ndarray:
        """Replace the missing output values with NaN.

        Args:
            output_history: The output data history with possibly missing values.
            input_history: The input data history with possibly missing values.
            full_input_history: The complete input data history, with no missing values.

        Returns:
            The output data history where missing values have been replaced with NaN.
        """
        database_size = full_input_history.shape[0]

        if len(input_history) != database_size:
            # There are fewer entries than in the full input history.
            # Add NaN values at the missing input data.
            # N.B. the input data are assumed to be in the same order.
            index = 0
            for input_data in input_history:
                while not array_equal(input_data, full_input_history[index]):
                    output_history = insert(output_history, index, nan, 0)
                    index += 1

                index += 1

            return insert(output_history, [index] * (database_size - index), nan, 0)

        return output_history

    @staticmethod
    def __h5_group_to_dict(
        h5_handle: h5py.File | h5py.Group,
        group_name: str,
    ) -> dict[str, str | list[str]]:
        """Convert the values of a hdf5 dataset.

        Values that are of the kind string or bytes are converted
        to string or list of strings.

        Args:
            h5_handle: A hdf5 file or group.
            group_name: The name of the group to be converted.

        Returns:
            The converted dataset.
        """
        converted = {}

        group = get_hdf5_group(h5_handle, group_name)

        for key, value in group.items():
            value = value[()]

            # h5py does not handle bytes natively, it maps it to a numpy generic type
            if isinstance(value, ndarray) and value.dtype.type in {
                numpy.object_,
                bytes_,
            }:
                value = value[0] if value.size == 1 else value.tolist()

            if isinstance(value, bytes):
                value = value.decode()

            if isinstance(value, list):
                value = [
                    sub_value.decode() if isinstance(sub_value, bytes) else sub_value
                    for sub_value in value
                ]

            converted[key] = value

        return converted

    def get_data_by_names(
        self,
        names: str | Iterable[str],
        as_dict: bool = True,
        filter_non_feasible: bool = False,
    ) -> ndarray | dict[str, ndarray]:
        """Return the data for specific names of variables.

        Args:
            names: The names of the variables.
            as_dict: If ``True``, return values as dictionary.
            filter_non_feasible: If ``True``, remove the non-feasible points from
                the data.

        Returns:
            The data related to the variables.
        """
        dataset = self.to_dataset("OptimizationProblem")
        if as_dict:
            data = dataset.get_view(variable_names=names).to_dict(orient="list")
        else:
            data = dataset.get_view(variable_names=names).to_numpy()

        if filter_non_feasible:
            x_feasible, _ = self.get_feasible_points()
            feasible_indexes = [self.database.get_iteration(x) - 1 for x in x_feasible]
            if as_dict:
                for key, value in data.items():
                    data[key] = array(value)[feasible_indexes]
            else:
                data = data[feasible_indexes, :]

        return data

    @property
    def is_mono_objective(self) -> bool:
        """Whether the optimization problem is mono-objective.

        Raises:
            ValueError: When the dimension of the objective cannot be determined.
        """
        obj_dim = self.objective.dim
        if obj_dim != 0:
            return obj_dim == 1
        n_outvars = len(self.objective.output_names)
        if n_outvars == 0:
            raise ValueError("Cannot determine the dimension of the objective.")
        return n_outvars == 1

    def get_functions_dimensions(
        self, names: Iterable[str] | None = None
    ) -> dict[str, int]:
        """Return the dimensions of the outputs of the problem functions.

        Args:
            names: The names of the functions.
                If ``None``, then the objective and all the constraints are considered.

        Returns:
            The dimensions of the outputs of the problem functions.
            The dictionary keys are the functions names
            and the values are the functions dimensions.
        """
        if names is None:
            names = [self.objective.name, *self.get_constraint_names()]

        return {name: self.get_function_dimension(name) for name in names}

    def get_function_dimension(self, name: str) -> int:
        """Return the dimension of a function of the problem (e.g. a constraint).

        Args:
            name: The name of the function.

        Returns:
            The dimension of the function.

        Raises:
            ValueError: If the function name is unknown to the problem.
            RuntimeError: If the function dimension is not unavailable.
        """
        # Check that the required function belongs to the problem and get it
        for func in self.get_all_functions():
            if func.name == name:
                function = func
                break
        else:
            raise ValueError(f"The problem has no function named {name}.")

        # Get the dimension of the function output
        if function.dim:
            return function.dim

        if self.design_space.has_current_value():
            if function.expects_normalized_inputs:
                current_variables = self.get_x0_normalized()
            else:
                current_variables = self.design_space.get_current_value()

            return atleast_1d(function(current_variables)).size

        raise RuntimeError(f"The output dimension of function {name} is not available.")

    def get_number_of_unsatisfied_constraints(
        self,
        design_variables: ndarray,
        values: Mapping[str, float | ndarray] = MappingProxyType({}),
    ) -> int:
        """Return the number of scalar constraints not satisfied by design variables.

        Args:
            design_variables: The design variables.
            values: The values of the constraints.
                N.B. the missing values will be read from the database or computed.

        Returns:
            The number of unsatisfied scalar constraints.
        """
        n_unsatisfied = 0
        missing_names = set(self.get_constraint_names()).difference(values)
        if missing_names:
            constraints_values = self.evaluate_functions(
                design_variables,
                eval_obj=False,
                eval_observables=False,
                normalize=False,
                constraint_names=missing_names,
            )[0]
            constraints_values.update(values)
        else:
            constraints_values = values

        for constraint in self.constraints:
            value = atleast_1d(constraints_values[constraint.name])
            if constraint.f_type == MDOFunction.ConstraintType.EQ:
                value = numpy.absolute(value)
                tolerance = self.eq_tolerance
            else:
                tolerance = self.ineq_tolerance

            n_unsatisfied += sum(value > tolerance)

        return n_unsatisfied

    def get_scalar_constraint_names(self) -> list[str]:
        """Return the names of the scalar constraints.

        Returns:
            The names of the scalar constraints.
        """
        constraint_names = []
        for constraint in self.constraints:
            dimension = self.get_function_dimension(constraint.name)
            if dimension == 1:
                constraint_names.append(constraint.name)
            else:
                constraint_names.extend([
                    constraint.get_indexed_name(index) for index in range(dimension)
                ])
        return constraint_names

    def reset(
        self,
        database: bool = True,
        current_iter: bool = True,
        design_space: bool = True,
        function_calls: bool = True,
        preprocessing: bool = True,
    ) -> None:
        """Partially or fully reset the optimization problem.

        Args:
            database: Whether to clear the database.
            current_iter: Whether to reset the current iteration
                :attr:`.OptimizationProblem.current_iter`.
            design_space: Whether to reset the current point
                of the :attr:`.OptimizationProblem.design_space`
                to its initial value (possibly none).
            function_calls: Whether to reset the number of calls of the functions.
            preprocessing: Whether to turn the pre-processing of functions to False.
        """
        if current_iter:
            self.current_iter = 0

        if database:
            self.database.clear()

        if design_space:
            self.design_space.set_current_value(self.__initial_current_x)

        if function_calls and MDOFunction.activate_counters:
            for func in self.get_all_functions():
                func.n_calls = 0

        if preprocessing and self.__functions_are_preprocessed:
            self.objective = self.nonproc_objective
            self.nonproc_objective = None
            self.constraints = self.nonproc_constraints
            self.nonproc_constraints = []
            self.observables = self.nonproc_observables
            self.nonproc_observables = []
            self.new_iter_observables = self.nonproc_new_iter_observables
            self.nonproc_new_iter_observables = []
            self.__functions_are_preprocessed = False

    def __get_constraint(
        self, name: str, from_original_constraints: bool = False
    ) -> MDOFunction:
        """Return a constraint of the problem.

        Args:
            name: The name of the constraint.
            from_original_constraints: Whether to get the constraint from the original
                constraints; otherwise get the constraint from the pre-processed
                constraints.

        Returns:
            The constraint.
        """
        return self.__get_function(
            name,
            from_original_constraints,
            self.get_constraint_names(),
            self.nonproc_constraints,
            self.constraints,
            self.CONSTRAINTS_GROUP,
        )

    def __get_function(
        self,
        name: str,
        from_original_functions: bool,
        names: Iterable[str],
        original_functions: Iterable[MDOFunction],
        preprocessed_functions: Iterable[MDOFunction],
        group_name: str,
    ) -> MDOFunction:
        """Return a function of the problem.

        Args:
            name: The name of the function.
            from_original_functions: Whether to get the function from the original
                functions; otherwise get the function for the pre-processed functions.
            names: The names of the available functions.
            original_functions: The original functions.
            preprocessed_functions: The pre-processed functions.
            group_name: The name of the group of functions.

        Returns:
            The function.

        Raises:
            ValueError: If the name is not among the names of the available functions.
        """
        if name not in names:
            raise ValueError(
                f"{name} is not among the names of the {group_name}: "
                f"{pretty_str(names)}."
            )

        if from_original_functions and self.__functions_are_preprocessed:
            functions = original_functions
        else:
            functions = preprocessed_functions

        return next(function for function in functions if function.name == name)
