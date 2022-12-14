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

A (possible vector) objective function with a :class:`.MDOFunction` type
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
from copy import deepcopy
from functools import reduce
from numbers import Number
from pathlib import Path
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Tuple

import h5py
import numpy
from numpy import abs as np_abs
from numpy import all as np_all
from numpy import any as np_any
from numpy import argmin
from numpy import array
from numpy import array_equal
from numpy import inf
from numpy import insert
from numpy import isnan
from numpy import issubdtype
from numpy import multiply
from numpy import nan
from numpy import ndarray
from numpy import number as np_number
from numpy import where
from numpy.core import atleast_1d
from numpy.linalg import norm

from gemseo.algos.aggregation.aggregation_func import aggregate_iks
from gemseo.algos.aggregation.aggregation_func import aggregate_ks
from gemseo.algos.aggregation.aggregation_func import aggregate_max
from gemseo.algos.aggregation.aggregation_func import aggregate_sum_square
from gemseo.algos.database import Database
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.dataset import Dataset
from gemseo.core.derivatives.derivation_modes import COMPLEX_STEP
from gemseo.core.derivatives.derivation_modes import FINITE_DIFFERENCES
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction
from gemseo.core.mdofunctions.mdo_quadratic_function import MDOQuadraticFunction
from gemseo.core.mdofunctions.norm_db_function import NormDBFunction
from gemseo.core.mdofunctions.norm_function import NormFunction
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.derivatives.complex_step import ComplexStep
from gemseo.utils.derivatives.finite_differences import FirstOrderFD
from gemseo.utils.hdf5 import get_hdf5_group
from gemseo.utils.python_compatibility import Final
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

LOGGER = logging.getLogger(__name__)

NAME_TO_METHOD = {
    "sum": aggregate_sum_square,
    "max": aggregate_max,
    "KS": aggregate_ks,
    "IKS": aggregate_iks,
}

BestInfeasiblePointType = Tuple[
    Optional[ndarray], Optional[ndarray], bool, Dict[str, ndarray]
]
OptimumType = Tuple[ndarray, ndarray, bool, Dict[str, ndarray], Dict[str, ndarray]]
OptimumSolutionType = Tuple[
    Optional[Sequence[ndarray]], ndarray, Dict[str, ndarray], Dict[str, ndarray]
]


class OptimizationProblem:
    """An optimization problem.

    Create an optimization problem from:

    - a :class:`.DesignSpace` specifying the design variables
      in terms of names, lower bounds, upper bounds and initial guesses,
    - the objective function as a :class:`.MDOFunction`,
      which can be a vector,

    execute it from an algorithm provided by a :class:`.DriverLib`,
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

    minimize_objective: bool
    """Whether to maximize the objective."""

    fd_step: float
    """The finite differences step."""

    pb_type: str
    """The type of optimization problem."""

    ineq_tolerance: float
    """The tolerance for the inequality constraints."""

    eq_tolerance: float
    """The tolerance for the equality constraints."""

    database: Database
    """The database to store the optimization problem data."""

    solution: OptimizationResult
    """The solution of the optimization problem."""

    design_space: DesignSpace
    """The design space on which the optimization problem is solved."""

    stop_if_nan: bool
    """Whether the optimization stops when a function returns ``NaN``."""

    preprocess_options: dict
    """The options to pre-process the functions."""

    use_standardized_objective: bool
    """Whether to use standardized objective for logging and post-processing.

    The standardized objective corresponds to the original one
    expressed as a cost function to minimize.
    A :class:`.DriverLib` works with this standardized objective
    and the :class:`.Database` stores its values.
    However, for convenience,
    it may be more relevant to log the expression
    and the values of the original objective.
    """

    constraint_names: dict[str, list[str]]
    """The standardized constraint names bound to the original ones."""

    LINEAR_PB: Final[str] = "linear"
    NON_LINEAR_PB: Final[str] = "non-linear"
    AVAILABLE_PB_TYPES: ClassVar[str] = [LINEAR_PB, NON_LINEAR_PB]

    USER_GRAD: Final[str] = "user"
    COMPLEX_STEP: Final[str] = COMPLEX_STEP
    FINITE_DIFFERENCES: Final[str] = FINITE_DIFFERENCES
    __DIFFERENTIATION_CLASSES: ClassVar[str] = {
        COMPLEX_STEP: ComplexStep,
        FINITE_DIFFERENCES: FirstOrderFD,
    }
    NO_DERIVATIVES: Final[str] = "no_derivatives"
    DIFFERENTIATION_METHODS: ClassVar[str] = [
        USER_GRAD,
        COMPLEX_STEP,
        FINITE_DIFFERENCES,
        NO_DERIVATIVES,
    ]
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
        pb_type: str = NON_LINEAR_PB,
        input_database: str | Path | Database | None = None,
        differentiation_method: str = USER_GRAD,
        fd_step: float = 1e-7,
        parallel_differentiation: bool = False,
        use_standardized_objective: bool = True,
        **parallel_differentiation_options: int | bool,
    ) -> None:
        """
        Args:
            design_space: The design space on which the functions are evaluated.
            pb_type: The type of the optimization problem
                among :attr:`.OptimizationProblem.AVAILABLE_PB_TYPES`.
            input_database: A database to initialize that of the optimization problem.
                If None, the optimization problem starts from an empty database.
            differentiation_method: The default differentiation method to be applied
                to the functions of the optimization problem.
            fd_step: The step to be used by the step-based differentiation methods.
            parallel_differentiation: Whether to approximate the derivatives in parallel.
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
        self.minimize_objective = True
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
        if isinstance(input_database, Database):
            self.database = input_database
        else:
            self.database = Database(input_hdf_file=input_database)
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
        self.constraint_names = {}
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
        if self.max_iter in [None, 0] or self.current_iter in [None, 0]:
            return False
        return self.current_iter >= self.max_iter

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
    def differentiation_method(self) -> str:
        """The differentiation method."""
        return self.__differentiation_method

    @differentiation_method.setter
    def differentiation_method(
        self,
        value: str,
    ) -> None:
        if not isinstance(value, str):
            value = value[0].decode()
        if value not in self.DIFFERENTIATION_METHODS:
            raise ValueError(
                "'{}' is not a differentiation methods; available ones are: '{"
                "}'.".format(value, "', '".join(self.DIFFERENTIATION_METHODS))
            )
        self.__differentiation_method = value

    @property
    def objective(self) -> MDOFunction:
        """The objective function."""
        return self._objective

    @objective.setter
    def objective(
        self,
        func: MDOFunction,
    ) -> None:
        self._objective = func

    @staticmethod
    def repr_constraint(
        func: MDOFunction,
        ctype: str,
        value: float | None = None,
        positive: bool = False,
    ) -> str:
        """Express a constraint as a string expression.

        Args:
            func: The constraint function.
            ctype: The type of the constraint.
                Either equality or inequality.
            value: The value for which the constraint is active.
                If None, this value is 0.
            positive: If True, then the inequality constraint is positive.

        Returns:
            A string representation of the constraint.
        """
        if value is None:
            value = 0.0
        str_repr = func.name
        if func.has_args():
            arguments = ", ".join(func.args)
            str_repr += f"({arguments})"

        if ctype == func.TYPE_EQ:
            sign = " == "
        elif positive:
            sign = " >= "
        else:
            sign = " <= "

        if func.has_expr():
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
        cstr_type: str | None = None,
        positive: bool = False,
    ) -> None:
        """Add a constraint (equality and inequality) to the optimization problem.

        Args:
            cstr_func: The constraint.
            value: The value for which the constraint is active.
                If None, this value is 0.
            cstr_type: The type of the constraint.
                Either equality or inequality.
            positive: If True, then the inequality constraint is positive.

        Raises:
            TypeError: When the constraint of a linear optimization problem
                is not an :class:`.MDOLinearFunction`.
            ValueError: When the type of the constraint is missing.
        """
        func_name = cstr_func.name
        has_default_name = cstr_func.has_default_name
        self.check_format(cstr_func)
        if self.pb_type == OptimizationProblem.LINEAR_PB and not isinstance(
            cstr_func, MDOLinearFunction
        ):
            raise TypeError(
                "The constraint of a linear optimization problem "
                "must be an MDOLinearFunction."
            )
        ctype = cstr_type or cstr_func.f_type
        cstr_repr = self.repr_constraint(cstr_func, ctype, value, positive)
        if value is not None:
            cstr_func = cstr_func.offset(-value)
        if positive:
            cstr_func = -cstr_func

        if cstr_type is not None:
            cstr_func.f_type = cstr_type
        else:
            if not cstr_func.is_constraint():
                msg = (
                    "Constraint type must be provided, "
                    "either in the function attributes or to the add_constraint method."
                )
                raise ValueError(msg)
        cstr_func.special_repr = cstr_repr
        self.constraints.append(cstr_func)
        if not has_default_name:
            cstr_func.name = func_name
            if cstr_func.outvars:
                cstr_repr = cstr_repr.replace(func_name, "#".join(cstr_func.outvars))
                cstr_func.special_repr = f"{func_name}: {cstr_repr}"

        if func_name not in self.constraint_names:
            self.constraint_names[func_name] = [cstr_func.name]
        else:
            self.constraint_names[func_name].append(cstr_func.name)

    def add_eq_constraint(
        self,
        cstr_func: MDOFunction,
        value: float | None = None,
    ) -> None:
        """Add an equality constraint to the optimization problem.

        Args:
            cstr_func: The constraint.
            value: The value for which the constraint is active.
                If None, this value is 0.
        """
        self.add_constraint(cstr_func, value, cstr_type=MDOFunction.TYPE_EQ)

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
                If None, this value is 0.
            positive: If True, then the inequality constraint is positive.
        """
        self.add_constraint(
            cstr_func, value, cstr_type=MDOFunction.TYPE_INEQ, positive=positive
        )

    def aggregate_constraint(
        self,
        constr_id: int,
        method: str | Callable[[Callable], Callable] = "max",
        groups: tuple[ndarray] | None = None,
        **options: Any,
    ):
        """Aggregates a constraint to generate a reduced dimension constraint.

        Args:
            constr_id: The index of the constraint in :attr:`.constraints`.
            method: The aggregation method, e.g. ``"max"``, ``"KS"`` or ``"IKS"``.
            groups: The groups for which to produce an output.
                If ``None``, a single output constraint is produced.
            **options: The options of the aggregation method.

        Raises:
            ValueError: When the given is index is greater or equal
                than the number of constraints
                or when the method is aggregation unknown.
        """
        if constr_id >= len(self.constraints):
            raise ValueError("constr_id must be lower than the number of constraints.")

        constr = self.constraints[constr_id]

        if callable(method):
            method_imp = method
        else:
            method_imp = NAME_TO_METHOD.get(method)
            if method_imp is None:
                raise ValueError(f"Unknown method {method}.")

        del self.constraints[constr_id]

        if groups is None:
            cstr = method_imp(constr, **options)
            self.constraints.insert(constr_id, cstr)
        else:
            cstrs = [method_imp(constr, indx, **options) for indx in groups]
            icstr = self.constraints[:constr_id]
            ecstr = self.constraints[constr_id:]
            self.constraints = icstr + cstrs + ecstr

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
            new_iter: If True, then the observable will be called at each new iterate.
        """
        name = obs_func.name
        if name in self.__observable_names:
            LOGGER.warning('The optimization problem already observes "%s".', name)
            return

        self.check_format(obs_func)
        obs_func.f_type = MDOFunction.TYPE_OBS
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
                A function.

            Returns:
                True if the function is an equality constraint.
            """
            return func.f_type == MDOFunction.TYPE_EQ

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
                A function.

            Returns:
                True if the function is an inequality constraint.
            """
            return func.f_type == MDOFunction.TYPE_INEQ

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

    def get_constraints_names(self) -> list[str]:
        """Retrieve the names of the constraints.

        Returns:
            The names of the constraints.
        """
        names = [constraint.name for constraint in self.constraints]
        return names

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
        return self.design_space.variables_names

    def get_all_functions(self) -> list[MDOFunction]:
        """Retrieve all the functions of the optimization problem.

        These functions are the constraints, the objective function and the observables.

        Returns:
            All the functions of the optimization problem.
        """
        return [self.objective] + self.constraints + self.observables

    def get_all_functions_names(self) -> list[str]:
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

    def has_constraints(self):
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

    def get_x0_normalized(self, cast_to_real: bool = False) -> ndarray:
        """Return the current values of the design variables after normalization.

        Args:
            cast_to_real: Whether to cast the return value to real.

        Returns:
            The current values of the design variables
            normalized between 0 and 1 from their lower and upper bounds.
        """
        dspace = self.design_space
        normalized_x0 = dspace.normalize_vect(dspace.get_current_value())
        if cast_to_real:
            return normalized_x0.real
        return normalized_x0

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
            TypeError: If the function is not a :class:`.MDOFunction`.
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
        return self.__count_cstr_total_dim(MDOFunction.TYPE_EQ)

    def get_ineq_cstr_total_dim(self) -> int:
        """Retrieve the total dimension of the inequality constraints.

        This dimension is the sum
        of all the outputs dimensions
        of all the inequality constraints.

        Returns:
            The total dimension of the inequality constraints.
        """
        return self.__count_cstr_total_dim(MDOFunction.TYPE_INEQ)

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
            if not constraint.has_dim():
                raise ValueError(
                    "Constraint dimension not available yet, "
                    "please call function {} once".format(constraint)
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
            func: atleast_1d(np_abs(func(x_vect)) <= tol)
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
            each_new_iter: If True, then callback at every iteration.
            each_store: If True,
                then callback at every call to :meth:`.Database.store`.
        """
        if each_store:
            self.database.add_store_listener(callback_func)
        if each_new_iter:
            self.database.add_new_iter_listener(callback_func)

    def clear_listeners(self) -> None:
        """Clear all the listeners."""
        self.database.clear_listeners()

    # TODO: API: set the default value of eval_observables to True.
    def evaluate_functions(
        self,
        x_vect: ndarray = None,
        eval_jac: bool = False,
        eval_obj: bool = True,
        eval_observables: bool = False,
        normalize: bool = True,
        no_db_no_norm: bool = False,
        constraints_names: Iterable[str] | None = None,
        observables_names: Iterable[str] | None = None,
        jacobians_names: Iterable[str] | None = None,
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
                If ``True`` and ``jacobians_names`` is ``None`` then
                compute the Jacobian matrices (or gradients) of the functions that are
                selected for evaluation (with ``eval_obj``, ``constraints_names``,
                ``eval_observables`` and``observables_names``).
                If ``False`` and ``jacobians_names`` is ``None`` then no Jacobian
                matrix is evaluated.
                If ``jacobians_names`` is not ``None`` then the value of
                ``eval_jac`` is ignored.
            eval_obj: Whether to consider the objective function
                as a function of interest.
            eval_observables: Whether to evaluate the observables.
                If ``True`` and ``observables_names`` is ``None`` then all the
                observables are evaluated.
                If ``False`` and ``observables_names`` is ``None`` then no observable
                is evaluated.
                If ``observables_names`` is not ``None`` then the value of
                ``eval_observables`` is ignored.
            normalize: Whether to consider the input vector ``x_vect`` normalized.
            no_db_no_norm: If True, then do not use the pre-processed functions,
                so we have no database, nor normalization.
            constraints_names: The names of the constraints to evaluate.
                If ``None`` then all the constraints are evaluated.
            observables_names: The names of the observables to evaluate.
                If ``None`` and ``eval_observables`` is ``True`` then all the
                observables are evaluated.
                If ``None`` and ``eval_observables`` is ``False`` then no observable is
                evaluated.
            jacobians_names: The names of the functions whose Jacobian matrices
                (or gradients) to compute.
                If ``None`` and ``eval_jac`` is ``True`` then
                compute the Jacobian matrices (or gradients) of the functions that are
                selected for evaluation (with ``eval_obj``, ``constraints_names``,
                ``eval_observables`` and``observables_names``).
                If ``None`` and ``eval_jac`` is ``False`` then no Jacobian matrix is
                computed.

        Returns:
            The output values of the functions of interest,
            as well as their Jacobian matrices if ``eval_jac`` is ``True``.

        Raises:
            ValueError: If a name in ``jacobians_names`` is not the name of
                a function of the problem.
        """
        # Get the functions to be evaluated
        from_original_functions = not self.__functions_are_preprocessed or no_db_no_norm
        functions = self.__get_functions(
            eval_obj,
            constraints_names,
            observables_names,
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
                except ValueError:
                    LOGGER.error("Failed to evaluate function %s", function.name)
                    raise

        if not eval_jac and jacobians_names is None:
            return outputs, {}

        # Evaluate the Jacobians
        if jacobians_names is not None:
            unknown_names = set(jacobians_names) - set(self.get_all_functions_names())
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
                self.objective.name in jacobians_names,
                [name for name in jacobians_names if name in self.constraint_names],
                [name for name in jacobians_names if name in self.__observable_names],
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
            except ValueError:
                LOGGER.error("Failed to evaluate Jacobian of %s.", function.name)
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
        constraints_names: Iterable[str] | None,
        observables_names: Iterable[str] | None,
        eval_observables: bool,
        from_original_functions: bool,
    ) -> list[MDOFunction]:
        """Return functions.

        Args:
            eval_obj: Whether to return the objective function.
            constraints_names: The names of the constraints to return.
                If ``None`` then all the constraints are evaluated.
            observables_names: The names of the observables to return.
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

        if constraints_names is not None:
            for name in constraints_names:
                functions.append(self.__get_constraint(name, from_original_functions))

        elif use_nonproc_functions:
            functions += self.nonproc_constraints
        else:
            functions += self.constraints

        if observables_names is not None:
            for name in observables_names:
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
        """
        if round_ints:
            # Keep the rounding option only if there is an integer design variable
            round_ints = any(
                np_any(var_type == DesignSpace.INTEGER)
                for var_type in self.design_space.variables_types.values()
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
                )

                p_obs.special_repr = obs.special_repr
                self.new_iter_observables[iobs] = p_obs

            # Preprocess the objective
            self.nonproc_objective = self.objective
            self.objective = self.__preprocess_func(
                self.objective,
                is_function_input_normalized=is_function_input_normalized,
                use_database=use_database,
                round_ints=round_ints,
            )
            self.objective.special_repr = self.objective.special_repr
            self.objective.f_type = MDOFunction.TYPE_OBJ
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
    ) -> MDOFunction:
        """Wrap the function to differentiate it and store its call in the database.

        Only the computed gradients are stored in the database,
        not the eventual finite differences or complex step perturbed evaluations.

        Args:
            func: The scaled and derived function to be pre-processed.
            is_function_input_normalized: Whether to consider the function input as
                normalized and unnormalize it before the evaluation takes place.
            use_database: If True, then the function is wrapped in the database.
            round_ints: If True, then round the integer variables.
            is_observable: If True, new_iter_listeners are not called
                when function is called (avoid recursive call)

        Returns:
            The pre-processed function.
        """
        self.check_format(func)
        # First differentiate it so that the finite differences evaluations
        # are not stored in the database, which would be the case in the other
        # way round
        # Also, store non normalized values in the database for further
        # exploitation
        if (
            isinstance(func, MDOLinearFunction)
            and not round_ints
            and is_function_input_normalized
        ):
            func = self.__normalize_linear_function(func)
        else:
            func = NormFunction(func, is_function_input_normalized, round_ints, self)

        if self.differentiation_method in self.__DIFFERENTIATION_CLASSES.keys():
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
            TypeError: If the original function is not a :class:`.MDOLinearFunction`.
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
        coefficients = multiply(orig_func.coefficients, norm_factors)
        value_at_zero = orig_func(shift)
        normalized_func = MDOLinearFunction(
            coefficients,
            orig_func.name,
            orig_func.f_type,
            orig_func.args,
            value_at_zero,
        )

        return normalized_func

    def __add_fd_jac(
        self,
        func: MDOFunction,
        normalize: bool,
    ) -> None:
        """Add a pointer to the approached Jacobian of the function.

        This Jacobian matrix is generated either by COMPLEX_STEP or FINITE_DIFFERENCES.

        Args:
            func: The function to be derivated.
            normalize: Whether to unnormalize the input vector of the function
                before evaluate it.

        Raises:
            ValueError: When the current value is not defined.
        """
        if not self.design_space.has_current_value():
            raise ValueError("The design space has no current value.")

        differentiation_class = self.__DIFFERENTIATION_CLASSES.get(
            self.differentiation_method
        )
        if differentiation_class is None:
            return

        differentiation_object = differentiation_class(
            func.evaluate,
            step=self.fd_step,
            parallel=self.__parallel_differentiation,
            design_space=self.design_space,
            normalize=normalize,
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
        self.__check_pb_type()
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
                    "Constraint type is not eq or ineq !, got {}"
                    " instead ".format(cstr.f_type)
                )
        self.check_format(self.objective)

    def __check_pb_type(self):
        """Check that the problem type is right.

        Available type are: :attr:`.OptimizationProblem.AVAILABLE_PB_TYPES`.

        Raises:
            ValueError: If a function declared as a constraint has the wrong type.
        """
        if self.pb_type not in self.AVAILABLE_PB_TYPES:
            raise TypeError(
                "Unknown problem type {}, "
                "available problem types are {}".format(
                    self.pb_type, self.AVAILABLE_PB_TYPES
                )
            )

    def __check_differentiation_method(self):
        """Check that the differentiation method is in allowed ones.

        Available ones are: :attr:`.OptimizationProblem.DIFFERENTIATION_METHODS`.

        Raises:
            ValueError: If either
                the differentiation method is unknown,
                the complex step is null or
                the finite differences' step is null.
        """
        if self.differentiation_method not in self.DIFFERENTIATION_METHODS:
            raise ValueError(
                "Unknown method {} "
                "is not among the supported ones: {}".format(
                    self.differentiation_method, self.DIFFERENTIATION_METHODS
                )
            )

        if self.differentiation_method == self.COMPLEX_STEP:
            if self.fd_step == 0:
                raise ValueError("ComplexStep step is null!")
            if self.fd_step.imag != 0:
                LOGGER.warning(
                    "Complex step method has an imaginary "
                    "step while required a pure real one."
                    " Auto setting the real part"
                )
                self.fd_step = self.fd_step.imag
        elif self.differentiation_method == self.FINITE_DIFFERENCES:
            if self.fd_step == 0:
                raise ValueError("Finite differences step is null!")
            if self.fd_step.imag != 0:
                LOGGER.warning(
                    "Finite differences method has a complex "
                    "step while required a pure real one."
                    " Auto setting the imaginary part to 0"
                )
                self.fd_step = self.fd_step.real

    def change_objective_sign(self) -> None:
        """Change the objective function sign in order to minimize its opposite.

        The :class:`.OptimizationProblem` expresses any optimization problem as a
        minimization problem. Then, an objective function originally expressed as a
        performance function to maximize must be converted into a cost function to
        minimize, by means of this method.
        """
        self.minimize_objective = not self.minimize_objective
        self.objective = -self.objective

    def _satisfied_constraint(
        self,
        cstr_type: str,
        value: ndarray,
    ) -> bool:
        """Determine if an evaluation satisfies a constraint within a given tolerance.

        Args:
            cstr_type: The type of the constraint.
            value: The value of the constraint.

        Returns:
            Whether a value satisfies a constraint.
        """
        if cstr_type == MDOFunction.TYPE_EQ:
            return np_all(np_abs(value) <= self.eq_tolerance)
        return np_all(value <= self.ineq_tolerance)

    def is_point_feasible(
        self,
        out_val: dict[str, ndarray],
        constraints: Iterable[MDOFunction] | None = None,
    ) -> bool:
        """Check if a point is feasible.

        Note:
            If the value of a constraint is absent from this point,
            then this constraint will be considered satisfied.

        Args:
            out_val: The values of the objective function, and eventually constraints.
            constraints: The constraints whose values are to be tested.
                If None, then take all constraints of the problem.

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

    def get_violation_criteria(
        self,
        x_vect: ndarray,
    ) -> tuple[bool, float]:
        """Compute a violation measure associated to an iteration.

        For each constraint,
        when it is violated,
        add the absolute distance to zero,
        in L2 norm.

        If 0, all constraints are satisfied

        Args:
            x_vect: The vector of the design variables values.

        Returns:
            The feasibility of the point and the violation measure.
        """
        f_violation = 0.0
        is_pt_feasible = True
        constraints = self.get_ineq_constraints() + self.get_eq_constraints()
        out_val = self.database.get(x_vect)
        for constraint in constraints:
            # look for the evaluation of the constraint
            eval_cstr = out_val.get(constraint.name, None)
            # if evaluation exists, check if it is satisfied
            if eval_cstr is None:
                break
            if not self._satisfied_constraint(constraint.f_type, eval_cstr):
                if isnan(eval_cstr).any():
                    return False, inf
                is_pt_feasible = False
                if constraint.f_type == MDOFunction.TYPE_INEQ:
                    if isinstance(eval_cstr, ndarray):
                        viol_inds = where(eval_cstr > self.ineq_tolerance)
                        f_violation += (
                            norm(eval_cstr[viol_inds] - self.ineq_tolerance) ** 2
                        )
                    else:
                        f_violation += (eval_cstr - self.ineq_tolerance) ** 2
                else:
                    f_violation += norm(abs(eval_cstr) - self.eq_tolerance) ** 2
        return is_pt_feasible, f_violation

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
        if isinstance(f_opt, ndarray):
            if len(f_opt) == 1:
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
            + " The least infeasible point is selected."
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
        for (i, out_val) in enumerate(feas_f):
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
        if isinstance(f_opt, ndarray):
            if len(f_opt) == 1:
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

    def __repr__(self) -> str:
        msg = MultiLineString()
        msg.add("Optimization problem:")
        msg.indent()

        # objective representation
        if self.minimize_objective or self.use_standardized_objective:
            optimize_verb = "minimize "
            start = 0
        else:
            optimize_verb = "maximize "
            start = 1

        objective_function = [line for line in repr(self.objective).split("\n") if line]
        msg.add(optimize_verb + objective_function[0][start:])
        for line in objective_function[1:]:
            msg.add(" " * len(optimize_verb) + line)

        # variables representation
        msg.add("with respect to {}", pretty_str(self.design_space.variables_names))
        if self.has_constraints():
            msg.add("subject to constraints:")
            msg.indent()
            for constraints in self.get_ineq_constraints():
                constraints = [cstr for cstr in str(constraints).split("\n") if cstr]
                for constraint in constraints:
                    msg.add(constraint)
            for constraints in self.get_eq_constraints():
                constraints = [cstr for cstr in str(constraints).split("\n") if cstr]
                for constraint in constraints:
                    msg.add(constraint)
        return str(msg)

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
            dtype: Numpy dtype or string. If None, dtype('f') will be used.
        """
        if data_array is None or data_array == []:
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
    def __store_attr_h5data(
        cls,
        obj: Any,
        group: str,
    ) -> None:
        """Store an object that has a to_dict attribute in the hdf5 dataset.

        Args:
            obj: The object to store
            group: The hdf5 group.
        """
        data_dict = obj.to_dict()
        for attr_name, attr in data_dict.items():
            dtype = None
            is_arr_n = isinstance(attr, ndarray) and issubdtype(attr.dtype, np_number)
            if isinstance(attr, str):
                attr = attr.encode("ascii", "ignore")
            elif isinstance(attr, bytes):
                attr = attr.decode()
            elif hasattr(attr, "__iter__") and not is_arr_n:
                attr = [
                    att.encode("ascii", "ignore") if isinstance(att, str) else att
                    for att in attr
                ]
                dtype = h5py.special_dtype(vlen=str)
            cls.__store_h5data(group, attr, attr_name, dtype)

    def export_hdf(
        self,
        file_path: str | Path,
        append: bool = False,
    ) -> None:
        """Export the optimization problem to an HDF file.

        Args:
            file_path: The path of the file to store the data.
            append: If True, then the data are appended to the file if not empty.
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
                    sol_group = h5file.require_group(self.SOLUTION_GROUP)
                    self.__store_attr_h5data(self.solution, sol_group)

        self.database.export_hdf(file_path, append=True)

        # Design space shall remain the same in append mode
        if not append or no_design_space:
            self.design_space.export_hdf(file_path, append=True)

    @classmethod
    def import_hdf(
        cls,
        file_path: str | Path,
        x_tolerance: float = 0.0,
    ) -> OptimizationProblem:
        """Import an optimization history from an HDF file.

        Args:
            file_path: The file containing the optimization history.
            x_tolerance: The tolerance on the design variables when reading the file.

        Returns:
            The read optimization problem.
        """
        LOGGER.info("Import optimization problem from file: %s", file_path)

        design_space = DesignSpace(file_path)
        opt_pb = OptimizationProblem(design_space, input_database=file_path)

        with h5py.File(file_path) as h5file:
            group = get_hdf5_group(h5file, opt_pb.OPT_DESCR_GROUP)

            for attr_name, attr in group.items():
                val = attr[()]
                val = val.decode() if isinstance(val, bytes) else val
                setattr(opt_pb, attr_name, val)

            if opt_pb.SOLUTION_GROUP in h5file:
                group_data = cls.__h5_group_to_dict(h5file, opt_pb.SOLUTION_GROUP)
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

            if opt_pb.CONSTRAINTS_GROUP in h5file:
                group = get_hdf5_group(h5file, opt_pb.CONSTRAINTS_GROUP)

                for cstr_name in group.keys():
                    group_data = cls.__h5_group_to_dict(group, cstr_name)
                    attr = MDOFunction.init_from_dict_repr(**group_data)
                    opt_pb.constraints.append(attr)

            if opt_pb.OBSERVABLES_GROUP in h5file:
                group = get_hdf5_group(h5file, opt_pb.OBSERVABLES_GROUP)

                for observable_name in group.keys():
                    group_data = cls.__h5_group_to_dict(group, observable_name)
                    attr = MDOFunction.init_from_dict_repr(**group_data)
                    opt_pb.observables.append(attr)

        return opt_pb

    def export_to_dataset(
        self,
        name: str | None = None,
        by_group: bool = True,
        categorize: bool = True,
        opt_naming: bool = True,
        export_gradients: bool = False,
        input_values: Iterable[ndarray] | None = None,
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
                If ``None``, use the name of the :attr:`.OptimizationProblem.database`.
            by_group: Whether to store the data by group in :attr:`.Dataset.data`,
                in the sense of one unique NumPy array per group.
                If ``categorize`` is ``False``,
                there is a unique group: :attr:`.Dataset.PARAMETER_GROUP``.
                If ``categorize`` is ``True``,
                the groups can be either
                :attr:`.Dataset.DESIGN_GROUP` and :attr:`.Dataset.FUNCTION_GROUP`
                if ``opt_naming`` is ``True``,
                or :attr:`.Dataset.INPUT_GROUP` and :attr:`.Dataset.OUTPUT_GROUP`.
                If ``by_group`` is ``False``, store the data by variable names.
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
                If ``None``, consider all the input values of the database.

        Returns:
            A dataset built from the database of the optimization problem.
        """
        if name is None:
            name = self.database.name

        dataset = Dataset(name, by_group)

        # Set the different groups
        input_group = output_group = gradient_group = dataset.DEFAULT_GROUP
        cache_output_as_input = True
        if categorize:
            if opt_naming:
                input_group = dataset.DESIGN_GROUP
                output_group = dataset.FUNCTION_GROUP
                gradient_group = dataset.GRADIENT_GROUP
            else:
                input_group = dataset.INPUT_GROUP
                output_group = dataset.OUTPUT_GROUP
                gradient_group = dataset.GRADIENT_GROUP
            cache_output_as_input = False

        # Add database inputs
        input_names = self.design_space.variables_names
        names_to_sizes = self.design_space.variables_sizes
        input_history = array(self.database.get_x_history())
        n_samples = len(input_history)
        positions = []
        if input_values is not None:
            for input_value in input_values:
                positions.extend(
                    where((input_history == input_value).all(axis=1))[0].tolist()
                )

        input_history = split_array_to_dict_of_arrays(
            input_history, names_to_sizes, input_names
        )
        for input_name, input_value in sorted(input_history.items()):
            dataset.add_variable(input_name, input_value.real, input_group)

        # Add database outputs
        variable_names = self.database.get_all_data_names(skip_iter=True)
        output_names = [name for name in variable_names if name not in input_names]

        self.__add_database_outputs(
            dataset,
            output_names,
            n_samples,
            output_group,
            cache_output_as_input,
        )

        # Add database output gradients
        if export_gradients:
            self.__add_database_output_gradients(
                dataset,
                output_names,
                n_samples,
                gradient_group,
                cache_output_as_input,
            )

        if positions:
            dataset.data = {
                name: value[positions, :] for name, value in dataset.data.items()
            }
            dataset.length = len(positions)

        return dataset

    def __add_database_outputs(
        self,
        dataset: Dataset,
        output_names: Iterable[str],
        n_samples: int,
        output_group: str,
        cache_output_as_input: bool,
    ) -> None:
        """Add the database outputs to the dataset.

        Args:
            dataset: The dataset where the outputs will be added.
            output_names: The names of the outputs in the database.
            n_samples: The total number of samples, including possible
                points where the evaluation failed.
            output_group: The dataset group where the variables will
                be added.
            cache_output_as_input: If True, cache these data as inputs
                when the cache is exported to a cache.
        """
        for output_name in output_names:
            output_history, input_history = self.database.get_func_history(
                output_name, x_hist=True
            )
            output_history = self.__replace_missing_values(
                output_history, input_history, array(self.database.get_x_history())
            )

            dataset.add_variable(
                output_name,
                output_history.reshape((n_samples, -1)).real,
                output_group,
                cache_as_input=cache_output_as_input,
            )

    def __add_database_output_gradients(
        self,
        dataset: Dataset,
        output_names: Iterable[str],
        n_samples: int,
        gradient_group: str,
        cache_output_as_input: bool,
    ) -> None:
        """Add the database output gradients to the dataset.

        Args:
            dataset: The dataset where the outputs will be added.
            output_names: The names of the outputs in the database.
            n_samples: The total number of samples, including possible
                points where the evaluation failed.
            gradient_group: The dataset group where the variables will
                be added.
            cache_output_as_input: If True, cache these data as inputs
                when the cache is exported to a cache.
        """
        for output_name in output_names:
            if self.database.is_func_grad_history_empty(output_name):
                continue

            gradient_history, input_history = self.database.get_func_grad_history(
                output_name, x_hist=True
            )
            gradient_history = self.__replace_missing_values(
                gradient_history,
                input_history,
                array(self.database.get_x_history()),
            )

            dataset.add_variable(
                Database.get_gradient_name(output_name),
                gradient_history.reshape(n_samples, -1).real,
                gradient_group,
                cache_as_input=cache_output_as_input,
            )

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
        else:
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
            if isinstance(value, ndarray) and value.dtype.type in (
                numpy.object_,
                numpy.string_,
            ):
                if value.size == 1:
                    value = value[0]
                else:
                    value = value.tolist()

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
            as_dict: If True, return values as dictionary.
            filter_non_feasible: If True, remove the non-feasible points from
                the data.

        Returns:
            The data related to the variables.
        """
        dataset = self.export_to_dataset("OptimizationProblem")
        data = dataset.get_data_by_names(names, as_dict)

        if filter_non_feasible:
            x_feasible, _ = self.get_feasible_points()
            feasible_indexes = [self.database.get_index_of(x) for x in x_feasible]
            if as_dict:
                for key, value in data.items():
                    data[key] = value[feasible_indexes, :]
            else:
                data = data[feasible_indexes, :]

        return data

    @property
    def is_mono_objective(self) -> bool:
        """Whether the optimization problem is mono-objective."""
        return len(self.objective.outvars) == 1

    def get_functions_dimensions(
        self, names: Iterable[str] | None = None
    ) -> dict[str, int]:
        """Return the dimensions of the outputs of the problem functions.

        Args:
            names: The names of the functions.
                If None, then the objective and all the constraints are considered.

        Returns:
            The dimensions of the outputs of the problem functions.
            The dictionary keys are the functions names
            and the values are the functions dimensions.
        """
        if names is None:
            names = [self.objective.name] + self.get_constraints_names()

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
        if function.has_dim():
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
    ) -> int:
        """Return the number of scalar constraints not satisfied by design variables.

        Args:
            design_variables: The design variables.

        Returns:
            The number of unsatisfied scalar constraints.
        """
        n_unsatisfied = 0
        values, _ = self.evaluate_functions(
            design_variables, eval_obj=False, normalize=False
        )
        for constraint in self.constraints:
            value = atleast_1d(values[constraint.name])
            if constraint.f_type == MDOFunction.TYPE_EQ:
                value = numpy.absolute(value)
                tolerance = self.eq_tolerance
            else:
                tolerance = self.ineq_tolerance
            n_unsatisfied += sum(value > tolerance)
        return n_unsatisfied

    def get_scalar_constraints_names(self) -> list[str]:
        """Return the names of the scalar constraints.

        Returns:
            The names of the scalar constraints.
        """
        constraints_names = list()
        for constraint in self.constraints:
            dimension = self.get_function_dimension(constraint.name)
            if dimension == 1:
                constraints_names.append(constraint.name)
            else:
                constraints_names.extend(
                    [constraint.get_indexed_name(index) for index in range(dimension)]
                )
        return constraints_names

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
            self.database.clear(current_iter)

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
            self.get_constraints_names(),
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
