# -*- coding: utf-8 -*-
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
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""Optimization problem.

The :class:`.OptimizationProblem` class operates on a :class:`.DesignSpace` defining:

- an initial guess :math:`x_0` for the design variables,
- the bounds :math:`l_b \\leq x \\leq u_b` of the design variables.

A (possible vector) objective function with a :class:`.MDOFunction` type
is set using the :code:`objective` attribute.
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
Concerning the derivatives computation,
the :class:`.OptimizationProblem` automates
the generation of the finite differences or complex step wrappers on functions,
when the analytical gradient is not available.

Lastly,
various getters and setters are available,
as well as methods to export the :class:`.Database`
to a HDF file or to a :class:`.Dataset` for future post-processing.
"""
from __future__ import division, unicode_literals

import logging
from functools import reduce
from numbers import Number
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import h5py
import numpy
from numpy import abs as np_abs
from numpy import all as np_all
from numpy import any as np_any
from numpy import (
    argmin,
    array,
    array_equal,
    concatenate,
    inf,
    insert,
    issubdtype,
    multiply,
    nan,
    ndarray,
)
from numpy import number as np_number
from numpy import where
from numpy.core import atleast_1d
from numpy.linalg import norm

from gemseo.algos.aggregation.aggregation_func import (
    aggregate_iks,
    aggregate_ks,
    aggregate_max,
    aggregate_sum_square,
)
from gemseo.algos.database import Database
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.dataset import Dataset
from gemseo.core.mdofunctions.mdo_function import (
    MDOFunction,
    MDOLinearFunction,
    MDOQuadraticFunction,
)
from gemseo.core.mdofunctions.norm_db_function import NormDBFunction
from gemseo.core.mdofunctions.norm_function import NormFunction
from gemseo.utils.data_conversion import DataConversion
from gemseo.utils.derivatives_approx import ComplexStep, FirstOrderFD
from gemseo.utils.hdf5 import get_hdf5_group
from gemseo.utils.py23_compat import PY3, string_array, string_types
from gemseo.utils.string_tools import MultiLineString, pretty_repr

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


class OptimizationProblem(object):
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

    Attributes:
        nonproc_objective (MDOFunction): The non-processed objective function.
        constraints (List(MDOFunction)): The constraints.
        nonproc_constraints (List(MDOFunction)): The non-processed constraints.
        observables (List(MDOFunction)): The observables.
        new_iter_observables (List(MDOFunction)): The observables to be called
            at each new iterate.
        nonproc_observables (List(MDOFunction)): The non-processed observables.
        nonproc_new_iter_observables (List(MDOFunction)): The non-processed observables
            to be called at each new iterate.
        minimize_objective (bool): If True, maximize the objective.
        fd_step (float): The finite differences step.
        pb_type (str): The type of optimization problem.
        ineq_tolerance (float): The tolerance for the inequality constraints.
        eq_tolerance (float): The tolerance for the equality constraints.
        database (Database): The database to store the optimization problem data.
        solution: The solution of the optimization problem.
        design_space (DesignSpace): The design space on which the optimization problem
            is solved.
        stop_if_nan (bool): If True, the optimization stops when a function returns NaN.
        preprocess_options (Dict): The options to pre-process the functions.
    """

    LINEAR_PB = "linear"
    NON_LINEAR_PB = "non-linear"
    AVAILABLE_PB_TYPES = [LINEAR_PB, NON_LINEAR_PB]

    USER_GRAD = "user"
    COMPLEX_STEP = "complex_step"
    FINITE_DIFFERENCES = "finite_differences"
    __DIFFERENTIATION_CLASSES = {
        COMPLEX_STEP: ComplexStep,
        FINITE_DIFFERENCES: FirstOrderFD,
    }
    NO_DERIVATIVES = "no_derivatives"
    DIFFERENTIATION_METHODS = [
        USER_GRAD,
        COMPLEX_STEP,
        FINITE_DIFFERENCES,
        NO_DERIVATIVES,
    ]
    DESIGN_VAR_NAMES = "x_names"
    DESIGN_VAR_SIZE = "x_size"
    DESIGN_SPACE_ATTRS = ["u_bounds", "l_bounds", "x_0", DESIGN_VAR_NAMES, "dimension"]
    FUNCTIONS_ATTRS = ["objective", "constraints"]
    OPTIM_DESCRIPTION = [
        "minimize_objective",
        "fd_step",
        "differentiation_method",
        "pb_type",
        "ineq_tolerance",
        "eq_tolerance",
    ]

    OPT_DESCR_GROUP = "opt_description"
    DESIGN_SPACE_GROUP = "design_space"
    OBJECTIVE_GROUP = "objective"
    SOLUTION_GROUP = "solution"
    CONSTRAINTS_GROUP = "constraints"

    HDF5_FORMAT = "hdf5"
    GGOBI_FORMAT = "ggobi"

    def __init__(
        self,
        design_space,  # type: DesignSpace
        pb_type=NON_LINEAR_PB,  # type: str
        input_database=None,  # type: Optional[Union[str,Database]]
        differentiation_method=USER_GRAD,  # type: str
        fd_step=1e-7,  # type: float
        parallel_differentiation=False,  # type: bool
        **parallel_differentiation_options  # type: Union[int,bool]
    ):  # type: (...) -> None
        # noqa: D205, D212, D415
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
            **parallel_differentiation_options: The options
                to approximate the derivatives in parallel.
        """
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
        self.max_iter = None
        self.current_iter = 0
        self.__functions_are_preprocessed = False
        if isinstance(input_database, Database):
            self.database = input_database
        else:
            self.database = Database(input_hdf_file=input_database)
        self.solution = None
        self.design_space = design_space
        self.__x0 = None
        self.stop_if_nan = True
        self.preprocess_options = {}
        self.__parallel_differentiation = parallel_differentiation
        self.__parallel_differentiation_options = parallel_differentiation_options

    def __raise_exception_if_functions_are_already_preprocessed(self):
        """Raise an exception if the function have already been pre-processed."""
        if self.__functions_are_preprocessed:
            raise RuntimeError(
                "The parallel differentiation cannot be changed "
                "because the functions have already been pre-processed."
            )

    def is_max_iter_reached(self):  # type: (...) -> bool
        """Check if the maximum amount of iterations has been reached.

        Returns:
            Whether the maximum amount of iterations has been reached.
        """
        if self.max_iter is None or self.current_iter is None:
            return False
        return self.current_iter >= self.max_iter

    @property
    def parallel_differentiation(self):  # type: (...) -> bool
        """Whether to approximate the derivatives in parallel."""
        return self.__parallel_differentiation

    @parallel_differentiation.setter
    def parallel_differentiation(
        self,
        value,  # type: bool
    ):  # type: (...) -> None
        self.__raise_exception_if_functions_are_already_preprocessed()
        self.__parallel_differentiation = value

    @property
    def parallel_differentiation_options(self):  # type: (...) -> bool
        """The options to approximate the derivatives in parallel."""
        return self.__parallel_differentiation_options

    @parallel_differentiation_options.setter
    def parallel_differentiation_options(
        self,
        value,  # type: bool
    ):  # type: (...) -> None
        self.__raise_exception_if_functions_are_already_preprocessed()
        self.__parallel_differentiation_options = value

    @property
    def differentiation_method(self):  # type: (...) -> str
        """The differentiation method."""
        return self.__differentiation_method

    @differentiation_method.setter
    def differentiation_method(
        self,
        value,  # type: str
    ):  # type: (...) -> None
        if not isinstance(value, string_types):
            value = value[0].decode()
        if value not in self.DIFFERENTIATION_METHODS:
            raise ValueError(
                "'{}' is not a differentiation methods; available ones are: '{}'.".format(
                    value, "', '".join(self.DIFFERENTIATION_METHODS)
                )
            )
        self.__differentiation_method = value

    @property
    def objective(self):  # type: (...) -> MDOFunction
        """The objective function."""
        return self._objective

    @objective.setter
    def objective(
        self,
        func,  # type: MDOFunction
    ):  # type: (...) -> None
        self._objective = func

    @staticmethod
    def repr_constraint(
        func,  # type: MDOFunction
        ctype,  # type: str
        value=None,  # type: Optional[float]
        positive=False,  # type: bool
    ):  # type: (...) -> str
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
            str_repr += "({})".format(arguments)

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
        cstr_func,  # type: MDOFunction
        value=None,  # type: Optional[value]
        cstr_type=None,  # type: Optional[str]
        positive=False,  # type: bool
    ):  # type: (...) -> None
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
                is not an :class:`MDOLinearFunction`.
            ValueError: When the type of the constraint is missing.
        """
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

    def add_eq_constraint(
        self,
        cstr_func,  # type: MDOFunction
        value=None,  # type: Optional[float]
    ):  # type: (...) -> None
        """Add an equality constraint to the optimization problem.

        Args:
            cstr_func: The constraint.
            value: The value for which the constraint is active.
                If None, this value is 0.
        """
        self.add_constraint(cstr_func, value, cstr_type=MDOFunction.TYPE_EQ)

    def add_ineq_constraint(
        self,
        cstr_func,  # type: MDOFunction
        value=None,  # type: Optional[value]
        positive=False,  # type: bool
    ):  # type: (...) -> None
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

    def aggregate_constraint(self, constr_id, method="max", groups=None, **options):
        """Aggregates a constraint to generate a reduced dimension constraint.

        :param constr_id: index of the constraint in self.constraints
        :type constr_id: int
        :param method: aggregation method, among ('max','KS', 'IKS')
        :type method: str or callable, that takes a function and returns a function
        :param groups: if None, a single output constraint is produced
            otherwise, one output per group is produced.
        :type groups: tuple of ndarray
        """
        if constr_id >= len(self.constraints):
            raise ValueError("constr_id must be lower than the number of constraints.")

        constr = self.constraints[constr_id]

        if callable(method):
            method_imp = method
        else:
            method_imp = NAME_TO_METHOD.get(method)
            if method_imp is None:
                raise ValueError("Unknown method {}".format(method))

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
        obs_func,  # type: MDOFunction
        new_iter=True,  # type: bool
    ):  # type: (...) -> None
        """Add a function to be observed.

        Args:
            obs_func: An observable to be observed.
            new_iter: If True, then the observable will be called at each new iterate.
        """
        self.check_format(obs_func)
        obs_func.f_type = MDOFunction.TYPE_OBS
        self.observables.append(obs_func)
        if new_iter:
            self.new_iter_observables.append(obs_func)

    def get_eq_constraints(self):  # type: (...) -> List[MDOFunction]
        """Retrieve all the equality constraints.

        Returns:
            The equality constraints.
        """

        def is_equality_constraint(
            func,  # type: MDOFunction
        ):  # type: (...) -> bool
            """Check if a function is an equality constraint.

            Args:
                A function.

            Returns:
                True if the function is an equality constraint.
            """
            return func.f_type == MDOFunction.TYPE_EQ

        return list(filter(is_equality_constraint, self.constraints))

    def get_ineq_constraints(self):  # type: (...) -> List[MDOFunction]
        """Retrieve all the inequality constraints.

        Returns:
            The inequality constraints.
        """

        def is_inequality_constraint(
            func,  # type: MDOFunction
        ):  # type: (...) -> bool
            """Check if a function is an inequality constraint.

            Args:
                A function.

            Returns:
                True if the function is an inequality constraint.
            """
            return func.f_type == MDOFunction.TYPE_INEQ

        return list(filter(is_inequality_constraint, self.constraints))

    def get_observable(
        self,
        name,  # type: str
    ):  # type: (...) -> MDOFunction
        """Retrieve an observable from its name.

        Args:
            name: The name of the observable.

        Returns:
            The observable.

        Raises:
            ValueError: If the observable cannot be found.
        """
        try:
            observable = next(obs for obs in self.observables if obs.name == name)
        except StopIteration:
            raise ValueError("Observable {} cannot be found.".format(name))

        return observable

    def get_ineq_constraints_number(self):  # type: (...) -> int
        """Retrieve the number of inequality constraints.

        Returns:
            The number of inequality constraints.
        """
        return len(self.get_ineq_constraints())

    def get_eq_constraints_number(self):  # type: (...) -> int
        """Retrieve the number of equality constraints.

        Returns:
            The number of equality constraints.
        """
        return len(self.get_eq_constraints())

    def get_constraints_number(self):  # type: (...) -> int
        """Retrieve the number of constraints.

        Returns:
            The number of constraints.
        """
        return len(self.constraints)

    def get_constraints_names(self):  # type: (...) -> List[str]
        """Retrieve the names of the constraints.

        Returns:
            The names of the constraints.
        """
        names = [constraint.name for constraint in self.constraints]
        return names

    def get_nonproc_constraints(self):  # type: (...) -> List[MDOFunction]
        """Retrieve the non-processed constraints.

        Returns:
            The non-processed constraints.
        """
        return self.nonproc_constraints

    def get_design_variable_names(self):  # type: (...) -> List[str]
        """Retrieve the names of the design variables.

        Returns:
            The names of the design variables.
        """
        return self.design_space.variables_names

    def get_all_functions(self):  # type: (...) -> List[MDOFunction]
        """Retrieve all the functions of the optimization problem.

        These functions are the constraints, the objective function and the observables.

        Returns:
            All the functions of the optimization problem.
        """
        return [self.objective] + self.constraints + self.observables

    def get_all_functions_names(self):  # type: (...) -> List[str]
        """Retrieve the names of all the function of the optimization problem.

        These functions are the constraints, the objective function and the observables.

        Returns:
            The names of all the functions of the optimization problem.
        """
        return [func.name for func in self.get_all_functions()]

    def get_objective_name(self):  # type: (...) -> str
        """Retrieve the name of the objective function.

        Returns:
            The name of the objective function.
        """
        return self.objective.name

    def get_nonproc_objective(self):  # type: (...) -> MDOFunction
        """Retrieve the non-processed objective function."""
        return self.nonproc_objective

    def has_nonlinear_constraints(self):  # type: (...) -> bool
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

    def has_eq_constraints(self):  # type: (...) -> bool
        """Check if the problem has equality constraints.

        Returns:
            True if the problem has equality constraints.
        """
        return len(self.get_eq_constraints()) > 0

    def has_ineq_constraints(self):  # type: (...) -> bool
        """Check if the problem has inequality constraints.

        Returns:
            True if the problem has inequality constraints.
        """
        return len(self.get_ineq_constraints()) > 0

    def get_x0_normalized(self):  # type: (...) -> ndarray
        """Return the current values of the design variables after normalization.

        Returns:
            The current values of the design variables
            normalized between 0 and 1 from their lower and upper bounds.
        """
        dspace = self.design_space
        return dspace.normalize_vect(dspace.get_current_x())

    def get_dimension(self):  # type: (...) -> int
        """Retrieve the total number of design variables.

        Returns:
            The dimension of the design space.
        """
        return self.design_space.dimension

    @property
    def dimension(self):  # type: (...) -> int
        """The dimension of the design space."""
        return self.design_space.dimension

    @staticmethod
    def check_format(input_function):  # type: (...) -> None
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

    def get_eq_cstr_total_dim(self):  # type: (...) -> int
        """Retrieve the total dimension of the equality constraints.

        This dimension is the sum
        of all the outputs dimensions
        of all the equality constraints.

        Returns:
            The total dimension of the equality constraints.
        """
        return self.__count_cstr_total_dim(MDOFunction.TYPE_EQ)

    def get_ineq_cstr_total_dim(self):  # type: (...) -> int
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
        cstr_type,  # type: str
    ):  # type: (...) -> int
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
        x_vect,  # type: ndarray
        tol=1e-6,  # type: float
    ):  # type: (...) -> Dict[str,ndarray]
        """For each constraint, indicate if its different components are active.

        Args:
            x_vect: The vector of design variables.
            tol: The tolerance for deciding whether a constraint is active.

        Returns:
            For each constraint,
            a boolean indicator of activation of its different components.
        """
        self.design_space.check_membership(x_vect)
        normalize = self.preprocess_options.get("normalize", False)
        if normalize:
            x_vect = self.design_space.normalize_vect(x_vect)

        act_funcs = {}
        for func in self.get_ineq_constraints():
            val = np_abs(func(x_vect))
            act_funcs[func] = where(val <= tol, True, False)

        return act_funcs

    def add_callback(
        self,
        callback_func,  # type: Callable
        each_new_iter=True,  # type: bool
        each_store=False,  # type: bool
    ):  # type: (...) -> None
        """Add a callback function after each store operation or new iteration.

        Args:
            callback_func: A function to be called after some event.
            each_new_iter: If True, then callback at every iteration.
            each_store: If True,
                then callback at every call to :class:`.Database.store`.
        """
        if each_store:
            self.database.add_store_listener(callback_func)
        if each_new_iter:
            self.database.add_new_iter_listener(callback_func)

    def clear_listeners(self):  # type: (...) -> None
        """Clear all the listeners."""
        self.database.clear_listeners()

    def evaluate_functions(
        self,
        x_vect=None,  # type: ndarray
        eval_jac=False,  # type: bool
        eval_obj=True,  # type:bool
        normalize=True,  # type:bool
        no_db_no_norm=False,  # type:bool
    ):  # type: (...) -> Tuple[Dict[str,Union[float,ndarray]],Dict[str,ndarray]]
        """Compute the objective and the constraints.

        Some optimization libraries require the number of constraints
        as an input parameter which is unknown by the formulation or the scenario.
        Evaluation of initial point allows to get this mandatory information.
        This is also used for design of experiments to evaluate samples.

        Args:
            x_vect: The input vector at which the functions must be evaluated;
                if None, x_0 is used.
            eval_jac: If True, then the Jacobian is evaluated
            eval_obj: If True, then the objective function is evaluated
            normalize: If True, then input vector is considered normalized
            no_db_no_norm: If True, then do not use the pre-processed functions,
                so we have no database, nor normalization.

        Returns:
            The functions values and/or the Jacobian values
            according to the passed arguments.

        Raises:
            ValueError: If both no_db_no_norm and normalize are True.
        """
        if no_db_no_norm and normalize:
            raise ValueError("Can't use no_db_no_norm and normalize options together")
        if normalize:
            if x_vect is None:
                x_vect = self.get_x0_normalized()
            else:
                # Checks proposed x wrt bounds
                x_u_r = self.design_space.unnormalize_vect(x_vect)
                self.design_space.check_membership(x_u_r)
        else:
            if x_vect is None:
                x_vect = self.design_space.get_current_x()
            else:
                # Checks proposed x wrt bounds
                self.design_space.check_membership(x_vect)

        if no_db_no_norm:
            if eval_obj:
                functions = self.nonproc_constraints + [self.nonproc_objective]
            else:
                functions = self.nonproc_constraints
        else:
            if eval_obj:
                functions = self.constraints + [self.objective]
            else:
                functions = self.constraints

        outputs = {}
        for func in functions:
            try:
                outputs[func.name] = func(x_vect)
            except ValueError:
                LOGGER.error("Failed to evaluate function %s", func.name)
                raise

        jacobians = {}
        if eval_jac:
            for func in functions:
                try:
                    jacobians[func.name] = func.jac(x_vect)
                except ValueError:
                    msg = "Failed to evaluate jacobian of {}".format(func.name)
                    LOGGER.error(msg)
                    raise

        return outputs, jacobians

    def preprocess_functions(
        self,
        normalize=True,  # type: bool
        use_database=True,  # type: bool
        round_ints=True,  # type: bool
    ):  # type: (...) -> None
        """Pre-process all the functions and eventually the gradient.

        Required to wrap the objective function and constraints with the database
        and eventually the gradients by complex step or finite differences.

        Args:
            normalize: Whether to unnormalize the input vector of the function
                before evaluate it.
            use_database: If True, then the functions are wrapped in the database.
            round_ints: If True, then round the integer variables.
        """
        if round_ints:
            # Keep the rounding option only if there is an integer design variable
            round_ints = any(
                (
                    np_any(var_type == DesignSpace.INTEGER)
                    for var_type in self.design_space.variables_types.values()
                )
            )
        # Avoids multiple wrappings of functions when multiple executions
        # are performed, in bi level scenarios for instance
        if not self.__functions_are_preprocessed:
            self.preprocess_options = {
                "normalize": normalize,
                "use_database": use_database,
                "round_ints": round_ints,
            }
            # Preprocess the constraints
            for icstr, cstr in enumerate(self.constraints):
                self.nonproc_constraints.append(cstr)
                p_cstr = self.__preprocess_func(
                    cstr,
                    normalize=normalize,
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
                    normalize=normalize,
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
                    normalize=normalize,
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
                normalize=normalize,
                use_database=use_database,
                round_ints=round_ints,
            )
            self.objective.special_repr = self.objective.special_repr
            self.objective.f_type = MDOFunction.TYPE_OBJ
            self.__functions_are_preprocessed = True
            self.check()

            self.database.add_new_iter_listener(self.execute_observables_callback)

    def execute_observables_callback(
        self, last_x  # type: ndarray
    ):  # type: (...)-> None
        """The callback function to be passed to the database.

        Call all the observables with the last design variables values as argument.

        Args:
            last_x: The design variables values from the last evaluation.
        """
        if not self.new_iter_observables:
            return

        if self.preprocess_options["normalize"]:
            last_x = self.design_space.normalize_vect(last_x)
        for func in self.new_iter_observables:
            func(last_x)

    def __preprocess_func(
        self,
        function,  # type: MDOFunction
        normalize=True,  # type: bool
        use_database=True,  # type: bool
        round_ints=True,  # type: bool
        is_observable=False,  # type: bool
    ):  # type: (...) -> MDOFunction
        """Wrap the function to differentiate it and store its call in the database.

        Only the computed gradients are stored in the database,
        not the eventual finite differences or complex step perturbed evaluations.

        Args:
            function: The scaled and derived function to be pre-processed.
            normalize: Whether to unnormalize the input vector of the function
                before evaluate it.
            use_database: If True, then the function is wrapped in the database.
            round_ints: If True, then round the integer variables.
            is_observable: If True, new_iter_listeners are not called
                when function is called (avoid recursive call)

        Returns:
            The preprocessed function.
        """
        self.check_format(function)
        # First differentiate it so that the finite differences evaluations
        # are not stored in the database, which would be the case in the other
        # way round
        # Also, store non normalized values in the database for further
        # exploitation
        if isinstance(function, MDOLinearFunction) and not round_ints and normalize:
            function = self.__normalize_linear_function(function)
        else:
            function = NormFunction(function, normalize, round_ints, self)

        if self.differentiation_method in self.__DIFFERENTIATION_CLASSES.keys():
            self.__add_fd_jac(function, normalize)

        # Cast to real value, the results can be a complex number (ComplexStep)
        function.force_real = True
        if use_database:
            function = NormDBFunction(function, normalize, is_observable, self)
        return function

    def __normalize_linear_function(
        self,
        orig_func,  # type: MDOLinearFunction
    ):  # type: (...) -> MDOLinearFunction
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
        function,  # type: MDOFunction
        normalize,  # type: bool
    ):  # type: (...) -> None
        """Add a pointer to the approached Jacobian of the function.

        This Jacobian matrix is generated either by COMPLEX_STEP or FINITE_DIFFERENCES.

        Args:
            function: The function to be derivated.
            normalize: Whether to unnormalize the input vector of the function
                before evaluate it.
        """
        differentiation_class = self.__DIFFERENTIATION_CLASSES.get(
            self.differentiation_method
        )
        if differentiation_class is None:
            return

        differentiation_object = differentiation_class(
            function.evaluate,
            step=self.fd_step,
            parallel=self.__parallel_differentiation,
            design_space=self.design_space,
            normalize=normalize,
            **self.__parallel_differentiation_options
        )
        function.jac = differentiation_object.f_gradient

    def check(self):  # type: (...) -> None
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

    def __check_functions(self):  # type: (...) -> None
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
                the finite differences step is null.
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

    def change_objective_sign(self):  # type: (...) -> None
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
        cstr_type,  # type: str
        value,  # type: ndarray
    ):  # type: (...) -> bool
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
        out_val,  # type: Dict[str,ndarray]
        constraints=None,  # type: Optional[Iterable[MDOFunction]]
    ):  # type: (...) -> bool
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
    ):  # type: (...) -> Tuple[List[ndarray],List[Dict[str,Union[float,List[int]]]]]
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
        x_vect,  # type: ndarray
    ):  # type: (...) -> Tuple[bool,float]
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
    ):  # type: (...) -> BestInfeasiblePointType
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

        opt_f_dict = {}
        if len(f_history) <= best_i:
            f_opt = None
            x_opt = None
        else:
            f_opt = f_history[best_i].get(self.objective.name)
            x_opt = x_history[best_i]
            opt_f_dict = f_history[best_i]
        return x_opt, f_opt, is_opt_feasible, opt_f_dict

    def __get_optimum_infeas(
        self,
    ):  # type: (...) -> OptimumSolutionType
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
        feas_x,  # type: Sequence[ndarray]
        feas_f,  # type: Sequence[Dict[str, Union[float, List[int]]]]
    ):  # type: (...) -> OptimumSolutionType
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
        return x_opt, f_opt, c_opt, c_opt_grad

    def get_optimum(self):  # type: (...) -> OptimumType
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

    def __repr__(self):  # type: (...) -> str
        msg = MultiLineString()
        msg.add("Optimization problem:")
        msg.indent()
        # objective representation
        minimize_str = "Minimize: "
        n_char = len(minimize_str)
        objective_repr = repr(self.objective)
        obj_repr_lines = [line for line in objective_repr.split("\n") if line]
        msg.add(minimize_str + obj_repr_lines[0])
        for line in obj_repr_lines[1:]:
            msg.add(" " * n_char + line)
        # variables representation
        msg.add("With respect to: {}", pretty_repr(self.design_space.variables_names))
        if self.has_constraints():
            msg.add("Subject to constraints:")
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
        group,  # type: Any
        data_array,  # type: ndarray
        dataset_name,  # type: str
        dtype=None,  # type: Optional[str]
    ):  # type: (...) -> None
        """Store an array in a hdf5 file group.

        Args:
            group: The group pointer.
            data_array: The data to be stored.
            dataset_name: The name of the dataset to store the array.
            dtype: Numpy dtype or string. If None, dtype('f') will be used.
        """
        if data_array is None:
            return
        if isinstance(data_array, ndarray):
            data_array = data_array.real
        if isinstance(data_array, string_types):
            data_array = string_array([data_array])
        if isinstance(data_array, list):
            all_str = reduce(
                lambda x, y: x or y,
                (isinstance(data, string_types) for data in data_array),
            )
            if all_str:
                data_array = string_array(data_array)
                dtype = data_array.dtype
        group.create_dataset(dataset_name, data=data_array, dtype=dtype)

    @classmethod
    def __store_attr_h5data(
        cls,
        obj,  # type: Any
        group,  # type: str
    ):  # type: (...) -> None
        """Store an object that has a get_data_dict_repr attribute in the hdf5 dataset.

        Args:
            obj: The object to store
            group: The hdf5 group.
        """
        data_dict = obj.get_data_dict_repr()
        for attr_name, attr in data_dict.items():
            dtype = None
            is_arr_n = isinstance(attr, ndarray) and issubdtype(attr.dtype, np_number)
            if isinstance(attr, string_types):
                attr = attr.encode("ascii", "ignore")
            elif isinstance(attr, bytes):
                attr = attr.decode()
            elif hasattr(attr, "__iter__") and not is_arr_n:
                if PY3:
                    attr = [
                        att.encode("ascii", "ignore")
                        if isinstance(att, string_types)
                        else att
                        for att in attr
                    ]
                dtype = h5py.special_dtype(vlen=str)
            cls.__store_h5data(group, attr, attr_name, dtype)

    def export_hdf(
        self,
        file_path,  # type: str
        append=False,  # type: bool
    ):  # type: (...) -> None
        """Export the optimization problem to an HDF file.

        Args:
            file_path: The file to store the data.
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

                if hasattr(self.solution, "get_data_dict_repr"):
                    sol_group = h5file.require_group(self.SOLUTION_GROUP)
                    self.__store_attr_h5data(self.solution, sol_group)

        self.database.export_hdf(file_path, append=True)

        # Design space shall remain the same in append mode
        if not append or no_design_space:
            self.design_space.export_hdf(file_path, append=True)

    @classmethod
    def import_hdf(
        cls,
        file_path,  # type: str
        x_tolerance=0.0,  # type: float
    ):  # type: (...) -> OptimizationProblem
        """Import an optimization history from an HDF file.

        Args:
            file_path: The file containing the optimization history.
            x_tolerance: The tolerance on the design variables when reading the file.

        Returns:
            The read optimization problem.
        """
        LOGGER.info("Import optimization problem from file: %s", str(file_path))

        design_space = DesignSpace(file_path)
        opt_pb = OptimizationProblem(design_space, input_database=file_path)

        with h5py.File(file_path, "r") as h5file:
            group = get_hdf5_group(h5file, opt_pb.OPT_DESCR_GROUP)

            for attr_name, attr in group.items():
                val = attr[()]
                val = val.decode() if isinstance(val, bytes) else val
                setattr(opt_pb, attr_name, val)

            if opt_pb.SOLUTION_GROUP in h5file:
                data_dict = cls.__h5_group_to_dict(h5file, opt_pb.SOLUTION_GROUP)
                attr = OptimizationResult.init_from_dict_repr(**data_dict)
                opt_pb.solution = attr

            data_dict = cls.__h5_group_to_dict(h5file, opt_pb.OBJECTIVE_GROUP)
            attr = MDOFunction.init_from_dict_repr(**data_dict)

            # The generated functions can be called at the x stored in
            # the database
            attr.set_pt_from_database(
                opt_pb.database, design_space, jac=True, x_tolerance=x_tolerance
            )
            opt_pb.objective = attr

            if opt_pb.CONSTRAINTS_GROUP in h5file:
                group = get_hdf5_group(h5file, opt_pb.CONSTRAINTS_GROUP)

                for cstr_name in group.keys():
                    data_dict = cls.__h5_group_to_dict(group, cstr_name)
                    attr = MDOFunction.init_from_dict_repr(**data_dict)
                    opt_pb.constraints.append(attr)

        return opt_pb

    def export_to_dataset(
        self,
        name=None,  # type: Optional[str]
        by_group=True,  # type: bool
        categorize=True,  # type: bool
        opt_naming=True,  # type: bool
        export_gradients=False,  # type: bool
    ):  # type: (...) -> Dataset
        """Export the database of the optimization problem to a :class:`.Dataset`.

        The variables can be classified into groups,
        separating the design variables and functions
        (objective function and constraints).
        This classification can use either an optimization naming,
        with :attr:`.Database.DESIGN_GROUP` and :attr:`.Database.FUNCTION_GROUP`
        or an input-output naming,
        with :attr:`.Database.INPUT_GROUP` and :attr:`.Database.OUTPUT_GROUP`

        Args:
            name: A name to be given to the dataset.
                If None, use the name of the :attr:`database`.
            by_group: If True, then store the data by group.
                Otherwise, store them by variables.
            categorize: If True, then distinguish
                between the different groups of variables.
            opt_naming: If True, then use an optimization naming.
            export_gradients: If True, then export also the gradients of the functions
                (objective function, constraints and observables)
                if the latter are available in the database of the optimization problem.

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
        inputs_names = self.design_space.variables_names
        sizes = self.design_space.variables_sizes
        inputs_history = array(self.database.get_x_history())
        n_samples = inputs_history.shape[0]
        inputs_history = DataConversion.array_to_dict(
            inputs_history, inputs_names, sizes
        )
        for input_name, input_value in inputs_history.items():
            dataset.add_variable(input_name, input_value, input_group)

        def replace_missing_values(
            output_data,  # type: ndarray
            input_data,  # type: ndarray
        ):  # type: (...) -> ndarray
            """Replace the missing output values with NaN.

            Args:
                output_data: The output data with possibly missing values.
                input_data: The input data.

            Returns:
                The output data where missing values have been replaced with NaN.
            """
            if len(inputs_history) != n_samples:
                # There are fewer output values than input values
                n_values = len(input_data)
                output_dimension = output_data.size // n_values
                output_data = output_data.reshape((n_values, output_dimension))
                # Add NaN values at the missing inputs
                # N.B. the inputs are assumed to be in the same order.
                index = 0
                for sub_input_data in input_data:
                    while not array_equal(sub_input_data, inputs_history[index]):
                        output_data = insert(output_data, index, nan, 0)
                        index += 1
                    index += 1
                return insert(output_data, [index] * (n_samples - index), nan, 0)
            else:
                return output_data

        # Add database outputs
        all_data_names = self.database.get_all_data_names()
        outputs_names = list(
            set(all_data_names) - set(inputs_names) - {self.database.ITER_TAG}
        )
        functions_history = []
        for function_name in outputs_names:
            function_history, inputs_history = self.database.get_func_history(
                function_name, x_hist=True
            )
            function_history = replace_missing_values(function_history, inputs_history)
            reshaped_function_history = function_history.reshape((n_samples, -1))
            functions_history.append(reshaped_function_history)
            sizes.update({function_name: functions_history[-1].shape[1]})

        functions_history = concatenate(functions_history, axis=1)
        functions_history = DataConversion.array_to_dict(
            functions_history, outputs_names, sizes
        )
        for output_name, output_value in functions_history.items():
            dataset.add_variable(
                output_name,
                output_value,
                output_group,
                cache_as_input=cache_output_as_input,
            )

        # Add database output gradients
        if export_gradients:
            gradients_history = []
            gradients_names = []

            for function_name in outputs_names:
                if self.database.is_func_grad_history_empty(function_name):
                    continue

                gradient_history, inputs_history = self.database.get_func_grad_history(
                    function_name, x_hist=True
                )
                gradient_history = replace_missing_values(
                    gradient_history, inputs_history
                )
                gradients_history.append(gradient_history.reshape(n_samples, -1))
                gradient_name = Database.get_gradient_name(function_name)
                sizes.update({gradient_name: gradients_history[-1].shape[1]})
                gradients_names.append(Database.get_gradient_name(function_name))

            if gradients_history:
                gradients_history = concatenate(gradients_history, axis=1)
                gradients_history = DataConversion.array_to_dict(
                    gradients_history, gradients_names, sizes
                )
                for gradient_name, gradient_value in gradients_history.items():
                    dataset.add_variable(
                        gradient_name,
                        gradient_value,
                        gradient_group,
                        cache_as_input=cache_output_as_input,
                    )

        return dataset

    @staticmethod
    def __h5_group_to_dict(
        h5_handle,  # type: Union[h5py.File, h5py.Group]
        group_name,  # type: str
    ):  # type: (...) -> Dict[str, Union[str,List[str]]]
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

            converted[key] = value

        return converted

    def get_data_by_names(
        self,
        names,  # type: Union[str,Iterable[str]]
        as_dict=True,  # type: bool
        filter_non_feasible=False,  # type: bool
    ):  # type: (...) -> Union[ndarray, Dict[str,ndarray]]
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
            data = data[feasible_indexes, :]

        return data

    @property
    def is_mono_objective(self):  # type: (...) -> bool
        """Whether the optimization problem is mono-objective."""
        return len(self.objective.outvars) == 1

    def get_functions_dimensions(self):  # type: (...) -> Dict[str, int]
        """Return the dimensions of the outputs of the problem functions.

        Returns:
            The dimensions of the outputs of the problem functions.
            The dictionary keys are the functions names
            and the values are the functions dimensions.
        """
        design_variables = self.design_space.get_current_x()
        values, _ = self.evaluate_functions(design_variables, normalize=False)
        return {name: atleast_1d(value).size for name, value in values.items()}

    def get_number_of_unsatisfied_constraints(
        self,
        design_variables,  # type: ndarray
    ):  # type: (...) -> int
        """Return the number of scalar constraints not satisfied by design variables.

        Args:
            design_variables: The design variables.

        Returns:
            The number of unsatisfied scalar constraints.
        """
        n_unsatisfied = 0
        values, _ = self.evaluate_functions(design_variables, normalize=False)
        for constraint in self.constraints:
            value = atleast_1d(values[constraint.name])
            if constraint.f_type == MDOFunction.TYPE_EQ:
                value = numpy.absolute(value)
                tolerance = self.eq_tolerance
            else:
                tolerance = self.ineq_tolerance
            n_unsatisfied += sum(value > tolerance)
        return n_unsatisfied

    def get_scalar_constraints_names(self):  # type: (...) -> List[str]
        """Return the names of the scalar constraints.

        Returns:
            The names of the scalar constraints.
        """
        constraints_names = list()
        dimensions = self.get_functions_dimensions()
        for name in self.get_constraints_names():
            dimension = dimensions[name]
            if dimension == 1:
                constraints_names.append(name)
            else:
                constraints_names.extend(
                    [
                        "{}{}{}".format(name, DesignSpace.SEP, index)
                        for index in range(dimension)
                    ]
                )
        return constraints_names
