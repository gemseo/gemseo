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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Benoit Pauwels
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

"""Wrapper for the Generic Tool for Optimization (GTOpt) of pSeven Core."""

from __future__ import unicode_literals

from enum import Enum
from math import sqrt
from typing import Any, Dict, List, Mapping, Optional, Union

from da import p7core
from numpy import ndarray

from gemseo.algos.opt.core.pseven_problem_adapter import CostType, PSevenProblem
from gemseo.algos.opt.opt_lib import OptimizationLibrary
from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.mdofunctions.mdo_function import (
    MDOLinearFunction,
    MDOQuadraticFunction,
)
from gemseo.utils.base_enum import CamelCaseEnum


class DiffScheme(CamelCaseEnum):
    """The differentiation schemes of pSeven."""

    FIRST_ORDER = 0
    SECOND_ORDER = 1
    # GTOpt switches between first and second order
    # depending on the estimated distance to optimality.
    ADAPTIVE = 2
    AUTO = 3  # GTOpt is left free to choose.


class DiffType(CamelCaseEnum):
    """The differentiation types of pSeven."""

    NUMERICAL = 0  # conventional numerical differentiation
    # framed simplex-based derivatives, recommended for noisy problems
    FRAMED = 1
    AUTO = 2  # GTOpt is left free to choose.


class GlobalMethod(Enum):
    """The globalization methods of pSeven.

    A item is represented with the name of its key.
    """

    RL = 0  # random linkages
    PM = 1  # plain multistart
    MS = 2  # surrogate model-based multistart

    def __str__(self):  # type: (...) -> str
        return self.name


class LogLevel(CamelCaseEnum):
    """The logging levels of pSeven."""

    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3
    FATAL = 4


class Smoothness(CamelCaseEnum):
    """The functions smoothness levels of pSeven."""

    SMOOTH = 0  # all functions are to be considered smooth
    NOISY = 1  # at least one function is noisy with noise level at most 10 %
    AUTO = 2  # GTOpt is free to assume whatever seems appropriate.


class PSevenOpt(OptimizationLibrary):
    """Interface for the Generic Tool for Optimization (GTOpt) of pSeven Core."""

    LIB_COMPUTE_GRAD = True
    OPTIONS_MAP = {
        "constraints_smoothness": "GTOpt/ConstraintsSmoothness",
        "global_phase_intensity": "GTOpt/GlobalPhaseIntensity",
        "max_expensive_func_iter": "GTOpt/MaximumExpensiveIterations",
        "max_func_iter": "GTOpt/MaximumIterations",
        "objectives_smoothness": "GTOpt/ObjectivesSmoothness",
        "deterministic": "GTOpt/Deterministic",
        "log_level": "GTOpt/LogLevel",
        "verbose_log": "GTOpt/VerboseOutput",
        "max_threads": "GTOpt/MaxParallel",
        "seed": "GTOpt/Seed",
        "time_limit": "GTOpt/TimeLimit",
        "grad_tol": "GTOpt/GradientTolerance",
        "grad_tol_is_abs": "GTOpt/AbsoluteGradientTolerance",
        "max_batch_size": "GTOpt/BatchSize",
        "detect_nan_clusters": "GTOpt/DetectNaNClusters",
        "diff_scheme": "GTOpt/DiffScheme",
        "diff_type": "GTOpt/DiffType",
        "diff_step": "GTOpt/NumDiffStepSize",
        "ensure_feasibility": "GTOpt/EnsureFeasibility",
        "local_search": "GTOpt/LocalSearch",
        "restore_analytic_func": "GTOpt/RestoreAnalyticResponses",
    }

    # Governing methods
    # Not yet used.
    # SO = "SO"
    # MO = "MO"
    __SBO = "SBO"

    # Local methods
    __FD = "FD"
    __MOM = "MoM"
    __NCG = "NCG"
    __NLS = "NLS"
    __POWELL = "Powell"
    __QP = "QP"
    __SQP = "SQP"
    __SQ2P = "S2P"
    __LOCAL_METHODS = (__FD, __MOM, __NCG, __NLS, __POWELL, __QP, __SQP, __SQ2P)

    __WEBSITE = "https://datadvance.net/product/pseven/manual/"

    def __init__(self):  # type: (...) -> None # noqa: D107
        super(PSevenOpt, self).__init__()
        self.lib_dict = {
            "PSEVEN": {
                self.INTERNAL_NAME: "pSeven",
                self.REQUIRE_GRAD: False,
                self.POSITIVE_CONSTRAINTS: False,
                self.HANDLE_EQ_CONS: True,
                self.HANDLE_INEQ_CONS: True,
                self.DESCRIPTION: "pSeven's Generic Tool for Optimization (GTOpt).",
                self.WEBSITE: self.__WEBSITE,
            },
            "PSEVEN_FD": {
                self.INTERNAL_NAME: self.__FD,
                self.REQUIRE_GRAD: False,
                self.POSITIVE_CONSTRAINTS: False,
                self.HANDLE_EQ_CONS: True,
                self.HANDLE_INEQ_CONS: True,
                self.DESCRIPTION: "pSeven's feasible direction method.",
                self.WEBSITE: self.__WEBSITE,
            },
            "PSEVEN_MOM": {
                self.INTERNAL_NAME: self.__MOM,
                self.REQUIRE_GRAD: False,
                self.POSITIVE_CONSTRAINTS: False,
                self.HANDLE_EQ_CONS: True,
                self.HANDLE_INEQ_CONS: True,
                self.DESCRIPTION: "pSeven's method of multipliers.",
                self.WEBSITE: self.__WEBSITE,
            },
            "PSEVEN_NCG": {
                self.INTERNAL_NAME: self.__NCG,
                self.REQUIRE_GRAD: False,
                self.POSITIVE_CONSTRAINTS: False,
                self.HANDLE_EQ_CONS: False,
                self.HANDLE_INEQ_CONS: False,
                self.DESCRIPTION: "pSeven's nonlinear conjugate gradient method.",
                self.WEBSITE: self.__WEBSITE,
            },
            "PSEVEN_NLS": {
                self.INTERNAL_NAME: self.__NLS,
                self.REQUIRE_GRAD: False,
                self.POSITIVE_CONSTRAINTS: False,
                self.HANDLE_EQ_CONS: False,
                self.HANDLE_INEQ_CONS: False,
                self.DESCRIPTION: "pSeven's nonlinear simplex method.",
                self.WEBSITE: self.__WEBSITE,
            },
            "PSEVEN_POWELL": {
                self.INTERNAL_NAME: self.__POWELL,
                self.REQUIRE_GRAD: False,
                self.POSITIVE_CONSTRAINTS: False,
                self.HANDLE_EQ_CONS: False,
                self.HANDLE_INEQ_CONS: False,
                self.DESCRIPTION: "pSeven's Powell conjugate direction method.",
                self.WEBSITE: self.__WEBSITE,
            },
            "PSEVEN_QP": {
                self.INTERNAL_NAME: self.__QP,
                self.REQUIRE_GRAD: False,
                self.POSITIVE_CONSTRAINTS: False,
                self.HANDLE_EQ_CONS: True,
                self.HANDLE_INEQ_CONS: True,
                self.DESCRIPTION: "pSeven's quadratic programming method.",
                self.WEBSITE: self.__WEBSITE,
            },
            "PSEVEN_SQP": {
                self.INTERNAL_NAME: self.__SQP,
                self.REQUIRE_GRAD: False,
                self.POSITIVE_CONSTRAINTS: False,
                self.HANDLE_EQ_CONS: True,
                self.HANDLE_INEQ_CONS: True,
                self.DESCRIPTION: "pSeven's sequential quadratic programming method.",
                self.WEBSITE: self.__WEBSITE,
            },
            "PSEVEN_SQ2P": {
                self.INTERNAL_NAME: self.__SQ2P,
                self.REQUIRE_GRAD: False,
                self.POSITIVE_CONSTRAINTS: False,
                self.HANDLE_EQ_CONS: True,
                self.HANDLE_INEQ_CONS: True,
                self.DESCRIPTION: "pSeven's sequential quadratic constrained quadratic "
                "programming method.",
                self.WEBSITE: self.__WEBSITE,
            },
        }

    def _get_options(
        self,
        max_iter=99,  # type: int
        evaluation_cost_type=None,  # type: Optional[Mapping[str, CostType]]
        expensive_evaluations=None,  # type: Optional[Mapping[str, int]]
        sample_x=None,  # type: Optional[Union[List[float],List[ndarray]]]
        sample_f=None,  # type: Optional[Union[List[float],List[ndarray]]]
        sample_c=None,  # type: Optional[Union[List[float],List[ndarray]]]
        constraints_smoothness=Smoothness.AUTO,  # type: Smoothness
        global_phase_intensity=None,  # type: Optional[float]
        max_expensive_func_iter=0,  # type: int
        max_func_iter=0,  # type: int
        objectives_smoothness=Smoothness.AUTO,  # type: Smoothness
        deterministic=None,  # type: Optional[bool]
        log_level=LogLevel.INFO,  # type: LogLevel
        verbose_log=False,  # type: bool
        max_threads=0,  # type: int
        seed=100,  # type: int
        time_limit=0,  # type: int
        grad_tol=1e-5,  # type: float
        grad_tol_is_abs=False,  # type: bool
        max_batch_size=0,  # type: int
        detect_nan_clusters=True,  # type: bool
        diff_scheme=DiffScheme.AUTO,  # type: DiffScheme
        diff_type=DiffType.AUTO,  # type: DiffType
        diff_step=1.1920929e-06,  # type: float
        ensure_feasibility=False,  # type: bool
        local_search=False,  # type: bool
        restore_analytic_func=True,  # type: bool
        globalization_method=None,  # type: Optional[GlobalMethod]
        surrogate_based=None,  # type: Optional[bool]
        **kwargs  # type: Any
    ):  # type: (...) -> Dict
        """Set the default options values.

        Args:
            max_iter: The maximum number of evaluations.
            evaluation_cost_type: The evaluation cost type of each function of the
                problem.
                If None, the evaluation cost types default to "Cheap".
            expensive_evaluations: The maximal number of expensive evaluations for
                each function of the problem. By default, set automatically by pSeven.
            sample_x: A sample of design points (in addition to the problem initial
                design).
            sample_f: The objectives values at the design points of the sample.
            sample_c: The constraints values at the design points of the sample.
            constraints_smoothness: The assumed smoothness of the constraints functions.
            global_phase_intensity: The configuration of global searching algorithms.
                This option has different meanings for expensive and non-expensive
                optimization problems. Refer to the pSeven Core API documentation.
                Defaults to "Auto".
            max_expensive_func_iter: The maximum number of evaluations for each
                expensive response, excluding the evaluations of initial guesses.
            max_func_iter: The maximum number of evaluations for any response,
                including the evaluations of initial guesses.
            objectives_smoothness: The assumed smoothness of the objective functions.
            deterministic: Whether to require optimization process to be reproducible
                using the passed seed value. Defaults to "Auto".
            log_level: The minimum log level.
            verbose_log: Whether to enable verbose logging.
            max_threads: The maximum number of parallel threads to use when solving.
            seed: The random seed for deterministic mode.
            time_limit: The maximum allowed time to solve a problem in seconds.
                Defaults to 0, unlimited.
            grad_tol: The tolerance on the infinity-norm of the gradient (or optimal
                descent for constrained and multi-objective problems) at which
                optimization stops.
                If 'gradient_tol_is_abs' is False then the tolerance is relative to
                the infinity-norm of the current objectives values; otherwise the
                tolerance is absolute.
                The value 0.0 deactivate the gradient-based stopping criterion.
            grad_tol_is_abs: Whether 'grad_tol' should be regarded as an absolute
                tolerance. See 'grad_tol' for details.
            max_batch_size: The maximum number of points in an evaluation batch.
                The (default) value 0 allows the optimizer to use any batch size.
            detect_nan_clusters: Whether to detect and avoid design space areas that
                yield NaN values (for at least one function).
                This option has no effect in the absence of "expensive" functions.
            diff_scheme: The order of the differentiation scheme (when the analytic
                derivatives are unavailable).
            diff_type: The strategy for differentiation (when the analytic
                derivatives are unavailable).
            diff_step: The numerical differentiation step size.
            ensure_feasibility: Whether to restrict the evaluations of the objectives
                to feasible designs only.
            local_search: Whether to force the surrogate models to explore the design
                space locally near the current optimum,
                or to disable the local search and let the surrogate models explore the
                whole design space.
            restore_analytic_func: Whether to restore the analytic forms of the
                linear and quadratic functions.
                Once the analytic forms are restored the original functions will not
                be evaluated anymore.
            globalization_method: The globalization method.
                If None, set automatically depending on the problem.
            surrogate_based: Whether to use surrogate models.
                If None, set automatically depending on the problem.
            **kwargs: Other driver options.

        Returns:
            The processed options.

        Raises:
            ValueError: If the value for one the following option is invalid:
                objectives smoothness,
                constraints smoothness,
                logging level,
                differentiation scheme,
                differentiation type,
                globalization method.
        """
        # Check the options
        self.__check_pseven_options(
            evaluation_cost_type,
            expensive_evaluations,
            constraints_smoothness,
            objectives_smoothness,
            log_level,
            diff_scheme,
            diff_type,
            globalization_method,
        )

        # Process the options
        if globalization_method is not None:
            globalization_method = str(globalization_method)

        processed_options = self._process_options(
            max_iter=max_iter,
            evaluation_cost_type=evaluation_cost_type,
            expensive_evaluations=expensive_evaluations,
            sample_x=sample_x,
            sample_f=sample_f,
            sample_c=sample_c,
            constraints_smoothness=str(constraints_smoothness),
            global_phase_intensity=global_phase_intensity,
            max_expensive_func_iter=max_expensive_func_iter,
            max_func_iter=max_func_iter,
            objectives_smoothness=str(objectives_smoothness),
            deterministic=deterministic,
            log_level=str(log_level),
            verbose_log=verbose_log,
            max_threads=max_threads,
            seed=seed,
            time_limit=time_limit,
            grad_tol=grad_tol,
            grad_tol_is_abs=grad_tol_is_abs,
            max_batch_size=max_batch_size,
            detect_nan_clusters=detect_nan_clusters,
            diff_scheme=str(diff_scheme),
            diff_type=str(diff_type),
            diff_step=diff_step,
            ensure_feasibility=ensure_feasibility,
            local_search=local_search,
            restore_analytic_func=restore_analytic_func,
            globalization_method=globalization_method,
            surrogate_based=surrogate_based,
            **kwargs
        )

        gtopt_local_search = "GTOpt/LocalSearch"

        if processed_options[gtopt_local_search]:
            processed_options[gtopt_local_search] = "Forced"
        else:
            processed_options[gtopt_local_search] = "Disabled"

        # Disable pSeven's internal stopping criterion based on successive designs
        processed_options["GTOpt/CoordinateTolerance"] = 0.0

        # Disable pSeven's internal stopping criterion based on successive objectives
        processed_options["GTOpt/ObjectiveTolerance"] = 0.0

        # Set the tolerance on the constraints (N.B. relative to the infinity-norm)
        processed_options[
            "GTOpt/ConstraintsTolerance"
        ] = self.__compute_constraints_tolerance()

        # Set the pSeven's techniques
        self.__set_pseven_techniques(processed_options)

        return processed_options

    def __check_pseven_options(
        self,
        evaluation_cost_type,  # type: Mapping[str, CostType]
        expensive_evaluations,  # type: Mapping[str, int]
        constraints_smoothness,  # type: Smoothness
        objectives_smoothness,  # type: Smoothness
        log_level,  # type: LogLevel
        diff_scheme,  # type: DiffScheme
        diff_type,  # type: DiffType
        globalization_method,  # type: GlobalMethod
    ):  # type: (...) -> None
        """Check pSeven's options.

        Args:
            evaluation_cost_type: The evaluation cost type of each function of the
                problem.
                If None, the evaluation cost types default to "Cheap".
            expensive_evaluations: The maximal number of expensive evaluations for
                each function of the problem.
                 If None, set automatically by pSeven.
            constraints_smoothness: The assumed smoothness of the constraints functions.
            objectives_smoothness: The assumed smoothness of the objective functions.
            log_level: The minimum log level.
            diff_scheme: The order of the differentiation scheme (when the analytic
                derivatives are unavailable).
            diff_type: The strategy for differentiation (when the analytic derivatives
                are unavailable).
            globalization_method: The globalization method.

        Raises:
            ValueError: If the value for one the following option is invalid:
                objectives smoothness,
                constraints smoothness,
                logging level,
                differentiation scheme,
                differentiation type,
                globalization method.
        """
        if evaluation_cost_type is not None:
            self.__check_evaluation_cost_type(evaluation_cost_type)

        if expensive_evaluations is not None:
            self.__check_expensive_evaluations(expensive_evaluations)

        if not isinstance(objectives_smoothness, Smoothness):
            raise ValueError(
                "Unknown objectives smoothness: {}".format(objectives_smoothness)
            )

        if not isinstance(constraints_smoothness, Smoothness):
            raise ValueError(
                "Unknown constraints smoothness: {}".format(constraints_smoothness)
            )

        if not isinstance(log_level, LogLevel):
            raise ValueError("Unknown log level: {}".format(log_level))

        if not isinstance(diff_scheme, DiffScheme):
            raise ValueError("Unknown differentiation scheme: {}".format(diff_scheme))

        if not isinstance(diff_type, DiffType):
            raise ValueError("Unknown differentiation type: {}".format(diff_type))

        if globalization_method is not None and not isinstance(
            globalization_method, GlobalMethod
        ):
            raise ValueError(
                "Unknown globalization method: {}".format(globalization_method)
            )

    def __check_evaluation_cost_type(
        self,
        evaluation_cost_type,  # type: Mapping[str, CostType]
    ):  # type: (...) -> None
        """Check the evaluation cost types.

        Args:
            evaluation_cost_type: The evaluation cost type of each function of the
                problem.

        Raises:
            ValueError: If a function name does not refer to a function of the problem,
                or if a cost type is invalid.
        """
        functions_names = [
            self.problem.get_objective_name()
        ] + self.problem.get_constraints_names()

        if evaluation_cost_type is not None:
            for func_name, func_type in evaluation_cost_type.items():
                if func_name not in functions_names:
                    raise ValueError("Unknown function name: {}".format(func_name))
                if not isinstance(func_type, CostType):
                    raise ValueError(
                        "Unknown cost type for function '{}': {}".format(
                            func_name, func_type
                        )
                    )

    def __check_expensive_evaluations(
        self,
        expensive_evaluations,  # type: Mapping[str, int]
    ):  # type: (...) -> None
        """Check the numbers of expensive evaluations.

        Args:
            expensive_evaluations: The maximal number of expensive evaluations for
                each function of the problem.

        Raises:
            ValueError: If a function name does not refer to a function of the problem.
            TypeError: If a number of expensive evaluations is not an integer.
        """
        functions_names = [
            self.problem.get_objective_name()
        ] + self.problem.get_constraints_names()

        for func_name, eval_number in expensive_evaluations.items():
            if func_name not in functions_names:
                raise ValueError("Unknown function name: {}".format(func_name))
            if not isinstance(eval_number, int):
                raise TypeError(
                    "Non-integer evaluations number for function '{}': {}".format(
                        func_name, eval_number
                    )
                )

    def __compute_constraints_tolerance(self):  # type: (...) -> Optional[float]
        """Compute the pSeven tolerance on the constraints.

        This tolerance is relative to the infinity norm.
        """
        tolerance = None

        if self.problem.has_eq_constraints() and self.problem.has_ineq_constraints():
            tolerance = min(self.problem.eq_tolerance, self.problem.ineq_tolerance)
        elif self.problem.has_eq_constraints():
            tolerance = self.problem.eq_tolerance
        elif self.problem.has_ineq_constraints():
            tolerance = self.problem.ineq_tolerance

        if tolerance:
            init_x = self.problem.design_space.get_current_x()
            constr_dim = sum(
                [constraint(init_x).size for constraint in self.problem.constraints]
            )
            tolerance /= sqrt(constr_dim)
            # N.B. ||c(x)||_2 <= sqrt(constr_dim) * ||c(x)||_inf

        return tolerance

    def __set_pseven_techniques(
        self,
        options,  # type: Dict[str, Any]
    ):  # type: (...) -> None
        """Get the pSeven techniques from the options."""
        techniques_list = list()
        internal_algo_name = self.lib_dict[self.algo_name][self.INTERNAL_NAME]

        if internal_algo_name in self.__LOCAL_METHODS:
            techniques_list.append(internal_algo_name)

        globalization_method = options.pop("globalization_method", None)
        if globalization_method is not None:
            techniques_list.append(globalization_method)

        surrogate_based = options.pop("surrogate_based", None)
        if surrogate_based is not None and surrogate_based:
            techniques_list.append(self.__SBO)

        if techniques_list:
            options["GTOpt/Techniques"] = "[" + ", ".join(techniques_list) + "]"

    def _run(
        self, **options  # type: Any
    ):  # type: (...) -> OptimizationResult
        """Run the algorithm.

        Args:
            **options: The options of the algorithm.

        Returns:
            The result of the optimization.

        Raises:
            RuntimeError: If a solver for constrained problems is called on
                a problem without constraint.
            TypeError: If the QP solver is called on a problem
                whose objective is neither quadratic nor linear
                or one of whose constraints are not linear.
        """
        problem = self.problem
        options.pop("max_iter")

        # Check the functions
        if self.internal_algo_name in [
            self.__MOM,
            self.__QP,
            self.__SQP,
            self.__SQ2P,
        ]:
            if not problem.constraints:
                raise RuntimeError(
                    "{} requires at least one constraint".format(self.algo_name)
                )

        if self.internal_algo_name == self.__QP:
            if not isinstance(
                problem.nonproc_objective, (MDOQuadraticFunction, MDOLinearFunction)
            ):
                # TODO: support several objectives
                raise TypeError(
                    "{} requires the objective to be quadratic or linear".format(
                        self.algo_name
                    )
                )

            for constraint in problem.nonproc_constraints:
                if not isinstance(constraint, MDOLinearFunction):
                    raise TypeError(
                        "{} requires the constraints to be linear,"
                        " the following is not: {}".format(
                            self.algo_name, constraint.name
                        )
                    )

        # Create the pSeven problem
        normalize_ds = options.pop(self.NORMALIZE_DESIGN_SPACE_OPTION, True)
        initial_x, lower_bnd, upper_bnd = self.get_x0_and_bounds_vects(normalize_ds)
        evaluation_cost_type = options.pop("evaluation_cost_type", None)
        expensive_evaluations = options.pop("expensive_evaluations", None)

        # Grab the initial sample from the options
        sample = dict()
        sample_x = options.pop("sample_x", None)
        if sample_x is not None:
            if normalize_ds:
                sample["sample_x"] = [
                    problem.design_space.normalize_vect(point) for point in sample_x
                ]
            else:
                sample["sample_x"] = sample_x

        for option in ["sample_f", "sample_c"]:
            sample_option = options.pop(option, None)
            if sample_option is not None:
                sample[option] = sample_option

        pseven_problem = PSevenProblem(
            problem,
            evaluation_cost_type,
            expensive_evaluations,
            lower_bnd,
            upper_bnd,
            initial_x,
        )

        # Run the algorithm and return the result
        try:
            result = p7core.gtopt.Solver().solve(
                pseven_problem, options=options, **sample
            )
        except p7core.exceptions.UserEvaluateException:
            # Gemseo terminated pSeven during evaluation as a stopping criterion was met
            status = p7core.status.USER_TERMINATED
        else:
            status = result.status

        return self.get_optimum_from_database(str(status), status.id)
