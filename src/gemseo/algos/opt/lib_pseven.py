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

import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import Union

import numpy
from da import p7core
from numpy import ndarray

from gemseo.algos.opt.core.pseven_problem_adapter import PSevenProblem
from gemseo.algos.opt.opt_lib import OptimizationLibrary
from gemseo.algos.opt_result import OptimizationResult
from gemseo.algos.stop_criteria import DesvarIsNan
from gemseo.algos.stop_criteria import FtolReached
from gemseo.algos.stop_criteria import FunctionIsNan
from gemseo.algos.stop_criteria import MaxIterReachedException
from gemseo.algos.stop_criteria import MaxTimeReached
from gemseo.algos.stop_criteria import XtolReached
from gemseo.core.mdofunctions.mdo_function import MDOLinearFunction
from gemseo.core.mdofunctions.mdo_function import MDOQuadraticFunction
from gemseo.utils.py23_compat import Path


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
        "max_batch_size": "GTOpt/BatchSize",
        "detect_nan_clusters": "GTOpt/DetectNaNClusters",
        "diff_scheme": "GTOpt/DiffScheme",
        "diff_type": "GTOpt/DiffType",
        "diff_step": "GTOpt/NumDiffStepSize",
        "ensure_feasibility": "GTOpt/EnsureFeasibility",
        "local_search": "GTOpt/LocalSearch",
        "restore_analytic_func": "GTOpt/RestoreAnalyticResponses",
        "responses_scalability": "GTOpt/ResponsesScalability",
    }

    LIBRARY_NAME = "pSeven"

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
                self.ALGORITHM_NAME: "PSEVEN",
                self.DESCRIPTION: "pSeven's Generic Tool for Optimization (GTOpt).",
                self.HANDLE_EQ_CONS: True,
                self.HANDLE_INEQ_CONS: True,
                self.HANDLE_INTEGER_VARIABLES: True,
                self.HANDLE_MULTIOBJECTIVE: False,
                self.INTERNAL_NAME: "pSeven",
                self.REQUIRE_GRAD: False,
                self.POSITIVE_CONSTRAINTS: False,
                self.WEBSITE: self.__WEBSITE,
            },
            "PSEVEN_FD": {
                self.ALGORITHM_NAME: "Feasible direction",
                self.DESCRIPTION: "pSeven's feasible direction method.",
                self.HANDLE_EQ_CONS: True,
                self.HANDLE_INEQ_CONS: True,
                self.HANDLE_INTEGER_VARIABLES: False,
                self.HANDLE_MULTIOBJECTIVE: False,
                self.INTERNAL_NAME: self.__FD,
                self.REQUIRE_GRAD: False,
                self.POSITIVE_CONSTRAINTS: False,
                self.WEBSITE: self.__WEBSITE,
            },
            "PSEVEN_MOM": {
                self.ALGORITHM_NAME: "MOM",
                self.DESCRIPTION: "pSeven's method of multipliers.",
                self.HANDLE_EQ_CONS: True,
                self.HANDLE_INEQ_CONS: True,
                self.HANDLE_INTEGER_VARIABLES: False,
                self.HANDLE_MULTIOBJECTIVE: False,
                self.INTERNAL_NAME: self.__MOM,
                self.REQUIRE_GRAD: False,
                self.POSITIVE_CONSTRAINTS: False,
                self.WEBSITE: self.__WEBSITE,
            },
            "PSEVEN_NCG": {
                self.ALGORITHM_NAME: "NCG",
                self.DESCRIPTION: "pSeven's nonlinear conjugate gradient method.",
                self.HANDLE_EQ_CONS: False,
                self.HANDLE_INEQ_CONS: False,
                self.HANDLE_INTEGER_VARIABLES: False,
                self.HANDLE_MULTIOBJECTIVE: False,
                self.INTERNAL_NAME: self.__NCG,
                self.REQUIRE_GRAD: False,
                self.POSITIVE_CONSTRAINTS: False,
                self.WEBSITE: self.__WEBSITE,
            },
            "PSEVEN_NLS": {
                self.ALGORITHM_NAME: "NLS",
                self.DESCRIPTION: "pSeven's nonlinear simplex method.",
                self.HANDLE_EQ_CONS: False,
                self.HANDLE_INEQ_CONS: False,
                self.HANDLE_INTEGER_VARIABLES: False,
                self.HANDLE_MULTIOBJECTIVE: False,
                self.POSITIVE_CONSTRAINTS: False,
                self.REQUIRE_GRAD: False,
                self.INTERNAL_NAME: self.__NLS,
                self.WEBSITE: self.__WEBSITE,
            },
            "PSEVEN_POWELL": {
                self.ALGORITHM_NAME: "POWELL",
                self.DESCRIPTION: "pSeven's Powell conjugate direction method.",
                self.HANDLE_EQ_CONS: False,
                self.HANDLE_INEQ_CONS: False,
                self.HANDLE_INTEGER_VARIABLES: False,
                self.HANDLE_MULTIOBJECTIVE: False,
                self.INTERNAL_NAME: self.__POWELL,
                self.POSITIVE_CONSTRAINTS: False,
                self.REQUIRE_GRAD: False,
                self.WEBSITE: self.__WEBSITE,
            },
            "PSEVEN_QP": {
                self.ALGORITHM_NAME: "QP",
                self.DESCRIPTION: "pSeven's quadratic programming method.",
                self.HANDLE_EQ_CONS: True,
                self.HANDLE_INEQ_CONS: True,
                self.HANDLE_INTEGER_VARIABLES: False,
                self.HANDLE_MULTIOBJECTIVE: False,
                self.INTERNAL_NAME: self.__QP,
                self.POSITIVE_CONSTRAINTS: False,
                self.REQUIRE_GRAD: False,
                self.WEBSITE: self.__WEBSITE,
            },
            "PSEVEN_SQP": {
                self.ALGORITHM_NAME: "SQP",
                self.DESCRIPTION: "pSeven's sequential quadratic programming method.",
                self.HANDLE_EQ_CONS: True,
                self.HANDLE_INEQ_CONS: True,
                self.HANDLE_INTEGER_VARIABLES: False,
                self.HANDLE_MULTIOBJECTIVE: False,
                self.INTERNAL_NAME: self.__SQP,
                self.POSITIVE_CONSTRAINTS: False,
                self.REQUIRE_GRAD: False,
                self.WEBSITE: self.__WEBSITE,
            },
            "PSEVEN_SQ2P": {
                self.ALGORITHM_NAME: "SQ2P",
                self.DESCRIPTION: (
                    "pSeven's sequential quadratic constrained quadratic "
                    "programming method."
                ),
                self.HANDLE_EQ_CONS: True,
                self.HANDLE_INEQ_CONS: True,
                self.HANDLE_INTEGER_VARIABLES: False,
                self.HANDLE_MULTIOBJECTIVE: False,
                self.INTERNAL_NAME: self.__SQ2P,
                self.POSITIVE_CONSTRAINTS: False,
                self.REQUIRE_GRAD: False,
                self.WEBSITE: self.__WEBSITE,
            },
        }

    def _get_options(
        self,
        max_iter=99,  # type: int
        evaluation_cost_type=None,  # type: Optional[Union[str, Mapping[str, str]]]
        expensive_evaluations=None,  # type: Optional[Mapping[str, int]]
        sample_x=None,  # type: Optional[Union[List[float], List[ndarray]]]
        sample_f=None,  # type: Optional[Union[List[float], List[ndarray]]]
        sample_c=None,  # type: Optional[Union[List[float], List[ndarray]]]
        constraints_smoothness="Auto",  # type: str
        global_phase_intensity="Auto",  # type: Union[str, float]
        max_expensive_func_iter=0,  # type: int
        max_func_iter=0,  # type: int
        objectives_smoothness="Auto",  # type: str
        deterministic="Auto",  # type: Union[str, bool]
        log_level="Error",  # type: str
        verbose_log=False,  # type: bool
        max_threads=0,  # type: int
        seed=100,  # type: int
        time_limit=0,  # type: int
        max_batch_size=0,  # type: int
        detect_nan_clusters=True,  # type: bool
        diff_scheme="Auto",  # type: str
        diff_type="Auto",  # type: str
        diff_step=1.1920929e-06,  # type: float
        ensure_feasibility=False,  # type: bool
        local_search="Disabled",  # type: str
        restore_analytic_func="Auto",  # type: Union[str, bool]
        responses_scalability=1,  # type: int
        globalization_method=None,  # type: Optional[str]
        surrogate_based=None,  # type: Optional[bool]
        use_gradient=True,  # type: bool
        ftol_abs=1e-14,  # type: float
        xtol_abs=1e-14,  # type: float
        ftol_rel=1e-8,  # type: float
        xtol_rel=1e-8,  # type: float
        stop_crit_n_x=3,  # type: int
        normalize_design_space=True,  # type: bool
        eq_tolerance=1e-2,  # type: float
        ineq_tolerance=1e-4,  # type: float
        log_path=None,  # type: Optional[str]
        **kwargs,  # type: Any
    ):  # type: (...) -> Dict
        """Set the default options values.

        Args:
            max_iter: The maximum number of evaluations.
            evaluation_cost_type: The evaluation cost type of each function of the
                problem: "Cheap" or "Expensive".
                If a string, then the same cost type is set for all the functions.
                If None, the evaluation cost types are set by pSeven.
            expensive_evaluations: The maximal number of expensive evaluations for
                each function of the problem. By default, set automatically by pSeven.
            sample_x: A sample of design points (in addition to the problem initial
                design).
            sample_f: The objectives values at the design points of the sample.
            sample_c: The constraints values at the design points of the sample.
            constraints_smoothness: The assumed smoothness of the constraints functions:
                "Smooth", "Noisy" or "Auto".
            global_phase_intensity: The configuration of global searching algorithms.
                This option has different meanings for expensive and non-expensive
                optimization problems. Refer to the pSeven Core API documentation.
                Defaults to "Auto".
            max_expensive_func_iter: The maximum number of evaluations for each
                expensive response, excluding the evaluations of initial guesses.
            max_func_iter: The maximum number of evaluations for any response,
                including the evaluations of initial guesses.
            objectives_smoothness: The assumed smoothness of the objective functions:
                "Smooth", "Noisy" or "Auto".
            deterministic: Whether to require optimization process to be reproducible
                using the passed seed value. Defaults to "Auto".
            log_level: The minimum log level:
                "Debug", "Info", "Warn", "Error" or "Fatal".
            verbose_log: Whether to enable verbose logging.
            max_threads: The maximum number of parallel threads to use when solving.
            seed: The random seed for deterministic mode.
            time_limit: The maximum allowed time to solve a problem in seconds.
                Defaults to 0, unlimited.
            max_batch_size: The maximum number of points in an evaluation batch.
                The (default) value 0 allows the optimizer to use any batch size.
            detect_nan_clusters: Whether to detect and avoid design space areas that
                yield NaN values (for at least one function).
                This option has no effect in the absence of "expensive" functions.
            diff_scheme: The order of the differentiation scheme (when the analytic
                derivatives are unavailable):
                "FirstOrder", "SecondOrder", "Adaptive" or "Auto".
            diff_type: The strategy for differentiation (when the analytic
                derivatives are unavailable):
                "Numerical", "Framed" or "Auto".
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
            responses_scalability: The maximum number of concurrent response evaluations
                supported by the problem.
            globalization_method: The globalization method:
                "RL" (random linkages),
                "PM" (plain multistart),
                or "MS" (surrogate model-based multistart)
                If None, set automatically by pSeven depending on the problem.
            surrogate_based: Whether to use surrogate models.
                If None, set automatically depending on the problem.
            use_gradient: Whether to use the functions derivatives.
            ftol_abs: The absolute tolerance on the objective function.
            xtol_abs: The absolute tolerance on the design parameters.
            ftol_rel: The relative tolerance on the objective function.
            xtol_rel: The relative tolerance on the design parameters.
            normalize_design_space: If True, normalize the design variables between 0
                and 1.
            stop_crit_n_x: The number of design vectors to take into account in the
                stopping criteria.
            eq_tolerance: The tolerance on the equality constraints.
            ineq_tolerance: The tolerance on the inequality constraints.
            log_path: The path where to save the pSeven log.
                If None, the pSeven log will not be saved.
            **kwargs: Other driver options.

        Returns:
            The processed options.
        """
        processed_options = self._process_options(
            max_iter=max_iter,
            evaluation_cost_type=evaluation_cost_type,
            expensive_evaluations=expensive_evaluations,
            sample_x=sample_x,
            sample_f=sample_f,
            sample_c=sample_c,
            constraints_smoothness=constraints_smoothness,
            global_phase_intensity=global_phase_intensity,
            max_expensive_func_iter=max_expensive_func_iter,
            max_func_iter=max_func_iter,
            objectives_smoothness=objectives_smoothness,
            deterministic=deterministic,
            log_level=log_level,
            verbose_log=verbose_log,
            max_threads=max_threads,
            seed=seed,
            time_limit=time_limit,
            max_batch_size=max_batch_size,
            detect_nan_clusters=detect_nan_clusters,
            diff_scheme=diff_scheme,
            diff_type=diff_type,
            diff_step=diff_step,
            ensure_feasibility=ensure_feasibility,
            local_search=local_search,
            restore_analytic_func=restore_analytic_func,
            responses_scalability=responses_scalability,
            globalization_method=globalization_method,
            surrogate_based=surrogate_based,
            use_gradient=use_gradient,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            stop_crit_n_x=stop_crit_n_x,
            normalize_design_space=normalize_design_space,
            eq_tolerance=eq_tolerance,
            ineq_tolerance=ineq_tolerance,
            log_path=log_path,
            **kwargs,
        )

        # Set the pSeven's techniques
        self.__set_pseven_techniques(processed_options)

        return processed_options

    def __set_pseven_techniques(
        self,
        options,  # type: MutableMapping[str, Any]
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
        """
        self.__check_functions()

        # Pop GEMSEO options
        normalize_ds = options.pop(self.NORMALIZE_DESIGN_SPACE_OPTION)
        del options["max_iter"]
        del options["ftol_abs"]
        del options["ftol_rel"]
        del options["xtol_abs"]
        del options["xtol_rel"]
        del options["stop_crit_n_x"]

        # Create the pSeven problem
        initial_x, lower_bnd, upper_bnd = self.get_x0_and_bounds_vects(normalize_ds)
        pseven_problem = PSevenProblem(
            self.problem,
            options.pop("evaluation_cost_type"),
            options.pop("expensive_evaluations"),
            lower_bnd,
            upper_bnd,
            initial_x,
            use_gradient=options.pop("use_gradient"),
        )

        # Set up the solver and its logger
        solver = p7core.gtopt.Solver()
        log_path = options.pop("log_path")
        if log_path is None:
            # Direct the logging to the standard output
            stream = sys.stdout
        else:
            # Direct the logging to a file
            stream = Path(log_path).open("w")

        solver.set_logger(
            p7core.loggers.StreamLogger(stream, options["GTOpt/LogLevel"])
        )

        # Disable pSeven stopping criteria
        options["GTOpt/CoordinateTolerance"] = 0.0
        options["GTOpt/ObjectiveTolerance"] = 0.0
        options["GTOpt/GradientTolerance"] = 0.0

        # Set the tolerance on the constraints to the minimum: the tolerances on the
        # constraints are effectively enforced in the definition of the constraints of
        # the PSevenProblem.
        options["GTOpt/ConstraintsTolerance"] = 0.01 * numpy.finfo(numpy.float32).eps

        # Grab the initial samples from the options
        samples = self.__get_samples(options, normalize_ds)

        # Run the algorithm and return the result
        try:
            result = solver.solve(pseven_problem, options=options, **samples)
        except p7core.exceptions.UserEvaluateException as exception:
            # Check whether a GEMSEO stopping criterion was raised during an
            # evaluation called by pSeven
            for criterion in [
                MaxIterReachedException,
                FunctionIsNan,
                DesvarIsNan,
                XtolReached,
                FtolReached,
                MaxTimeReached,
            ]:
                if str(exception).startswith(criterion.__name__):
                    raise criterion

            raise exception
        else:
            status = result.status

        # Close the log file
        if log_path is not None:
            stream.close()

        return self.get_optimum_from_database(str(status), status.id)

    def __get_samples(
        self,
        options,  # type: MutableMapping[str, Any]
        normalize_design_space,  # type: bool
    ):  # type: (...) -> Dict[str, ndarray]
        """Get the pSeven initial samples.

        Args:
            options: The processed options.
            normalize_design_space: Whether the design space is normalized.

        Returns:
            The pSeven initial samples.
        """
        samples = dict()
        sample_x = options.pop("sample_x")
        if sample_x is not None:
            if normalize_design_space:
                sample_x = [
                    self.problem.design_space.normalize_vect(point)
                    for point in sample_x
                ]

            samples["sample_x"] = sample_x

        for option in ["sample_f", "sample_c"]:
            if option in options:
                samples[option] = options.pop(option)

        return samples

    def __check_functions(self):  # type: (...) -> None
        """Check that the algorithm is consistent with the problem functions.

        Raises:
            RuntimeError: If a solver for constrained problems is called on
                a problem without constraint.
            TypeError: If the QP solver is called on a problem
                whose objective is neither quadratic nor linear
                or one of whose constraints are not linear.
        """
        if self.internal_algo_name in [
            self.__MOM,
            self.__QP,
            self.__SQP,
            self.__SQ2P,
        ]:
            if not self.problem.constraints:
                raise RuntimeError(
                    "{} requires at least one constraint".format(self.algo_name)
                )

        if self.internal_algo_name == self.__QP:
            if not isinstance(
                self.problem.nonproc_objective,
                (MDOQuadraticFunction, MDOLinearFunction),
            ):
                # TODO: support several objectives
                raise TypeError(
                    "{} requires the objective to be quadratic or linear".format(
                        self.algo_name
                    )
                )

            for constraint in self.problem.nonproc_constraints:
                if not isinstance(constraint, MDOLinearFunction):
                    raise TypeError(
                        "{} requires the constraints to be linear,"
                        " the following is not: {}".format(
                            self.algo_name, constraint.name
                        )
                    )
