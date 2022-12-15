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
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import MutableMapping

import numpy
from da import p7core
from numpy import ndarray

from gemseo.algos.opt.core.pseven_problem_adapter import PSevenProblem
from gemseo.algos.opt.opt_lib import OptimizationAlgorithmDescription
from gemseo.algos.opt.opt_lib import OptimizationLibrary
from gemseo.algos.opt_result import OptimizationResult
from gemseo.algos.stop_criteria import DesvarIsNan
from gemseo.algos.stop_criteria import FtolReached
from gemseo.algos.stop_criteria import FunctionIsNan
from gemseo.algos.stop_criteria import MaxIterReachedException
from gemseo.algos.stop_criteria import MaxTimeReached
from gemseo.algos.stop_criteria import XtolReached
from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction
from gemseo.core.mdofunctions.mdo_quadratic_function import MDOQuadraticFunction


@dataclass
class PSevenAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of an optimization algorithm from the NLopt library."""

    library_name: str = "pSeven"
    website: str = "https://datadvance.net/product/pseven/manual/"


LOGGER = logging.getLogger(__name__)


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

    # Initial sample
    __SAMPLE_X = "sample_x"
    __SAMPLE_F = "sample_f"
    __SAMPLE_C = "sample_c"

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

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.descriptions = {
            "PSEVEN": PSevenAlgorithmDescription(
                algorithm_name="PSEVEN",
                description="pSeven's Generic Tool for Optimization (GTOpt).",
                handle_equality_constraints=True,
                handle_inequality_constraints=True,
                handle_integer_variables=True,
                internal_algorithm_name="pSeven",
            ),
            "PSEVEN_FD": PSevenAlgorithmDescription(
                algorithm_name="Feasible direction",
                description="pSeven's feasible direction method.",
                handle_equality_constraints=True,
                handle_inequality_constraints=True,
                internal_algorithm_name=self.__FD,
            ),
            "PSEVEN_MOM": PSevenAlgorithmDescription(
                algorithm_name="MOM",
                description="pSeven's method of multipliers.",
                handle_equality_constraints=True,
                handle_inequality_constraints=True,
                internal_algorithm_name=self.__MOM,
            ),
            "PSEVEN_NCG": PSevenAlgorithmDescription(
                algorithm_name="NCG",
                description="pSeven's nonlinear conjugate gradient method.",
                internal_algorithm_name=self.__NCG,
            ),
            "PSEVEN_NLS": PSevenAlgorithmDescription(
                algorithm_name="NLS",
                description="pSeven's nonlinear simplex method.",
                internal_algorithm_name=self.__NLS,
            ),
            "PSEVEN_POWELL": PSevenAlgorithmDescription(
                algorithm_name="POWELL",
                description="pSeven's Powell conjugate direction method.",
                internal_algorithm_name=self.__POWELL,
            ),
            "PSEVEN_QP": PSevenAlgorithmDescription(
                algorithm_name="QP",
                description="pSeven's quadratic programming method.",
                handle_equality_constraints=True,
                handle_inequality_constraints=True,
                internal_algorithm_name=self.__QP,
            ),
            "PSEVEN_SQP": PSevenAlgorithmDescription(
                algorithm_name="SQP",
                description="pSeven's sequential quadratic programming method.",
                handle_equality_constraints=True,
                handle_inequality_constraints=True,
                internal_algorithm_name=self.__SQP,
            ),
            "PSEVEN_SQ2P": PSevenAlgorithmDescription(
                algorithm_name="SQ2P",
                description=(
                    "pSeven's sequential quadratic constrained quadratic "
                    "programming method."
                ),
                handle_equality_constraints=True,
                handle_inequality_constraints=True,
                internal_algorithm_name=self.__SQ2P,
            ),
        }

    def _get_options(
        self,
        max_iter: int = 99,
        evaluation_cost_type: str | Mapping[str, str] | None = None,
        expensive_evaluations: Mapping[str, int] | None = None,
        sample_x: list[float] | list[ndarray] | None = None,
        sample_f: list[float] | list[ndarray] | None = None,
        sample_c: list[float] | list[ndarray] | None = None,
        constraints_smoothness: str = "Auto",
        global_phase_intensity: str | float = "Auto",
        max_expensive_func_iter: int = 0,
        objectives_smoothness: str = "Auto",
        deterministic: str | bool = "Auto",
        log_level: str = "Error",
        verbose_log: bool = False,
        max_threads: int = 0,
        seed: int = 100,
        time_limit: int = 0,
        max_batch_size: int = 0,
        detect_nan_clusters: bool = True,
        diff_scheme: str = "Auto",
        diff_type: str = "Auto",
        diff_step: float = 1.1920929e-06,
        ensure_feasibility: bool = False,
        local_search: str = "Disabled",
        restore_analytic_func: str | bool = "Auto",
        responses_scalability: int = 1,
        globalization_method: str | None = None,
        surrogate_based: bool | None = None,
        use_gradient: bool = True,
        ftol_abs: float = 1e-14,
        xtol_abs: float = 1e-14,
        ftol_rel: float = 1e-8,
        xtol_rel: float = 1e-8,
        stop_crit_n_x: int = 3,
        normalize_design_space: bool = True,
        eq_tolerance: float = 1e-2,
        ineq_tolerance: float = 1e-4,
        log_path: str | None = None,
        use_threading: bool = False,
        **kwargs: Any,
    ) -> dict:
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
                Batches of size more than one are evaluated in parallel.
                The (default) value 0 allows the optimizer to use any batch size.
                The value 1 implements sequential evaluation.
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
            use_threading: Whether to use threads instead of processes to parallelize
                the evaluation of the functions.
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
            max_func_iter=max_iter,
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
            use_threading=use_threading,
            **kwargs,
        )

        # Set the pSeven's techniques
        self.__set_pseven_techniques(processed_options)

        return processed_options

    def __set_pseven_techniques(
        self,
        options: MutableMapping[str, Any],
    ) -> None:
        """Get the pSeven techniques from the options."""
        technique_names = list()
        internal_algo_name = self.descriptions[self.algo_name].internal_algorithm_name

        if internal_algo_name in self.__LOCAL_METHODS:
            technique_names.append(internal_algo_name)

        globalization_method = options.pop("globalization_method", None)
        if globalization_method is not None:
            technique_names.append(globalization_method)

        surrogate_based = options.pop("surrogate_based", None)
        if surrogate_based is not None and surrogate_based:
            technique_names.append(self.__SBO)

        if technique_names:
            options["GTOpt/Techniques"] = "[" + ", ".join(technique_names) + "]"

    def _run(self, **options: Any) -> OptimizationResult:
        """Run the algorithm.

        Args:
            **options: The options of the algorithm.

        Returns:
            The result of the optimization.
        """
        self.__check_functions()

        # Pop GEMSEO options
        normalize_ds = options.pop(self.NORMALIZE_DESIGN_SPACE_OPTION)
        max_iter = options.pop(self.MAX_ITER)
        del options[self.F_TOL_ABS]
        del options[self.F_TOL_REL]
        del options[self.X_TOL_ABS]
        del options[self.X_TOL_REL]
        del options[self.STOP_CRIT_NX]

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
            use_threading=options.pop("use_threading"),
            normalize_design_space=normalize_ds,
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

        # Check whether the evaluations budget is sufficient for the expensive functions
        self.__check_expensive_evaluations_budget(options, samples, max_iter)

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
        options: MutableMapping[str, Any],
        normalize_design_space: bool,
    ) -> dict[str, ndarray]:
        """Get the pSeven initial samples.

        Args:
            options: The processed options.
            normalize_design_space: Whether the design space is normalized.

        Returns:
            The pSeven initial samples.
        """
        samples = dict()
        sample_x = options.pop(self.__SAMPLE_X)
        if sample_x is not None:
            if normalize_design_space:
                sample_x = [
                    self.problem.design_space.normalize_vect(point)
                    for point in sample_x
                ]

            samples[self.__SAMPLE_X] = sample_x

            for option in [self.__SAMPLE_F, self.__SAMPLE_C]:
                sample_functions = options.pop(option)
                if sample_functions is not None:
                    samples[option] = sample_functions
        else:
            del options[self.__SAMPLE_F]
            del options[self.__SAMPLE_C]

        return samples

    def __check_functions(self) -> None:
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
                    f"{self.algo_name} requires at least one constraint."
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

    def __check_expensive_evaluations_budget(
        self,
        options: dict[str, Any],
        samples: dict[str, ndarray],
        max_iter: int,
    ) -> None:
        """Check whether the expensive evaluations budget is sufficient.

        Args:
            options: The pSeven options.
            samples: The additional initial guesses.
            max_iter: The maximum number of evaluated designs.
        """
        max_expensive_iter = options.get("GTOpt/MaximumExpensiveIterations")
        if max_expensive_iter is None:
            return

        number_of_initial_guesses = 0
        if self.problem.design_space.has_current_value():
            number_of_initial_guesses += 1

        if self.__SAMPLE_X in samples:
            number_of_initial_guesses += len(samples[self.__SAMPLE_X])

        if max_iter < max_expensive_iter + number_of_initial_guesses:
            LOGGER.warning(
                "The evaluations budget (%s=%d) is to small to compute the "
                "expensive functions at both the initial guesses (%d) and the "
                "iterates (%d).",
                self.MAX_ITER,
                max_iter,
                number_of_initial_guesses,
                max_expensive_iter,
            )
