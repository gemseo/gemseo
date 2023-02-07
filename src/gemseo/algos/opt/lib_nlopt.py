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
#        :author: Damien Guenot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#         Francois Gallard : refactoring for v1, May 2016
"""NLopt library wrapper."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Union

import nlopt
from nlopt import RoundoffLimited
from numpy import atleast_1d
from numpy import atleast_2d
from numpy import ndarray

from gemseo.algos.opt.opt_lib import OptimizationAlgorithmDescription
from gemseo.algos.opt.opt_lib import OptimizationLibrary
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result import OptimizationResult
from gemseo.algos.stop_criteria import TerminationCriterion
from gemseo.core.mdofunctions.mdo_function import MDOFunction

LOGGER = logging.getLogger(__name__)

NLoptOptionsType = Union[bool, int, float]


class NloptRoundOffException(Exception):
    """NLopt roundoff error."""


@dataclass
class NLoptAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of an optimization algorithm from the NLopt library."""

    library_name: str = "NLopt"


class Nlopt(OptimizationLibrary):
    """NLopt optimization library interface.

    See OptimizationLibrary.
    """

    LIB_COMPUTE_GRAD = False
    INNER_MAXEVAL = "inner_maxeval"
    STOPVAL = "stopval"
    CTOL_ABS = "ctol_abs"
    INIT_STEP = "init_step"
    SUCCESS = "NLOPT_SUCCESS: Generic success return value"
    STOPVAL_REACHED = (
        "NLOPT_STOPVAL_REACHED: Optimization stopped  "
        "because stopval (above) was reached"
    )
    FTOL_REACHED = (
        "NLOPT_FTOL_REACHED: Optimization stopped "
        "because ftol_rel or ftol_abs (above) was reached"
    )

    XTOL_REACHED = (
        "NLOPT_XTOL_REACHED Optimization stopped "
        "because xtol_rel or xtol_abs (above) was reached"
    )

    MAXEVAL_REACHED = (
        "NLOPT_MAXEVAL_REACHED: Optimization stopped "
        "because maxeval (above) was reached"
    )

    MAXTIME_REACHED = (
        "NLOPT_MAXTIME_REACHED: Optimization stopped "
        "because maxtime (above) was reached"
    )
    FAILURE = "NLOPT_FAILURE:    Generic failure code"

    INVALID_ARGS = (
        "NLOPT_INVALID_ARGS: Invalid arguments (e.g. lower "
        "bounds are bigger than upper bounds, an unknown"
        " algorithm was specified, etcetera)."
    )
    OUT_OF_MEMORY = "OUT_OF_MEMORY: Ran out of memory"
    ROUNDOFF_LIMITED = (
        "NLOPT_ROUNDOFF_LIMITED: Halted because "
        "roundoff errors limited progress. (In this "
        "case, the optimization still typically "
        "returns a useful result.)"
    )
    FORCED_STOP = (
        "NLOPT_FORCED_STOP: Halted because of a forced "
        "termination: the user called nlopt_force_stop"
        "(opt) on the optimization’s nlopt_opt"
        " object opt from the user’s objective "
        "function or constraints."
    )

    NLOPT_MESSAGES = {
        1: SUCCESS,
        2: STOPVAL_REACHED,
        3: FTOL_REACHED,
        4: XTOL_REACHED,
        5: MAXEVAL_REACHED,
        6: MAXTIME_REACHED,
        -1: FAILURE,
        -2: INVALID_ARGS,
        -3: OUT_OF_MEMORY,
        -4: ROUNDOFF_LIMITED,
        -5: FORCED_STOP,
    }

    LIBRARY_NAME = "NLopt"

    def __init__(self) -> None:  # noqa:D107
        super().__init__()

        nlopt_doc = "https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/"
        self.descriptions = {
            "NLOPT_MMA": NLoptAlgorithmDescription(
                algorithm_name="MMA",
                description=(
                    "Method of Moving Asymptotes (MMA)"
                    "implemented in the NLOPT library"
                ),
                handle_inequality_constraints=True,
                internal_algorithm_name=nlopt.LD_MMA,
                require_gradient=True,
                website=f"{nlopt_doc}#mma-method-of-moving-asymptotes-and-ccsa",
            ),
            "NLOPT_COBYLA": NLoptAlgorithmDescription(
                algorithm_name="COBYLA",
                description=(
                    "Constrained Optimization BY Linear "
                    "Approximations (COBYLA) implemented "
                    "in the NLOPT library"
                ),
                handle_equality_constraints=True,
                handle_inequality_constraints=True,
                internal_algorithm_name=nlopt.LN_COBYLA,
                website=(
                    f"{nlopt_doc}#cobyla-constrained-optimization-by-linear-"
                    "approximations"
                ),
            ),
            "NLOPT_SLSQP": NLoptAlgorithmDescription(
                algorithm_name="SLSQP",
                description=(
                    "Sequential Least-Squares Quadratic "
                    "Programming (SLSQP) implemented in "
                    "the NLOPT library"
                ),
                handle_equality_constraints=True,
                handle_inequality_constraints=True,
                internal_algorithm_name=nlopt.LD_SLSQP,
                require_gradient=True,
                website=f"{nlopt_doc}#slsqp",
            ),
            "NLOPT_BOBYQA": NLoptAlgorithmDescription(
                algorithm_name="BOBYQA",
                description=(
                    "Bound Optimization BY Quadratic "
                    "Approximation (BOBYQA) implemented "
                    "in the NLOPT library"
                ),
                internal_algorithm_name=nlopt.LN_BOBYQA,
                website=f"{nlopt_doc}#bobyqa",
            ),
            "NLOPT_BFGS": NLoptAlgorithmDescription(
                algorithm_name="BFGS",
                description=(
                    "Broyden-Fletcher-Goldfarb-Shanno method "
                    "(BFGS) implemented in the NLOPT library"
                ),
                internal_algorithm_name=nlopt.LD_LBFGS,
                require_gradient=True,
                website=f"{nlopt_doc}#low-storage-bfgs",
            ),
            # Does not work on Rastrigin => banned
            #             'NLOPT_ESCH': { Does not work on Rastrigin
            #                 self.INTERNAL_NAME: nlopt.GN_ESCH,
            #                 self.REQUIRE_GRAD: False,
            #                 self.HANDLE_EQ_CONS: False,
            #                 self.HANDLE_INEQ_CONS: False},
            "NLOPT_NEWUOA": NLoptAlgorithmDescription(
                algorithm_name="NEWUOA",
                description=(
                    "NEWUOA + bound constraints implemented in the NLOPT library"
                ),
                internal_algorithm_name=nlopt.LN_NEWUOA_BOUND,
                website=f"{nlopt_doc}#newuoa-bound-constraints",
            ),
            # Does not work on Rastrigin => banned
            #             'NLOPT_ISRES': {
            #                 self.INTERNAL_NAME: nlopt.GN_ISRES,
            #                 self.REQUIRE_GRAD: False,
            #                 self.HANDLE_EQ_CONS: True,
            #                 self.HANDLE_INEQ_CONS: True}
        }

    def _get_options(
        self,
        ftol_abs: float = 1e-14,  # pylint: disable=W0221
        xtol_abs: float = 1e-14,
        max_time: float = 0.0,
        max_iter: int = 999,
        ftol_rel: float = 1e-8,
        xtol_rel: float = 1e-8,
        ctol_abs: float = 1e-6,
        stopval: float | None = None,
        normalize_design_space: bool = True,
        eq_tolerance: float = 1e-2,
        ineq_tolerance: float = 1e-4,
        init_step: float = 0.25,
        kkt_tol_abs: float | None = None,
        kkt_tol_rel: float | None = None,
        stop_crit_n_x: int | None = None,
        **kwargs: Any,
    ) -> dict[str, NLoptOptionsType]:
        r"""Retrieve the options of the Nlopt library.

        Args:
            ftol_abs: The absolute tolerance on the objective function.
            xtol_abs: The absolute tolerance on the design parameters.
            max_time: The maximum runtime in seconds. The value 0 means no runtime limit.
            max_iter: The maximum number of iterations.
            ftol_rel: The relative tolerance on the objective function.
            xtol_rel: The relative tolerance on the design parameters.
            ctol_abs: The absolute tolerance on the constraints.
            stopval: The objective value at which the optimization will stop.
                Stop minimizing when an objective value :math:`\leq` stopval is
                found, or stop maximizing when a value :math:`\geq` stopval
                is found. If None, this termination condition will not be active.
            kkt_tol_abs: The absolute tolerance on the KKT residual norm.
                If ``None`` this criterion is not activated.
            kkt_tol_rel: The relative tolerance on the KKT residual norm.
                If ``None`` this criterion is not activated.
            normalize_design_space: If True, normalize the design variables between 0 and 1.
            eq_tolerance: The tolerance on the equality constraints.
            ineq_tolerance: The tolerance on the inequality constraints.
            init_step: The initial step size for derivative-free algorithms.
                Increasing init_step will make the initial DOE in COBYLA
                take wider steps in the design variables. By default, each variable
                is set to x0 plus a perturbation given by
                0.25*(ub_i-x0_i) for i=0, …, len(x0)-1.
            stop_crit_n_x: The minimum number of design vectors to take into account in
                the stopping criteria.
            **kwargs: The additional algorithm-specific options.

        Returns:
            The NLopt library options with their values.
        """
        nds = normalize_design_space
        popts = self._process_options(
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            stop_crit_n_x=stop_crit_n_x,
            max_time=max_time,
            max_iter=max_iter,
            ctol_abs=ctol_abs,
            normalize_design_space=nds,
            stopval=stopval,
            eq_tolerance=eq_tolerance,
            ineq_tolerance=ineq_tolerance,
            init_step=init_step,
            kkt_tol_abs=kkt_tol_abs,
            kkt_tol_rel=kkt_tol_rel,
            **kwargs,
        )

        return popts

    def __opt_objective_grad_nlopt(
        self,
        xn_vect: ndarray,
        grad: ndarray,
    ) -> float:
        """Evaluate the objective and gradient functions for NLopt.

        Args:
            xn_vect: The normalized design variables vector.
            grad: The gradient of the objective function.

        Returns:
            The evaluation of the objective function for the given `xn_vect`.
        """
        obj_func = self.problem.objective
        if grad.size > 0:
            grad[:] = obj_func.jac(xn_vect)
        return float(obj_func.func(xn_vect).real)

    def __make_constraint(
        self,
        func: Callable[[ndarray], ndarray],
        jac: Callable[[ndarray], ndarray],
        index_cstr: int,
    ) -> Callable[[ndarray, ndarray], ndarray]:
        """Build NLopt-like constraints.

        No vector functions are allowed. The database will avoid
        multiple evaluations.

        Args:
            func: The function pointer.
            jac: The Jacobian pointer.
            index_cstr: The index of the constraint.

        Returns:
            The constraint function.
        """

        def cstr_fun_grad(
            xn_vect: ndarray,
            grad: ndarray,
        ) -> ndarray:
            """Define the function to be given as a pointer to the optimizer.

            Used to compute constraints and constraints gradients if required.

            Args:
                xn_vect: The normalized design vector.
                grad: The gradient of the objective function.

            Returns:
                The result of evaluating the function for a given constraint.
            """
            if self.descriptions[self.algo_name].require_gradient:
                if grad.size > 0:
                    cstr_jac = jac(xn_vect)
                    grad[:] = atleast_2d(cstr_jac)[index_cstr,]
            return atleast_1d(func(xn_vect).real)[index_cstr]

        return cstr_fun_grad

    def __add_constraints(
        self,
        nlopt_problem: nlopt.opt,
        ctol: float = 0.0,
    ) -> None:
        """Add all the constraints to the optimization problem.

        Args:
            nlopt_problem: The optimization problem.
            ctol: The absolute tolerance on the constraints.
        """
        for constraint in self.problem.constraints:
            f_type = constraint.f_type
            func = constraint.func
            jac = constraint.jac
            dim = constraint.dim
            for idim in range(dim):
                nl_fun = self.__make_constraint(func, jac, idim)
                if f_type == MDOFunction.TYPE_INEQ:
                    nlopt_problem.add_inequality_constraint(nl_fun, ctol)
                elif f_type == MDOFunction.TYPE_EQ:
                    nlopt_problem.add_equality_constraint(nl_fun, ctol)

    def __set_prob_options(
        self,
        nlopt_problem: nlopt.opt,
        **opt_options: Any,
    ) -> nlopt.opt:
        """Set the options for the NLopt algorithm.

        Args:
            nlopt_problem: The optimization problem from NLopt.
            **opt_options: The NLopt optimization options.

        Returns:
            The updated NLopt problem.
        """
        # ALready 0 by default
        # nlopt_problem.set_xtol_abs(0.0)
        # nlopt_problem.set_xtol_rel(0.0)
        # nlopt_problem.set_ftol_rel(0.0)
        # nlopt_problem.set_ftol_abs(0.0)
        nlopt_problem.set_maxeval(int(1.5 * opt_options[self.MAX_ITER]))  # anti-cycling
        nlopt_problem.set_maxtime(opt_options[self.MAX_TIME])
        nlopt_problem.set_initial_step(opt_options[self.INIT_STEP])
        if self.STOPVAL in opt_options:
            stopval = opt_options[self.STOPVAL]
            if stopval is not None:
                nlopt_problem.set_stopval(stopval)
        if self.INNER_MAXEVAL in opt_options:
            nlopt_problem.set_param(self.INNER_MAXEVAL, opt_options[self.INNER_MAXEVAL])

        return nlopt_problem

    def _pre_run(
        self,
        problem: OptimizationProblem,
        algo_name: str,
        **options: NLoptOptionsType,
    ):
        """Set :attr:`.STOP_CRIT_NX` depending on the algorithm.

        The COBYLA and BOBYQA algorithms create sets of interpolation points
        of sizes ``N+1`` and ``2*N+1`` respectively at initialization,
        where ``N`` is the dimension of the design space.
        In some cases, a termination criterion can be matched during this phase,
        leading to a premature termination.

        In order to circumvent this, :attr:`.STOP_CRIT_NX` is set accordingly,
        depending on the algorithm used.
        It ensures that the termination criterion will not be triggered during this
        preliminary Design of Experiment phase of the algorithm.

        Args:
            problem: The optimization problem.
            algo_name: The name of the algorithm.
            **options: The options of the algorithm,
                see the associated JSON file.
        """
        n_stop_crit_x = options[self.STOP_CRIT_NX]
        if algo_name == "NLOPT_COBYLA" and not n_stop_crit_x:
            design_space_dimension = self.problem.design_space.dimension
            options[self.STOP_CRIT_NX] = design_space_dimension + 1
        elif algo_name == "NLOPT_BOBYQA" and not n_stop_crit_x:
            design_space_dimension = self.problem.design_space.dimension
            options[self.STOP_CRIT_NX] = 2 * design_space_dimension + 1
        else:
            options[self.STOP_CRIT_NX] = n_stop_crit_x or 3
        super()._pre_run(problem, algo_name, **options)

    def _run(self, **options: NLoptOptionsType) -> OptimizationResult:
        """Run the algorithm.

        Args:
            **options: The options for the algorithm,
                see associated JSON file.

        Returns:
            The optimization result.

        Raises:
            TerminationCriterion: If the driver stops for some reason.
        """
        normalize_ds = options.pop(self.NORMALIZE_DESIGN_SPACE_OPTION, True)
        # Get the bounds anx x0
        x_0, l_b, u_b = self.get_x0_and_bounds_vects(normalize_ds)

        nlopt_problem = nlopt.opt(self.internal_algo_name, x_0.shape[0])

        # Set the normalized bounds:
        nlopt_problem.set_lower_bounds(l_b.real)
        nlopt_problem.set_upper_bounds(u_b.real)

        nlopt_problem.set_min_objective(self.__opt_objective_grad_nlopt)
        if self.CTOL_ABS in options:
            ctol = options[self.CTOL_ABS]
        self.__add_constraints(nlopt_problem, ctol)
        nlopt_problem = self.__set_prob_options(nlopt_problem, **options)
        try:
            nlopt_problem.optimize(x_0.real)
        except (RoundoffLimited, RuntimeError) as err:
            LOGGER.error(
                "NLopt run failed: %s, %s",
                str(err.args[0]),
                str(err.__class__.__name__),
            )
            raise TerminationCriterion()
        message = self.NLOPT_MESSAGES[nlopt_problem.last_optimize_result()]
        status = nlopt_problem.last_optimize_result()
        return self.get_optimum_from_database(message, status)
