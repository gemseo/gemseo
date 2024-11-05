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
"""The library of NLopt optimization algorithms.

Warnings:
    If the objective, or a constraint, of the :class:`.OptimizationProblem`
    returns a value of type ``int``
    then ``nlopt.opt.optimize`` will terminate with
    ``ValueError: nlopt invalid argument``.

    This behavior has been identified as
    `a bug internal to NLopt 2.7.1 <https://github.com/stevengj/nlopt/issues/530>`_
    and has been fixed in the development version of NLopt.

    Until a new version of NLopt including the bugfix is released,
    the user of |g| shall provide objective and constraint functions
    that return values of type ``float`` and ``NDArray[float]``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Final
from typing import Union

from nlopt import LD_LBFGS
from nlopt import LD_MMA
from nlopt import LD_SLSQP
from nlopt import LN_BOBYQA
from nlopt import LN_COBYLA
from nlopt import LN_NEWUOA_BOUND
from nlopt import RoundoffLimited
from nlopt import opt
from numpy import array
from numpy import atleast_1d
from numpy import atleast_2d
from numpy import ndarray

from gemseo.algos.design_space_utils import get_value_and_bounds
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.opt.nlopt.settings.base_nlopt_settings import BaseNLoptSettings
from gemseo.algos.opt.nlopt.settings.nlopt_bfgs_settings import NLOPT_BFGS_Settings
from gemseo.algos.opt.nlopt.settings.nlopt_bobyqa_settings import NLOPT_BOBYQA_Settings
from gemseo.algos.opt.nlopt.settings.nlopt_cobyla_settings import NLOPT_COBYLA_Settings
from gemseo.algos.opt.nlopt.settings.nlopt_mma_settings import NLOPT_MMA_Settings
from gemseo.algos.opt.nlopt.settings.nlopt_newuoa_settings import NLOPT_NEWUOA_Settings
from gemseo.algos.opt.nlopt.settings.nlopt_slsqp_settings import NLOPT_SLSQP_Settings
from gemseo.algos.stop_criteria import TerminationCriterion
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.utils.constants import C_LONG_MAX

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import OptimizationProblem

LOGGER = logging.getLogger(__name__)

NLoptOptionsType = Union[bool, int, float]


@dataclass
class NLoptAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of an optimization algorithm from the NLopt library."""

    library_name: str = "NLopt"

    Settings: type[BaseNLoptSettings] = BaseNLoptSettings
    """The option validation model for NLopt optimization library."""


class Nlopt(BaseOptimizationLibrary):
    """The library of NLopt optimization algorithms."""

    _INIT_STEP: Final[str] = "init_step"
    _INNER_MAXEVAL: Final[str] = "inner_maxeval"
    _STOPVAL: Final[str] = "stopval"

    __NLOPT_MESSAGES: ClassVar[dict[int, str]] = {
        1: "NLOPT_SUCCESS: Generic success return value",
        2: (
            "NLOPT_STOPVAL_REACHED: Optimization stopped  "
            "because stopval (above) was reached"
        ),
        3: (
            "NLOPT_FTOL_REACHED: Optimization stopped "
            "because ftol_rel or ftol_abs (above) was reached"
        ),
        4: (
            "NLOPT_XTOL_REACHED Optimization stopped "
            "because xtol_rel or xtol_abs (above) was reached"
        ),
        5: (
            "NLOPT_MAXEVAL_REACHED: Optimization stopped "
            "because maxeval (above) was reached"
        ),
        6: (
            "NLOPT_MAXTIME_REACHED: Optimization stopped "
            "because maxtime (above) was reached"
        ),
        -1: "NLOPT_FAILURE:    Generic failure code",
        -2: (
            "NLOPT_INVALID_ARGS: Invalid arguments (e.g. lower "
            "bounds are bigger than upper bounds, an unknown"
            " algorithm was specified, etcetera)."
        ),
        -3: "OUT_OF_MEMORY: Ran out of memory",
        -4: (
            "NLOPT_ROUNDOFF_LIMITED: Halted because "
            "roundoff errors limited progress. (In this "
            "case, the optimization still typically "
            "returns a useful result.)"
        ),
        -5: (
            "NLOPT_FORCED_STOP: Halted because of a forced "
            "termination: the user called nlopt_force_stop"
            "(opt) on the optimization's nlopt_opt"
            " object opt from the user's objective "
            "function or constraints."
        ),
    }

    __NLOPT_DOC = "https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/"

    ALGORITHM_INFOS: ClassVar[dict[str, NLoptAlgorithmDescription]] = {
        "NLOPT_MMA": NLoptAlgorithmDescription(
            algorithm_name="MMA",
            description=(
                "Method of Moving Asymptotes (MMA)implemented in the NLOPT library"
            ),
            handle_inequality_constraints=True,
            internal_algorithm_name=LD_MMA,
            require_gradient=True,
            website=f"{__NLOPT_DOC}#mma-method-of-moving-asymptotes-and-ccsa",
            Settings=NLOPT_MMA_Settings,
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
            internal_algorithm_name=LN_COBYLA,
            website=(
                f"{__NLOPT_DOC}#cobyla-constrained-optimization-by-linear-"
                "approximations"
            ),
            Settings=NLOPT_COBYLA_Settings,
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
            internal_algorithm_name=LD_SLSQP,
            require_gradient=True,
            website=f"{__NLOPT_DOC}#slsqp",
            Settings=NLOPT_SLSQP_Settings,
        ),
        "NLOPT_BOBYQA": NLoptAlgorithmDescription(
            algorithm_name="BOBYQA",
            description=(
                "Bound Optimization BY Quadratic "
                "Approximation (BOBYQA) implemented "
                "in the NLOPT library"
            ),
            internal_algorithm_name=LN_BOBYQA,
            website=f"{__NLOPT_DOC}#bobyqa",
            Settings=NLOPT_BOBYQA_Settings,
        ),
        "NLOPT_BFGS": NLoptAlgorithmDescription(
            algorithm_name="BFGS",
            description=(
                "Broyden-Fletcher-Goldfarb-Shanno method "
                "(BFGS) implemented in the NLOPT library"
            ),
            internal_algorithm_name=LD_LBFGS,
            require_gradient=True,
            website=f"{__NLOPT_DOC}#low-storage-bfgs",
            Settings=NLOPT_BFGS_Settings,
        ),
        "NLOPT_NEWUOA": NLoptAlgorithmDescription(
            algorithm_name="NEWUOA",
            description=("NEWUOA + bound constraints implemented in the NLOPT library"),
            internal_algorithm_name=LN_NEWUOA_BOUND,
            website=f"{__NLOPT_DOC}#newuoa-bound-constraints",
            Settings=NLOPT_NEWUOA_Settings,
        ),
    }

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
        obj_func = self._problem.objective
        if grad.size > 0:
            grad[:] = obj_func.jac(xn_vect)
        return array(obj_func.evaluate(xn_vect).real).ravel()[0]

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
            if self.ALGORITHM_INFOS[self._algo_name].require_gradient and grad.size > 0:
                cstr_jac = jac(xn_vect)
                grad[:] = atleast_2d(cstr_jac)[index_cstr,]
            return atleast_1d(func(xn_vect).real)[index_cstr]

        return cstr_fun_grad

    def __add_constraints(self, nlopt_problem: opt, **settings: Any) -> None:
        """Add all the constraints to the optimization problem.

        Args:
            nlopt_problem: The optimization problem.
            **settings: The NLopt optimizer settings.
        """
        for constraint in self._problem.constraints:
            f_type = constraint.f_type
            func = constraint.evaluate
            jac = constraint.jac
            dim = constraint.dim
            for idim in range(dim):
                nl_fun = self.__make_constraint(func, jac, idim)
                if f_type == MDOFunction.ConstraintType.INEQ:
                    nlopt_problem.add_inequality_constraint(
                        nl_fun, settings[self._INEQ_TOLERANCE]
                    )
                elif f_type == MDOFunction.ConstraintType.EQ:
                    nlopt_problem.add_equality_constraint(
                        nl_fun, settings[self._EQ_TOLERANCE]
                    )

    def __set_prob_options(self, nlopt_problem: opt, **settings: Any) -> None:
        """Set the options for the NLopt algorithm.

        Args:
            nlopt_problem: The optimization problem from NLopt.
            **settings: The NLopt optimizer settings.
        """
        # Deactivate stopping criteria which are handled by GEMSEO
        nlopt_problem.set_ftol_abs(0.0)
        nlopt_problem.set_ftol_rel(0.0)

        nlopt_problem.set_xtol_abs(0.0)
        nlopt_problem.set_xtol_rel(0.0)

        nlopt_problem.set_maxtime(C_LONG_MAX)

        # Only set an initial step size for derivative-free optimization algorithms.
        if not self.ALGORITHM_INFOS[self.algo_name].require_gradient:
            nlopt_problem.set_initial_step(settings[self._INIT_STEP])

        max_eval = int(1.5 * settings[self._MAX_ITER])
        nlopt_problem.set_maxeval(max_eval)  # Anti-cycling
        nlopt_problem.set_stopval(settings[self._STOPVAL])

        if self.algo_name == "NLOPT_MMA":
            nlopt_problem.set_param(self._INNER_MAXEVAL, settings[self._INNER_MAXEVAL])

    def _pre_run(
        self,
        problem: OptimizationProblem,
        **settings: NLoptOptionsType,
    ) -> None:
        """Set ``"stop_crit_n_x"`` depending on the algorithm.

        The COBYLA and BOBYQA algorithms create sets of interpolation points
        of sizes ``N+1`` and ``2*N+1`` respectively at initialization,
        where ``N`` is the dimension of the design space.
        In some cases, a termination criterion can be matched during this phase,
        leading to a premature termination.

        In order to circumvent this, ``"stop_crit_n_x"`` is set accordingly,
        depending on the algorithm used.
        It ensures that the termination criterion will not be triggered during this
        preliminary Design of Experiment phase of the algorithm.
        """
        algo_name = self._algo_name
        if algo_name == "NLOPT_COBYLA" and not settings[self._STOP_CRIT_NX]:
            settings[self._STOP_CRIT_NX] = problem.design_space.dimension + 1
        elif algo_name == "NLOPT_BOBYQA" and not settings[self._STOP_CRIT_NX]:
            settings[self._STOP_CRIT_NX] = 2 * problem.design_space.dimension + 1
        else:
            settings[self._STOP_CRIT_NX] = settings[self._STOP_CRIT_NX] or 3

        super()._pre_run(problem, **settings)

    def _run(
        self,
        problem: OptimizationProblem,
        **settings: NLoptOptionsType,
    ) -> tuple[str, Any]:
        """
        Raises:
            TerminationCriterion: If the driver stops for some reason.
        """  # noqa: D205, D212
        # Get the bounds anx x0
        x_0, l_b, u_b = get_value_and_bounds(problem.design_space, self._normalize_ds)

        nlopt_problem = opt(
            self.ALGORITHM_INFOS[self._algo_name].internal_algorithm_name, x_0.shape[0]
        )

        # Set the normalized bounds:
        nlopt_problem.set_lower_bounds(l_b.real)
        nlopt_problem.set_upper_bounds(u_b.real)

        nlopt_problem.set_min_objective(self.__opt_objective_grad_nlopt)

        self.__set_prob_options(nlopt_problem, **settings)
        self.__add_constraints(nlopt_problem, **settings)
        try:
            nlopt_problem.optimize(x_0.real)
        except (RoundoffLimited, RuntimeError) as err:
            LOGGER.exception(
                "NLopt run failed: %s, %s",
                str(err.args[0]),
                str(err.__class__.__name__),
            )
            raise TerminationCriterion from None
        message = self.__NLOPT_MESSAGES[nlopt_problem.last_optimize_result()]
        status = nlopt_problem.last_optimize_result()
        return message, status
