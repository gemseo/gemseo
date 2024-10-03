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
"""The library of SciPy local optimization algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

from numpy import isfinite
from numpy import real
from scipy.optimize import minimize

from gemseo.algos.design_space_utils import get_value_and_bounds
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.opt.base_optimization_library_settings import (
    BaseOptimizationLibrarySettings,
)
from gemseo.algos.opt.scipy_local._base_scipy_local_settings import (
    BaseScipyLocalSettings,
)
from gemseo.algos.opt.scipy_local.settings.cobyqa import COBYQASettings
from gemseo.algos.opt.scipy_local.settings.lbfgsb import LBFGSBSettings
from gemseo.algos.opt.scipy_local.settings.nelder_mead import NelderMeadSettings
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQPSettings
from gemseo.algos.opt.scipy_local.settings.tnc import TNCSettings
from gemseo.utils.compatibility.scipy import SCIPY_GREATER_THAN_1_14

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.algos.optimization_result import OptimizationResult


@dataclass
class SciPyAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of the SciPy local optimization library."""

    library_name: str = "SciPy Local"
    """The library name."""

    settings: type[BaseScipyLocalSettings] = BaseScipyLocalSettings
    """The option validation model for SciPy local optimization library."""


class ScipyOpt(BaseOptimizationLibrary):
    """The library of SciPy optimization algorithms."""

    __DOC: Final[str] = "https://docs.scipy.org/doc/scipy/reference/"

    ALGORITHM_INFOS: ClassVar[dict[str, SciPyAlgorithmDescription]] = {
        "SLSQP": SciPyAlgorithmDescription(
            algorithm_name="SLSQP",
            description=(
                "Sequential Least-Squares Quadratic Programming (SLSQP) "
                "implemented in the SciPy library"
            ),
            handle_equality_constraints=True,
            handle_inequality_constraints=True,
            internal_algorithm_name="SLSQP",
            require_gradient=True,
            positive_constraints=True,
            website=f"{__DOC}optimize.minimize-slsqp.html",
            settings=SLSQPSettings,
        ),
        "L-BFGS-B": SciPyAlgorithmDescription(
            algorithm_name="L-BFGS-B",
            description=(
                "Limited-memory BFGS algorithm implemented in the SciPy library"
            ),
            internal_algorithm_name="L-BFGS-B",
            require_gradient=True,
            website=f"{__DOC}optimize.minimize-lbfgsb.html",
            settings=LBFGSBSettings,
        ),
        "TNC": SciPyAlgorithmDescription(
            algorithm_name="TNC",
            description=(
                "Truncated Newton (TNC) algorithm implemented in SciPy library"
            ),
            internal_algorithm_name="TNC",
            require_gradient=True,
            website=f"{__DOC}optimize.minimize-tnc.html",
            settings=TNCSettings,
        ),
        "NELDER-MEAD": SciPyAlgorithmDescription(
            algorithm_name="NELDER-MEAD",
            description="Nelder-Mead algorithm implemented in the SciPy library",
            internal_algorithm_name="Nelder-Mead",
            website=f"{__DOC}optimize.minimize-neldermead.html",
            settings=NelderMeadSettings,
        ),
    }

    if SCIPY_GREATER_THAN_1_14:
        ALGORITHM_INFOS["COBYQA"] = SciPyAlgorithmDescription(
            algorithm_name="COBYQA",
            description=(
                "Derivative-free trust-region SQP method "
                "based on quadratic models for constrained optimization."
            ),
            internal_algorithm_name="COBYQA",
            handle_equality_constraints=True,
            handle_inequality_constraints=True,
            positive_constraints=True,
            website=f"{__DOC}optimize.minimize-cobyqa.html",
            settings=COBYQASettings,
        )

    def _run(self, problem: OptimizationProblem, **options: Any) -> OptimizationResult:
        # Get the normalized bounds:
        x_0, l_b, u_b = get_value_and_bounds(problem.design_space, self._normalize_ds)
        # Replace infinite values with None:
        l_b = [val if isfinite(val) else None for val in l_b]
        u_b = [val if isfinite(val) else None for val in u_b]
        bounds = list(zip(l_b, u_b))

        # Get constraint in SciPy format
        scipy_constraints = [
            {
                "type": constraint.f_type,
                "fun": constraint.evaluate,
                "jac": constraint.jac,
            }
            for constraint in self._get_right_sign_constraints(problem)
        ]

        # Filter settings to get only the scipy.optimize.minimize ones
        options_ = self._filter_settings(options, BaseOptimizationLibrarySettings)

        opt_result = minimize(
            fun=lambda x: real(problem.objective.evaluate(x)),
            x0=x_0,
            method=self.ALGORITHM_INFOS[self._algo_name].internal_algorithm_name,
            jac=problem.objective.jac,
            bounds=bounds,
            constraints=scipy_constraints,
            options=options_,
            tol=0.0,  # Let GEMSEO monitor the convergence
        )

        return self._get_optimum_from_database(
            problem, opt_result.message, opt_result.status
        )