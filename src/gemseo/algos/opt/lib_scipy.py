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
"""The library of SciPy optimization algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

from numpy import isfinite
from numpy import ndarray
from numpy import real
from scipy import optimize

from gemseo.algos.design_space_utils import get_value_and_bounds
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.algos.optimization_result import OptimizationResult


@dataclass
class SciPyAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of an optimization algorithm from the SciPy library."""

    library_name: str = "SciPy"


class ScipyOpt(BaseOptimizationLibrary):
    """The library of SciPy optimization algorithms."""

    _OPTIONS_MAP: ClassVar[dict[str, str]] = {
        # Available only in the doc !
        BaseOptimizationLibrary._LS_STEP_NB_MAX: "maxls",
        BaseOptimizationLibrary._LS_STEP_SIZE_MAX: "stepmx",
        BaseOptimizationLibrary._MAX_FUN_EVAL: "maxfun",
        BaseOptimizationLibrary._PG_TOL: "gtol",
    }

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
        ),
        "L-BFGS-B": SciPyAlgorithmDescription(
            algorithm_name="L-BFGS-B",
            description=(
                "Limited-memory BFGS algorithm implemented in the SciPy library"
            ),
            internal_algorithm_name="L-BFGS-B",
            require_gradient=True,
            website=f"{__DOC}optimize.minimize-lbfgsb.html",
        ),
        "TNC": SciPyAlgorithmDescription(
            algorithm_name="TNC",
            description=(
                "Truncated Newton (TNC) algorithm implemented in SciPy library"
            ),
            internal_algorithm_name="TNC",
            require_gradient=True,
            website=f"{__DOC}optimize.minimize-tnc.html",
        ),
        "NELDER-MEAD": SciPyAlgorithmDescription(
            algorithm_name="NELDER-MEAD",
            description="Nelder-Mead algorithm implemented in the SciPy library",
            internal_algorithm_name="Nelder-Mead",
            require_gradient=False,
            website=f"{__DOC}optimize.minimize-neldermead.html",
        ),
    }

    def _get_options(
        self,
        max_iter: int = 999,
        ftol_rel: float = 1e-9,
        ftol_abs: float = 1e-9,
        xtol_rel: float = 1e-9,
        xtol_abs: float = 1e-9,
        max_ls_step_size: float = 0.0,
        max_ls_step_nb: int = 20,
        max_fun_eval: int = 999,
        max_time: float = 0,
        pg_tol: float = 1e-5,
        disp: bool = False,
        maxCGit: int = -1,  # noqa: N803
        eta: float = -1.0,
        factr: float = 1e7,
        maxcor: int = 20,
        normalize_design_space: bool = True,
        eq_tolerance: float = 1e-2,
        ineq_tolerance: float = 1e-4,
        stepmx: float = 0.0,
        minfev: float = 0.0,
        scale: float | None = None,
        rescale: float = -1,
        offset: float | None = None,
        kkt_tol_abs: float | None = None,
        kkt_tol_rel: float | None = None,
        adaptive: bool = False,
        initial_simplex: Sequence[Sequence[float]] | None = None,
        stop_crit_n_x: int = 3,
        **kwargs: Any,
    ) -> dict[str, Any]:
        r"""Set the options default values.

        To get the best and up-to-date information about algorithms options,
        go to scipy.optimize documentation:
        https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html

        Args:
            max_iter: The maximum number of iterations, i.e. unique calls to f(x).
            ftol_rel: A stop criteria, the relative tolerance on the
               objective function.
               If abs(f(xk)-f(xk+1))/abs(f(xk))<= ftol_rel: stop.
            ftol_abs: A stop criteria, the absolute tolerance on the objective
               function. If abs(f(xk)-f(xk+1))<= ftol_rel: stop.
            xtol_rel: A stop criteria, the relative tolerance on the
               design variables. If norm(xk-xk+1)/norm(xk)<= xtol_rel: stop.
            xtol_abs: A stop criteria, absolute tolerance on the
               design variables.
               If norm(xk-xk+1)<= xtol_abs: stop.
            max_ls_step_size: The maximum step for the line search.
            max_ls_step_nb: The maximum number of line search steps
               per iteration.
            max_fun_eval: The internal stop criteria on the
               number of algorithm outer iterations.
            max_time: The maximum runtime in seconds, disabled if 0.
            pg_tol: A stop criteria on the projected gradient norm.
            disp: The display information flag.
            maxCGit: The maximum Conjugate Gradient internal solver
                iterations.
            eta: The severity of the line search, specific to the
                TNC algorithm.
            factr: A stop criteria on the projected gradient norm,
                stop if max_i (grad_i)<eps_mach \* factr, where eps_mach is the
                machine precision.
            maxcor: The maximum BFGS updates.
            normalize_design_space: If ``True``, scales variables to [0, 1].
            eq_tolerance: The equality tolerance.
            ineq_tolerance: The inequality tolerance.
            stepmx: The maximum step for the line search.
            minfev: The minimum function value estimate.
            scale: The scaling factor to apply to each variable.
                If ``None``, the factors are up-low for interval bounded variables
                and 1+|x| for the others.
            rescale: The scaling factor (in log10) used to trigger f value
                rescaling.
            offset: Value to subtract from each variable. If ``None``, the offsets are
                (up+low)/2 for interval bounded variables and x for the others.
            kkt_tol_abs: The absolute tolerance on the KKT residual norm.
                If ``None`` and ``kkt_tol_rel`` is ``None``,
                this criterion is not considered.
            kkt_tol_rel: The relative tolerance on the KKT residual norm.
                If ``None`` and ``kkt_tol_abs`` is ``None``,
                this criterion is not considered.
            adaptive: Whether to adapt the Nelder-Mead algorithm parameters to the
                dimensionality of the problem. Useful for high-dimensional minimization.
            initial_simplex: If not ``None``, overrides x0 in the Nelder-Mead algorithm.
                ``initial_simplex[j,:]`` should contain the coordinates of the jth
                vertex of the N+1 vertices in the simplex, where N is the dimension.
            stop_crit_n_x: The minimum number of design vectors to take into account in
                the stopping criteria.
            **kwargs: The other algorithm options.
        """
        return self._process_options(
            max_iter=max_iter,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            max_time=max_time,
            max_ls_step_size=max_ls_step_size,
            max_ls_step_nb=max_ls_step_nb,
            max_fun_eval=max_fun_eval,
            pg_tol=pg_tol,
            disp=disp,
            maxCGit=maxCGit,  # noqa: N803
            eta=eta,
            factr=factr,
            maxcor=maxcor,
            normalize_design_space=normalize_design_space,
            ineq_tolerance=ineq_tolerance,
            eq_tolerance=eq_tolerance,
            stepmx=stepmx,
            minfev=minfev,
            scale=scale,
            rescale=rescale,
            offset=offset,
            adaptive=adaptive,
            initial_simplex=initial_simplex,
            kkt_tol_abs=kkt_tol_abs,
            kkt_tol_rel=kkt_tol_rel,
            stop_crit_n_x=stop_crit_n_x,
            **kwargs,
        )

    def _run(self, problem: OptimizationProblem, **options: Any) -> OptimizationResult:
        # remove normalization from options for algo
        normalize_ds = options.pop(self._NORMALIZE_DESIGN_SPACE_OPTION, True)
        # Get the normalized bounds:
        x_0, l_b, u_b = get_value_and_bounds(problem.design_space, normalize_ds)
        # Replace infinite values with None:
        l_b = [val if isfinite(val) else None for val in l_b]
        u_b = [val if isfinite(val) else None for val in u_b]
        bounds = list(zip(l_b, u_b))

        def real_part_fun(
            x: ndarray,
        ) -> int | float:
            """Wrap the function and return the real part.

            Args:
                x: The values to be given to the function.

            Returns:
                The real part of the evaluation of the objective function.
            """
            return real(problem.objective.evaluate(x))

        fun = real_part_fun

        constraints = self._get_right_sign_constraints(problem)
        cstr_scipy = []
        for cstr in constraints:
            c_scipy = {"type": cstr.f_type, "fun": cstr.evaluate, "jac": cstr.jac}
            cstr_scipy.append(c_scipy)
        jac = problem.objective.jac

        # |g| is in charge of ensuring max iterations, and
        # xtol, ftol, since it may
        # have a different definition of iterations, such as for SLSQP
        # for instance which counts duplicate calls to x as a new iteration
        options["maxfun" if self._algo_name == "TNC" else "maxiter"] = 10000000

        # Deactivate scipy stop criteria to use |g|' ones
        options["ftol"] = 0.0
        options["xtol"] = 0.0
        options.pop(self._F_TOL_ABS)
        options.pop(self._X_TOL_ABS)
        options.pop(self._F_TOL_REL)
        options.pop(self._X_TOL_REL)
        options.pop(self._MAX_TIME)
        options.pop(self._MAX_ITER)
        options.pop(self._KKT_TOL_REL)
        options.pop(self._KKT_TOL_ABS)
        del options[self._STOP_CRIT_NX]

        if self._algo_name != "TNC":
            options.pop("xtol")

        if self._algo_name == "NELDER-MEAD":
            options["fatol"] = 0.0
            options["xatol"] = 0.0
            options.pop("ftol")
            jac = None

        opt_result = optimize.minimize(
            fun=fun,
            x0=x_0,
            method=self.ALGORITHM_INFOS[self._algo_name].internal_algorithm_name,
            jac=jac,
            bounds=bounds,
            constraints=cstr_scipy,
            options=options,
        )

        return self._get_optimum_from_database(
            problem, opt_result.message, opt_result.status
        )
