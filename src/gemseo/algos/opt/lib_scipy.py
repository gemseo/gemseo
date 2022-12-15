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
"""scipy.optimize optimization library wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from numpy import isfinite
from numpy import ndarray
from numpy import real
from scipy import optimize

from gemseo.algos.opt.opt_lib import OptimizationAlgorithmDescription
from gemseo.algos.opt.opt_lib import OptimizationLibrary
from gemseo.algos.opt_result import OptimizationResult


@dataclass
class SciPyAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of an optimization algorithm from the SciPy library."""

    library_name: str = "SciPy"


class ScipyOpt(OptimizationLibrary):
    """Scipy optimization library interface.

    See OptimizationLibrary.
    """

    LIB_COMPUTE_GRAD = True

    OPTIONS_MAP = {
        # Available only in the doc !
        OptimizationLibrary.LS_STEP_NB_MAX: "maxls",
        OptimizationLibrary.LS_STEP_SIZE_MAX: "stepmx",
        OptimizationLibrary.MAX_FUN_EVAL: "maxfun",
        OptimizationLibrary.PG_TOL: "gtol",
    }

    LIBRARY_NAME = "SciPy"

    def __init__(self):
        """Constructor.

        Generate the library dict, contains the list
        of algorithms with their characteristics:

        - does it require gradient
        - does it handle equality constraints
        - does it handle inequality constraints
        """
        super().__init__()
        doc = "https://docs.scipy.org/doc/scipy/reference/"
        self.descriptions = {
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
                website=f"{doc}optimize.minimize-slsqp.html",
            ),
            "L-BFGS-B": SciPyAlgorithmDescription(
                algorithm_name="L-BFGS-B",
                description=(
                    "Limited-memory BFGS algorithm implemented in SciPy library"
                ),
                internal_algorithm_name="L-BFGS-B",
                require_gradient=True,
                website=f"{doc}generated/scipy.optimize.fmin_l_bfgs_b.html",
            ),
            "TNC": SciPyAlgorithmDescription(
                algorithm_name="TNC",
                description=(
                    "Truncated Newton (TNC) algorithm implemented in SciPy library"
                ),
                internal_algorithm_name="TNC",
                require_gradient=True,
                website=f"{doc}optimize.minimize-tnc.html",
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
        disp: int = 0,
        maxCGit: int = -1,  # noqa: N803
        eta: float = -1.0,
        factr: float = 1e7,
        maxcor: int = 20,
        normalize_design_space: int = True,
        eq_tolerance: float = 1e-2,
        ineq_tolerance: float = 1e-4,
        stepmx: float = 0.0,
        minfev: float = 0.0,
        scale: float | None = None,
        rescale: float = -1,
        offset: float | None = None,
        kkt_tol_abs: float | None = None,
        kkt_tol_rel: float | None = None,
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
            normalize_design_space: If True, scales variables to [0, 1].
            eq_tolerance: The equality tolerance.
            ineq_tolerance: The inequality tolerance.
            stepmx: The maximum step for the line search.
            minfev: The minimum function value estimate.
            scale: The scaling factor to apply to each variable.
                If None, the factors are up-low for interval bounded variables
                and 1+|x| for the others.
            rescale: The scaling factor (in log10) used to trigger f value
                rescaling.
            offset: Value to subtract from each variable. If None, the offsets are
                (up+low)/2 for interval bounded variables and x for the others.
            kkt_tol_abs: The absolute tolerance on the KKT residual norm.
                If ``None`` this criterion is not activated.
            kkt_tol_rel: The relative tolerance on the KKT residual norm.
                If ``None`` this criterion is not activated.
            **kwargs: The other algorithm options.
        """
        nds = normalize_design_space
        popts = self._process_options(
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
            normalize_design_space=nds,
            ineq_tolerance=ineq_tolerance,
            eq_tolerance=eq_tolerance,
            stepmx=stepmx,
            minfev=minfev,
            scale=scale,
            rescale=rescale,
            offset=offset,
            kkt_tol_abs=kkt_tol_abs,
            kkt_tol_rel=kkt_tol_rel,
            **kwargs,
        )
        return popts

    def _run(self, **options: Any) -> OptimizationResult:
        """Run the algorithm, to be overloaded by subclasses.

        Args:
            **options: The options for the algorithm.

        Returns:
            The optimization result.
        """
        # remove normalization from options for algo
        normalize_ds = options.pop(self.NORMALIZE_DESIGN_SPACE_OPTION, True)
        # Get the normalized bounds:
        x_0, l_b, u_b = self.get_x0_and_bounds_vects(normalize_ds)
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
            return real(self.problem.objective.func(x))

        fun = real_part_fun

        constraints = self.get_right_sign_constraints()
        cstr_scipy = []
        for cstr in constraints:
            c_scipy = {"type": cstr.f_type, "fun": cstr.func, "jac": cstr.jac}
            cstr_scipy.append(c_scipy)
        jac = self.problem.objective.jac

        # |g| is in charge of ensuring max iterations, and
        # xtol, ftol, since it may
        # have a different definition of iterations, such as for SLSQP
        # for instance which counts duplicate calls to x as a new iteration
        options["maxiter"] = 10000000

        # Deactivate scipy stop criteria to use |g|' ones
        options["ftol"] = 0.0
        options["xtol"] = 0.0
        options.pop(self.F_TOL_ABS)
        options.pop(self.X_TOL_ABS)
        options.pop(self.F_TOL_REL)
        options.pop(self.X_TOL_REL)
        options.pop(self.MAX_TIME)
        options.pop(self.MAX_ITER)
        options.pop(self._KKT_TOL_REL)
        options.pop(self._KKT_TOL_ABS)
        if self.algo_name != "TNC":
            options.pop("xtol")

        opt_result = optimize.minimize(
            fun=fun,
            x0=x_0,
            method=self.internal_algo_name,
            jac=jac,
            bounds=bounds,
            constraints=cstr_scipy,
            options=options,
        )

        return self.get_optimum_from_database(opt_result.message, opt_result.status)
