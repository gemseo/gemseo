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
#        :author: Damien Guenot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#         Francois Gallard : refactoring for v1, May 2016
"""
scipy.optimize optimization library wrapper
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from builtins import super, zip

from future import standard_library
from numpy import isfinite, real
from scipy import optimize

from gemseo.algos.opt.opt_lib import OptimizationLibrary

standard_library.install_aliases()


from gemseo import LOGGER


class ScipyOpt(OptimizationLibrary):
    """Scipy optimization library interface.

    See OptimizationLibrary.
    """

    LIB_COMPUTE_GRAD = True

    OPTIONS_MAP = {
        OptimizationLibrary.MAX_ITER: "maxiter",
        OptimizationLibrary.F_TOL_REL: "ftol",
        OptimizationLibrary.X_TOL_REL: "xtol",
        # Available only in the doc !
        # OptimizationLibrary.LS_STEP_NB_MAX: "maxls",
        OptimizationLibrary.LS_STEP_SIZE_MAX: "stepmx",
        OptimizationLibrary.MAX_FUN_EVAL: "maxfun",
        OptimizationLibrary.PG_TOL: "gtol",
    }

    def __init__(self):
        """
        Constructor

        Generate the library dict, contains the list
        of algorithms with their characteristics:

        - does it require gradient
        - does it handle equality constraints
        - does it handle inequality constraints

        """
        super(ScipyOpt, self).__init__()
        doc = "https://docs.scipy.org/doc/scipy/reference/"
        self.lib_dict = {
            "SLSQP": {
                self.INTERNAL_NAME: "SLSQP",
                self.REQUIRE_GRAD: True,
                self.POSITIVE_CONSTRAINTS: True,
                self.HANDLE_EQ_CONS: True,
                self.HANDLE_INEQ_CONS: True,
                self.DESCRIPTION: "Sequential Least-Squares Quadratic "
                "Programming (SLSQP) implemented in "
                "the SciPy library",
                self.WEBSITE: doc + "optimize.minimize-slsqp.html",
            },
            "L-BFGS-B": {
                self.INTERNAL_NAME: "L-BFGS-B",
                self.REQUIRE_GRAD: True,
                self.HANDLE_EQ_CONS: False,
                self.HANDLE_INEQ_CONS: False,
                self.DESCRIPTION: "Limited-memory BFGS algorithm "
                "implemented in SciPy library",
                self.WEBSITE: doc + "generated/scipy.optimize.fmin_l_bfgs_b.html",
            },
            "TNC": {
                self.INTERNAL_NAME: "TNC",
                self.REQUIRE_GRAD: True,
                self.HANDLE_EQ_CONS: False,
                self.HANDLE_INEQ_CONS: False,
                self.DESCRIPTION: "Truncated Newton (TNC) algorithm "
                "implemented in SciPy library",
                self.WEBSITE: doc + "optimize.minimize-tnc.html",
            },
        }

    def _get_options(
        self,
        max_iter=999,
        ftol_rel=1e-9,  # pylint: disable=W0221
        ftol_abs=1e-9,
        xtol_rel=1e-9,
        xtol_abs=1e-9,
        max_ls_step_size=0.0,
        max_ls_step_nb=20,
        max_fun_eval=999,
        max_time=-1,
        pg_tol=1e-5,
        disp=0,
        maxCGit=-1,
        eta=-1.0,
        factr=1e7,
        maxcor=20,
        normalize_design_space=True,
        eq_tolerance=1e-2,
        ineq_tolerance=1e-4,
        stepmx=0.0,
        minfev=0.0,
        scale=None,
        rescale=None,
        offset=None,
        **kwargs
    ):
        r"""Sets the options default values

        To get the best and up to date information about algorithms options,
        go to scipy.optimize documentation:
        https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html

        :param max_iter: maximum number of iterations, ie unique calls to f(x)
        :type max_iter: int
        :param ftol_rel: stop criteria, relative tolerance on the
               objective function,
               if abs(f(xk)-f(xk+1))/abs(f(xk))<= ftol_rel: stop
               (Default value = 1e-9)
        :type ftol_rel: float
        :param ftol_abs: stop criteria, absolute tolerance on the objective
               function, if abs(f(xk)-f(xk+1))<= ftol_rel: stop
               (Default value = 1e-9)
        :type ftol_abs: float
        :param xtol_rel: stop criteria, relative tolerance on the
               design variables,
               if norm(xk-xk+1)/norm(xk)<= xtol_rel: stop
               (Default value = 1e-9)
        :type xtol_rel: float
        :param xtol_abs: stop criteria, absolute tolerance on the
               design variables,
               if norm(xk-xk+1)<= xtol_abs: stop
               (Default value = 1e-9)
        :type xtol_abs: float
        :param max_ls_step_size: Maximum step for the line search
               (Default value = 0.)
        :type max_ls_step_size: float
        :param max_ls_step_nb: Maximum number of line search steps
               per iteration. (Default value = 20)
        :type max_ls_step_nb: int
        :param max_fun_eval: internal stop criteria on the
               number of algorithm outer iterations (Default value = 999)
        :type max_ls_step_size: int
        :param max_time: maximum runtime (Default value = -1)
        :type max_time: float
        :param pg_tol: stop criteria on the projected gradient norm
               (Default value = 1e-5)
        :type pg_tol: float
        :param disp: display information, (Default value = 0)
        :type disp: int
        :param maxCGit: Maximum Conjugate Gradient internal solver
               iterations (Default value = -1)
        :type maxCGit: int
        :param eta: severity of the linesearch, specific to
               TNC algorithm (Default value = -1.)
        :type eta: int
        :param factr: stop criteria on the projected gradient norm,
               stop if max_i (grad_i)<eps_mach \* factr, where eps_mach is the
               machine precision ( Default value = 1e7)
        :type factr: float
        :param maxcor: maximum BFGS updates (Default value = 20)
        :type maxcor: int
        :param normalize_design_space: If True, scales variables in [0, 1]
        :type normalize_design_space: bool
        :param eq_tolerance: equality tolerance
        :type eq_tolerance: float
        :param ineq_tolerance: inequality tolerance
        :type ineq_tolerance: float
        :param stepmx: Maximum step for the line search.
        :type stepmx: float
        :param minfev: Minimum function value estimate
        :type minfev: float
        :param scale: Scaling factors to apply to each variable
        :type scale: array
        :param rescale: Scaling factor (in log10) used to trigger f value
            rescaling
        :type rescale: float
        :param offset: Value to subtract from each variable.
        :type offset: float
        :param kwargs: other algorithms options
        :tupe kwargs: kwargs
        """
        nds = normalize_design_space
        popts = self._process_options(
            max_iter=max_iter,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            max_ls_step_size=max_ls_step_size,
            max_ls_step_nb=max_ls_step_nb,
            max_fun_eval=max_fun_eval,
            max_time=max_time,
            pg_tol=pg_tol,
            disp=disp,
            maxCGit=maxCGit,
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
            **kwargs
        )
        return popts

    def _run(self, **options):
        """Runs the algorithm, to be overloaded by subclasses

        :param options: the options dict for the algorithm
        """
        # remove normalization from options for algo
        normalize_ds = options.pop(self.NORMALIZE_DESIGN_SPACE_OPTION, True)
        # Get the normalized bounds:
        x_0, l_b, u_b = self.get_x0_and_bounds_vects(normalize_ds)
        # Replace infinite values with None:
        l_b = [val if isfinite(val) else None for val in l_b]
        u_b = [val if isfinite(val) else None for val in u_b]
        bounds = list(zip(l_b, u_b))

        def real_part_fun(x_vect):
            """
            Wraps the function and returns the real part
            """
            return real(self.problem.objective.func(x_vect))

        fun = real_part_fun

        constraints = self.get_right_sign_constraints()
        cstr_scipy = []
        for cstr in constraints:
            c_scipy = {"type": cstr.f_type, "fun": cstr.func, "jac": cstr.jac}
            cstr_scipy.append(c_scipy)
        jac = self.problem.objective.jac

        # |g| is in charge of ensuring max iterations, since it may
        # have a different definition of iterations, such as for SLSQP
        # for instance which counts duplicate calls to x as a new iteration
        options[self.OPTIONS_MAP[self.MAX_ITER]] = 10000000
        opt_result = optimize.minimize(
            fun=fun,
            x0=x_0,
            method=self.internal_algo_name,
            jac=jac,
            bounds=bounds,
            constraints=cstr_scipy,
            tol=None,
            options=options,
        )

        return self.get_optimum_from_database(opt_result.message, opt_result.status)
