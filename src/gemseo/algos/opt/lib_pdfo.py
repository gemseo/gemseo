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
#        :author: Jean-Christophe Giret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""PDFO optimization library wrapper See www.pdfo.net."""
from __future__ import absolute_import, division, print_function

import logging
from builtins import super, zip

from numpy import inf, isfinite, real
from pdfo import pdfo

from gemseo.algos.opt.opt_lib import OptimizationLibrary
from gemseo.utils.py23_compat import PY2

LOGGER = logging.getLogger(__name__)


class PDFOOpt(OptimizationLibrary):
    """PDFO optimization library interface.

    See OptimizationLibrary.
    """

    LIB_COMPUTE_GRAD = False

    OPTIONS_MAP = {
        OptimizationLibrary.MAX_ITER: "max_iter",
    }

    def __init__(self):
        """Constructor.

        Generate the library dict, contains the list
        of algorithms with their characteristics:

        - does it require gradient
        - does it handle equality constraints
        - does it handle inequality constraints
        """
        super(PDFOOpt, self).__init__()
        doc = "www.pdfo.net"
        self.lib_dict = {
            "PDFO_COBYLA": {
                self.INTERNAL_NAME: "cobyla",
                self.REQUIRE_GRAD: False,
                self.POSITIVE_CONSTRAINTS: True,
                self.HANDLE_EQ_CONS: True,
                self.HANDLE_INEQ_CONS: True,
                self.DESCRIPTION: "Constrained Optimization"
                "By Linear Approximations ",
                self.WEBSITE: doc,
            },
            "PDFO_BOBYQA": {
                self.INTERNAL_NAME: "bobyqa",
                self.REQUIRE_GRAD: False,
                self.HANDLE_EQ_CONS: False,
                self.HANDLE_INEQ_CONS: False,
                self.DESCRIPTION: "Bound Optimization By " "Quadratic Approximation",
                self.WEBSITE: doc,
            },
            "PDFO_NEWUOA": {
                self.INTERNAL_NAME: "newuoa",
                self.REQUIRE_GRAD: False,
                self.HANDLE_EQ_CONS: False,
                self.HANDLE_INEQ_CONS: False,
                self.DESCRIPTION: "NEWUOA",
                self.WEBSITE: doc,
            },
        }
        self.name = "PDFO"

    def _get_options(
        self,
        ftol_rel=1e-12,
        ftol_abs=1e-12,
        xtol_rel=1e-12,
        xtol_abs=1e-12,
        max_time=0,
        rhobeg=0.5,
        rhoend=1e-6,
        max_iter=500,
        ftarget=-inf,
        scale=False,
        quiet=True,
        classical=False,
        debug=False,
        chkfunval=False,
        ensure_bounds=True,
        normalize_design_space=True,
        **kwargs
    ):
        r"""Sets the options default values

        To get the best and up to date information about algorithms options,
        go to pdfo documentation (www.pdfo.net)

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
        :param max_time: maximum runtime in seconds,
            disabled if 0 (Default value = 0)
        :type max_time: float
        :param rhobeg: Initial value of the trust region radius.
        :type max_iter: float
        :param rhoend:  Final value of the trust region radius.  Indicate
        the accuracy required in the final values of the variables
        :type rhoend: float
        :param maxfev:  Upper bound of the number of calls of the objective
        function `fun`.
        :type maxfev: int
        :param ftarget: Target value of the objective function. If a feasible
        iterate achieves an objective function value lower or equal to
        `options['ftarget']`, the algorithm stops immediately.
        :type ftarget: float
        :param scale: Flag indicating whether to scale the problem according to
        the bound constraints.
        :type scale: bool
        :param quiet: Flag of quietness of the interface. If it is set to True,
        the output message will not be printed.
        :type quiet: bool
        :param classical: Flag indicating whether to call the classical Powell
        code or not.
        :type classical: bool
        :param debug: Debugging flag.
        :type debug: bool
        :param chkfunval: Flag used when debugging. If both `options['debug']`
        and `options['chkfunval']` are True, an extra function/constraint
        evaluation would be performed to check whether the returned values of
        objective function and constraint match the returned x.
        :type chkfunval: bool
        :param ensure_bounds:
        :type ensure_bounds: bool
        :param normalize_design_space: If True, normalize the design space
        :type normalize_design_space: bool
        """
        nds = normalize_design_space
        popts = self._process_options(
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            max_time=max_time,
            rhobeg=rhobeg,
            rhoend=rhoend,
            max_iter=max_iter,
            ftarget=ftarget,
            scale=scale,
            quiet=quiet,
            classical=classical,
            debug=debug,
            chkfunval=chkfunval,
            ensure_bounds=ensure_bounds,
            normalize_design_space=nds,
            **kwargs
        )
        return popts

    def _run(self, **options):
        """Runs the algorithm, to be overloaded by subclasses.

        :param options: the options dict for the algorithm
        """
        # remove normalization from options for algo
        normalize_ds = options.pop(self.NORMALIZE_DESIGN_SPACE_OPTION, True)

        # Get the normalized bounds:
        x_0, l_b, u_b = self.get_x0_and_bounds_vects(normalize_ds)

        # Ensure bounds
        ensure_bounds = options["ensure_bounds"]

        # Replace infinite values with None:
        l_b = [val if isfinite(val) else None for val in l_b]
        u_b = [val if isfinite(val) else None for val in u_b]
        bounds = list(zip(l_b, u_b))

        def real_part_fun(x_vect):
            """Wraps the function and returns the real part."""
            return real(self.problem.objective.func(x_vect))

        if ensure_bounds:
            fun = self.ensure_bounds(real_part_fun, normalize_ds)
        else:
            fun = real_part_fun

        constraints = self.get_right_sign_constraints()

        cstr_scipy = []
        for cstr in constraints:
            if PY2:
                f_type = cstr.f_type.encode("ascii")
            else:
                f_type = cstr.f_type
            if ensure_bounds:
                c_scipy = {
                    "type": f_type,
                    "fun": self.ensure_bounds(cstr.func, normalize_ds),
                }
            else:
                c_scipy = {"type": f_type, "fun": cstr.func}

            cstr_scipy.append(c_scipy)

        # |g| is in charge of ensuring max iterations, since it may
        # have a different definition of iterations, such as for SLSQP
        # for instance which counts duplicate calls to x as a new iteration
        max_iter = options[self.MAX_ITER]
        options["maxfev"] = int(max_iter * 1.2)

        opt_result = pdfo(
            fun=fun,
            x0=x_0,
            method=self.internal_algo_name,
            bounds=bounds,
            constraints=cstr_scipy,
            options=options,
        )

        return self.get_optimum_from_database(opt_result.message, opt_result.status)
