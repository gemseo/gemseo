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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""scipy.optimize global optimization library wrapper."""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from builtins import super, zip

from numpy import isfinite, real
from scipy import optimize

from gemseo.algos.opt.opt_lib import OptimizationLibrary

LOGGER = logging.getLogger(__name__)


class ScipyGlobalOpt(OptimizationLibrary):
    """Scipy optimization library interface.

    See OptimizationLibrary.
    """

    LIB_COMPUTE_GRAD = True

    def __init__(self):
        """Constructor.

        Generate the library dict, contains the list
        of algorithms with their characteristics:

        - does it require gradient
        - does it handle equality constraints
        - does it handle inequality constraints
        """
        super(ScipyGlobalOpt, self).__init__()
        doc = "https://docs.scipy.org/doc/scipy/reference/generated/"
        self.lib_dict = {
            "DUAL_ANNEALING": {
                self.INTERNAL_NAME: "dual_annealing",
                self.REQUIRE_GRAD: False,
                self.POSITIVE_CONSTRAINTS: False,
                self.HANDLE_EQ_CONS: False,
                self.HANDLE_INEQ_CONS: False,
                self.DESCRIPTION: "Dual annealing",
                self.WEBSITE: doc + "scipy.optimize.dual_annealing.html",
            },
            "SHGO": {
                self.INTERNAL_NAME: "shgo",
                self.REQUIRE_GRAD: False,
                self.POSITIVE_CONSTRAINTS: False,
                self.HANDLE_EQ_CONS: False,
                self.HANDLE_INEQ_CONS: False,
                self.DESCRIPTION: "Simplicial homology global optimization",
                self.WEBSITE: doc + "scipy.optimize.shgo.html",
            },
            "DIFFERENTIAL_EVOLUTION": {
                self.INTERNAL_NAME: "differential_evolution",
                self.REQUIRE_GRAD: False,
                self.POSITIVE_CONSTRAINTS: False,
                self.HANDLE_EQ_CONS: False,
                self.HANDLE_INEQ_CONS: False,
                self.DESCRIPTION: "Differential Evolution algorithm",
                self.WEBSITE: doc + "scipy.optimize.differential_evolution.html",
            },
        }

    def _get_options(
        self,
        max_iter=999,
        ftol_rel=1e-9,
        ftol_abs=1e-9,
        xtol_rel=1e-9,
        xtol_abs=1e-9,
        workers=1,
        updating="immediate",
        atol=0,
        init="latinhypercube",
        recombination=0.7,
        tol=0.01,
        popsize=15,
        strategy="best1bin",
        sampling_method="simplicial",
        niters=1,
        n=100,
        seed=1,
        polish=True,
        **kwargs
    ):  # pylint: disable=W0221
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
        :param kwargs: other algorithms options
        :type kwargs: kwargs
        """
        popts = self._process_options(
            max_iter=max_iter,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            workers=workers,
            updating=updating,
            atol=atol,
            init=init,
            recombination=recombination,
            seed=seed,
            tol=tol,
            popsize=popsize,
            strategy=strategy,
            sampling_method=sampling_method,
            niters=niters,
            n=n,
            polish=polish,
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
        _, l_b, u_b = self.get_x0_and_bounds_vects(normalize_ds)
        # Replace infinite values with None:
        l_b = [val if isfinite(val) else None for val in l_b]
        u_b = [val if isfinite(val) else None for val in u_b]
        bounds = list(zip(l_b, u_b))

        def real_part_fun(x_vect):
            """Wraps the function and returns the real part."""
            return real(self.problem.objective.func(x_vect))

        fun = real_part_fun

        if self.internal_algo_name == "dual_annealing":
            opt_result = optimize.dual_annealing(
                func=fun,
                bounds=bounds,
                maxiter=10000000,
                local_search_options={},
                initial_temp=5230.0,
                restart_temp_ratio=2e-05,
                visit=2.62,
                accept=-5.0,
                maxfun=10000000.0,
                seed=1,
                no_local_search=False,
                callback=None,
                x0=None,
            )
        elif self.internal_algo_name == "shgo":
            opt_result = optimize.shgo(
                func=fun,
                bounds=bounds,
                args=(),
                constraints=None,
                n=options["n"],
                iters=options["iters"],
                callback=None,
                minimizer_kwargs=None,
                options=None,
                sampling_method=options["sampling_method"],
            )
        elif self.internal_algo_name == "differential_evolution":
            opt_result = optimize.differential_evolution(
                func=fun,
                bounds=bounds,
                args=(),
                strategy=options["strategy"],
                maxiter=10000000,
                popsize=options["popsize"],
                tol=options["tol"],
                mutation=options.get("mutation", (0.5, 1)),
                recombination=options["recombination"],
                seed=options["seed"],
                polish=options["polish"],
                init=options["init"],
                atol=options["atol"],
                updating=options["updating"],
                workers=options["workers"],
            )
        return self.get_optimum_from_database(opt_result.message, opt_result.success)
