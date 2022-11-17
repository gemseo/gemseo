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
"""PDFO optimization library wrapper, see `PDFO website <https://www.pdfo.net/>`_."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from typing import Optional
from typing import Union

from numpy import inf
from numpy import isfinite
from numpy import ndarray
from numpy import real

from gemseo.algos.opt.opt_lib import OptimizationAlgorithmDescription
from gemseo.algos.opt.opt_lib import OptimizationLibrary
from gemseo.algos.opt_result import OptimizationResult

# workaround to prevent dll error with xlwings from pypi with anaconda python:
# backup the state of the environment variable CONDA_DLL_SEARCH_MODIFICATION_ENABLE
# which could be modified by pdfo
conda_dll_search_modification_enable = os.environ.get(
    "CONDA_DLL_SEARCH_MODIFICATION_ENABLE"
)

from pdfo import pdfo  # noqa: E402

# workaround to prevent dll error with xlwings from pypi with anaconda python:
# restore the state of the environment variable CONDA_DLL_SEARCH_MODIFICATION_ENABLE
if conda_dll_search_modification_enable is None:
    os.environ.pop("CONDA_DLL_SEARCH_MODIFICATION_ENABLE", None)
else:
    os.environ[
        "CONDA_DLL_SEARCH_MODIFICATION_ENABLE"
    ] = conda_dll_search_modification_enable

OptionType = Optional[Union[str, int, float, bool, ndarray]]


@dataclass
class PDFOAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of an optimization algorithm from the PDFO library."""

    library_name: str = "PDFO"
    website: str = "https://www.pdfo.net/"


class PDFOOpt(OptimizationLibrary):
    """PDFO optimization library interface.

    See OptimizationLibrary.
    """

    LIB_COMPUTE_GRAD = False

    OPTIONS_MAP = {
        OptimizationLibrary.MAX_ITER: "max_iter",
    }

    LIBRARY_NAME = "PDFO"

    def __init__(self):
        """Constructor.

        Generate the library dict, contains the list
        of algorithms with their characteristics:

        - does it require gradient
        - does it handle equality constraints
        - does it handle inequality constraints
        """
        super().__init__()
        self.descriptions = {
            "PDFO_COBYLA": PDFOAlgorithmDescription(
                algorithm_name="COBYLA",
                description="Constrained Optimization By Linear Approximations ",
                handle_equality_constraints=True,
                handle_inequality_constraints=True,
                internal_algorithm_name="cobyla",
                positive_constraints=True,
            ),
            "PDFO_BOBYQA": PDFOAlgorithmDescription(
                algorithm_name="BOBYQA",
                description="Bound Optimization By Quadratic Approximation",
                internal_algorithm_name="bobyqa",
            ),
            "PDFO_NEWUOA": PDFOAlgorithmDescription(
                algorithm_name="NEWUOA",
                description="NEWUOA",
                internal_algorithm_name="newuoa",
            ),
        }
        self.name = "PDFO"

    def _get_options(
        self,
        ftol_rel: float = 1e-12,
        ftol_abs: float = 1e-12,
        xtol_rel: float = 1e-12,
        xtol_abs: float = 1e-12,
        max_time: float = 0,
        rhobeg: float = 0.5,
        rhoend: float = 1e-6,
        max_iter: int = 500,
        ftarget: float = -inf,
        scale: bool = False,
        quiet: bool = True,
        classical: bool = False,
        debug: bool = False,
        chkfunval: bool = False,
        ensure_bounds: bool = True,
        normalize_design_space: bool = True,
        **kwargs: OptionType,
    ) -> dict[str, Any]:
        r"""Set the options default values.

        To get the best and up-to-date information about algorithms options,
        go to pdfo documentation on the `PDFO website <https://www.pdfo.net/>`_.

        Args:
            ftol_rel: A stop criteria, relative tolerance on the
               objective function,
               if abs(f(xk)-f(xk+1))/abs(f(xk))<= ftol_rel: stop.
            ftol_abs: A stop criteria, absolute tolerance on the objective
               function, if abs(f(xk)-f(xk+1))<= ftol_rel: stop.
            xtol_rel: A stop criteria, relative tolerance on the
               design variables,
               if norm(xk-xk+1)/norm(xk)<= xtol_rel: stop.
            xtol_abs: A stop criteria, absolute tolerance on the
               design variables,
               if norm(xk-xk+1)<= xtol_abs: stop.
            max_time: The maximum runtime in seconds,
                disabled if 0.
            rhobeg: The initial value of the trust region radius.
            max_iter: The maximum number of iterations.
            rhoend: The final value of the trust region radius. Indicates
                the accuracy required in the final values of the variables.
            maxfev:  The upper bound of the number of calls of the objective function `fun`.
            ftarget: The target value of the objective function. If a feasible
                iterate achieves an objective function value lower or equal to
                `options['ftarget']`, the algorithm stops immediately.
            scale: The flag indicating whether to scale the problem according to
                the bound constraints.
            quiet: The flag of quietness of the interface. If True,
                the output message will not be printed.
            classical: The flag indicating whether to call the classical Powell code or not.
            debug: The debugging flag.
            chkfunval: A flag used when debugging. If both `options['debug']`
                and `options['chkfunval']` are True, an extra function/constraint
                evaluation would be performed to check whether the returned values of
                the objective function and constraint match the returned x.
            ensure_bounds: Whether to project the design vector
                onto the design space before execution.
            normalize_design_space: If True, normalize the design space.
            **kwargs: The other algorithm's options.
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
            **kwargs,
        )
        return popts

    def _run(self, **options: OptionType) -> OptimizationResult:
        """Run the algorithm, to be overloaded by subclasses.

        Args:
            **options: The options of the algorithm.

        Returns:
            The optimization result.
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

        def real_part_fun(
            x: ndarray,
        ) -> int | float:
            """Wrap the objective function and keep the real part.

            Args:
                x: The values to be given to the function.

            Returns:
                The real part of the evaluation of the function.
            """
            return real(self.problem.objective.func(x))

        if ensure_bounds:
            fun = self.ensure_bounds(real_part_fun, normalize_ds)
        else:
            fun = real_part_fun

        constraints = self.get_right_sign_constraints()

        cstr_scipy = []
        for cstr in constraints:
            c_scipy = {"type": cstr.f_type}
            if ensure_bounds:
                c_scipy["fun"] = self.ensure_bounds(cstr.func, normalize_ds)
            else:
                c_scipy["fun"] = cstr.func

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
