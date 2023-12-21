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
"""A wrapper for the global optimization algorithms of the SciPy library."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Union

from numpy import float64
from numpy import inf as np_inf
from numpy import int32
from numpy import isfinite
from numpy import real
from numpy.typing import NDArray
from scipy import optimize
from scipy.optimize import NonlinearConstraint

from gemseo import SEED
from gemseo.algos.opt.optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.opt.optimization_library import OptimizationLibrary

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.algos.opt_result import OptimizationResult
    from gemseo.core.mdofunctions.mdo_function import WrappedFunctionType
    from gemseo.core.mdofunctions.mdo_function import WrappedJacobianType

InputType = NDArray[Union[float64, int32]]


@dataclass
class SciPyGlobalAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of a global optimization algorithm from the SciPy library."""

    library_name: str = "SciPy"
    website: str = (
        "https://docs.scipy.org/doc/scipy/reference/optimize.html#global-optimization"
    )


class ScipyGlobalOpt(OptimizationLibrary):
    """A wrapper for the global optimization algorithms of the SciPy library."""

    LIB_COMPUTE_GRAD = True
    LIBRARY_NAME = "SciPy"

    def __init__(self) -> None:  # noqa:D107
        super().__init__()
        doc = "https://docs.scipy.org/doc/scipy/reference/generated/"
        self.descriptions = {
            "DUAL_ANNEALING": SciPyGlobalAlgorithmDescription(
                algorithm_name="Dual annealing",
                description="Dual annealing",
                handle_integer_variables=True,
                internal_algorithm_name="dual_annealing",
                website=f"{doc}scipy.optimize.dual_annealing.html",
            ),
            "SHGO": SciPyGlobalAlgorithmDescription(
                algorithm_name="SHGO",
                description="Simplicial homology global optimization",
                handle_equality_constraints=True,
                handle_inequality_constraints=True,
                handle_integer_variables=True,
                internal_algorithm_name="shgo",
                positive_constraints=True,
                website=f"{doc}scipy.optimize.shgo.html",
            ),
            "DIFFERENTIAL_EVOLUTION": SciPyGlobalAlgorithmDescription(
                algorithm_name="Differential evolution",
                description="Differential Evolution algorithm",
                handle_equality_constraints=True,
                handle_inequality_constraints=True,
                handle_integer_variables=True,
                internal_algorithm_name="differential_evolution",
                website=f"{doc}scipy.optimize.differential_evolution.html",
            ),
        }
        # maximum function calls option passed to the algorithm
        self.max_func_calls = sys.maxsize

    def _get_options(
        self,
        max_iter: int = 999,
        max_time: float = 0.0,
        ftol_rel: float = 1e-9,
        ftol_abs: float = 1e-9,
        xtol_rel: float = 1e-9,
        xtol_abs: float = 1e-9,
        stop_crit_n_x: int = 3,
        workers: int = 1,
        updating: str = "immediate",
        atol: float = 0.0,
        init: str = "latinhypercube",
        recombination: float = 0.7,
        tol: float = 0.01,
        popsize: int = 15,
        strategy: str = "best1bin",
        sampling_method: str = "simplicial",
        niters: int = 1,
        n: int = 100,
        seed: int = SEED,
        polish: bool = True,
        iters: int = 1,
        eq_tolerance: float = 1e-6,
        ineq_tolerance: float = 1e-6,
        normalize_design_space: bool = True,
        local_options: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:  # pylint: disable=W0221
        r"""Set the options default values.

        To get the best and up-to-date information about algorithms options,
        go to scipy.optimize documentation:
        https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html

        Args:
            max_iter: The maximum number of iterations, i.e. unique calls to f(x).
            max_time: The maximum runtime in seconds.
                The value 0 disables the cap on the runtime.
            ftol_rel: A stop criteria, the relative tolerance on the
               objective function.
               If abs(f(xk)-f(xk+1))/abs(f(xk))<= ftol_rel: stop.
            ftol_abs: A stop criteria, the absolute tolerance on the objective
               function. If abs(f(xk)-f(xk+1))<= ftol_rel: stop.
            xtol_rel: A stop criteria, the relative tolerance on the
               design variables. If norm(xk-xk+1)/norm(xk)<= xtol_rel: stop.
            xtol_abs: A stop criteria, the absolute tolerance on the
                   design variables. If norm(xk-xk+1)<= xtol_abs: stop.
            stop_crit_n_x: The number of design vectors to take into account in the
                stopping criteria.
            workers: The number of processes for parallel execution.
            updating: The strategy to update the solution vector.
                If ``"immediate"``, the best solution vector is continuously
                updated within a single generation.
                With ``"deferred"``,
                the best solution vector is updated once per generation.
                Only ``"deferred"`` is compatible with parallelization,
                and the ``"workers"`` keyword can over-ride this option.
            atol: The absolute tolerance for convergence.
            init: Either the type of population initialization to be used
                or an array specifying the initial population.
            recombination: The recombination constant, should be in the
                range [0, 1].
            tol: The relative tolerance for convergence.
            popsize: A multiplier for setting the total population size.
                The population has popsize * len(x) individuals.
            strategy: The differential evolution strategy to use.
            sampling_method: The method to compute the initial points.
                Current built in sampling method options
                are ``halton``, ``sobol`` and ``simplicial``.
            n: The number of sampling points
                used in the construction of the simplicial complex.
            seed: The seed to be used for repeatable minimizations.
                If ``None``, the ``numpy.random.RandomState`` singleton is used.
            polish: Whether to use the L-BFGS-B algorithm
                to polish the best population member at the end.
            mutation: The mutation constant.
            recombination: The recombination constant.
            initial_temp: The initial temperature.
            visit: The parameter for the visiting distribution.
            accept: The parameter for the acceptance distribution.
            niters: The number of iterations
                used in the construction of the simplicial complex.
            restart_temp_ratio: During the annealing process,
                temperature is decreasing,
                when it reaches ``initial_temp * restart_temp_ratio``,
                the reannealing process is triggered.
            normalize_design_space: If ``True``, variables are scaled in [0, 1].
            eq_tolerance: The tolerance on equality constraints.
            ineq_tolerance: The tolerance on inequality constraints.
            n: The number of sampling points used in the construction
                of the simplicial complex.
            iters: The number of iterations used in the construction
                of the simplicial complex.
            local_options: The options for the local optimization algorithm,
                only for shgo, see scipy.optimize doc.
            **kwargs: The other algorithms options.
        """
        return self._process_options(
            max_iter=max_iter,
            max_time=max_time,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            stop_crit_n_x=stop_crit_n_x,
            workers=workers,
            updating=updating,
            init=init,
            tol=tol,
            atol=atol,
            recombination=recombination,
            seed=seed,
            popsize=popsize,
            strategy=strategy,
            sampling_method=sampling_method,
            iters=iters,
            n=n,
            polish=polish,
            eq_tolerance=eq_tolerance,
            ineq_tolerance=ineq_tolerance,
            normalize_design_space=normalize_design_space,
            local_options=local_options,
            **kwargs,
        )

    def iter_callback(self, x_vect: InputType) -> None:
        """Call the objective and constraints functions.

        Args:
            x_vect: The input data with which to call the functions.
        """
        if self.normalize_ds:
            x_vect = self.problem.design_space.normalize_vect(x_vect)
        self.problem.objective(x_vect)
        for constraint in self.problem.constraints:
            constraint(x_vect)

    def real_part_obj_fun(self, x: InputType) -> int | float:
        """Wrap the function and return the real part.

        Args:
            x: The values to be given to the function.

        Returns:
            The real part of the evaluation of the objective function.
        """
        return real(self.problem.objective.func(x))

    def _run(self, **options: Any) -> OptimizationResult:
        """Run the algorithm, to be overloaded by subclasses.

        Args:
            **options: The options for the algorithm.

        Returns:
            The optimization result.
        """
        # remove normalization from options for algo
        self.normalize_ds = options.pop(self.NORMALIZE_DESIGN_SPACE_OPTION, True)
        # Get the normalized bounds:
        _, l_b, u_b = self.get_x0_and_bounds_vects(self.normalize_ds)
        # Replace infinite values with None:
        l_b = [val if isfinite(val) else None for val in l_b]
        u_b = [val if isfinite(val) else None for val in u_b]
        bounds = list(zip(l_b, u_b))
        # This is required because some algorithms do not
        # call the objective very often when the problem
        # is very constrained (Power2) and OptProblem may fail
        # to detect the optimum.

        if self.problem.has_constraints():
            self.problem.add_callback(self.iter_callback)

        if self.internal_algo_name == "dual_annealing":
            opt_result = optimize.dual_annealing(
                func=self.real_part_obj_fun,
                bounds=bounds,
                maxiter=self.max_func_calls,
                initial_temp=5230.0,
                restart_temp_ratio=2e-05,
                maxfun=self.max_func_calls,
                seed=options["seed"],
            )
        elif self.internal_algo_name == "shgo":
            opt_result = optimize.shgo(
                func=self.real_part_obj_fun,
                bounds=bounds,
                args=(),
                n=options["n"],
                iters=options["iters"],
                sampling_method=options["sampling_method"],
                constraints=self.__get_constraints_as_scipy_dictionary(),
                options=options.get("local_options"),
            )
        elif self.internal_algo_name == "differential_evolution":
            opt_result = optimize.differential_evolution(
                func=self.real_part_obj_fun,
                bounds=bounds,
                args=(),
                strategy=options["strategy"],
                maxiter=self.max_func_calls,
                popsize=options["popsize"],
                tol=options["tol"],
                atol=options["atol"],
                mutation=options.get("mutation", (0.5, 1)),
                recombination=options["recombination"],
                seed=options["seed"],
                polish=options["polish"],
                init=options["init"],
                updating=options["updating"],
                workers=options["workers"],
                constraints=self.__get_non_linear_constraints(),
            )
        else:  # pragma: no cover
            raise ValueError(f"Unknown algorithm: {self.internal_algo_name}.")

        return self.get_optimum_from_database(opt_result.message, opt_result.success)

    def __get_non_linear_constraints(self) -> tuple[NonlinearConstraint]:
        """Return the SciPy nonlinear constraints.

        Returns:
            The SciPy nonlinear constraints.
        """
        eq_tolerance = self.problem.eq_tolerance
        constraints = [
            NonlinearConstraint(constr, -eq_tolerance, eq_tolerance, jac=constr.jac)
            for constr in self.problem.get_eq_constraints()
        ]
        ineq_tolerance = self.problem.ineq_tolerance
        constraints.extend([
            NonlinearConstraint(constr, -np_inf, ineq_tolerance, jac=constr.jac)
            for constr in self.problem.get_ineq_constraints()
        ])
        return tuple(constraints)

    def __get_constraints_as_scipy_dictionary(
        self,
    ) -> list[dict[str, str | WrappedFunctionType | WrappedJacobianType]]:
        """Create the constraints to be passed to a SciPy algorithm as dictionaries.

        Returns:
            The constraints.
        """
        return [
            {"type": constraint.f_type, "fun": constraint.func, "jac": constraint.jac}
            for constraint in self.get_right_sign_constraints()
        ]
