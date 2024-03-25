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
#
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial documentation
#        :author: Vincent DROUET
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Modified Normal Boundary Intersection (mNBI) algorithm.

Based on :cite:`shukla2007normal`.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from multiprocessing import Manager
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final
from typing import NamedTuple
from typing import Union

from numpy import append
from numpy import argwhere
from numpy import array
from numpy import block
from numpy import column_stack
from numpy import concatenate
from numpy import dot
from numpy import eye
from numpy import hstack
from numpy import linspace
from numpy import newaxis
from numpy import ones
from numpy import vstack
from numpy import zeros
from numpy import zeros_like
from numpy.linalg import LinAlgError
from numpy.linalg import solve
from scipy.optimize import linprog

from gemseo.algos.database import Database
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.algos.opt._mnbi.constraint_function_wrapper import ConstraintFunctionWrapper
from gemseo.algos.opt._mnbi.function_component_extractor import (
    FunctionComponentExtractor,
)
from gemseo.algos.opt._mnbi.sub_optim_constraint import SubOptimConstraint
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt.optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.opt.optimization_library import OptimizationLibrary
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result_multiobj import MultiObjectiveOptimizationResult
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.mdofunctions.mdo_function import NotImplementedCallable
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.multiprocessing.execution import execute

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from gemseo.algos.doe.doe_library import DOELibraryOptionType
    from gemseo.algos.driver_library import DriverLibOptionType
    from gemseo.algos.opt_problem import OptimizationResult
    from gemseo.typing import RealArray

LOGGER = logging.getLogger(__name__)


MNBIOptionsType = Union[bool, int, float]


class IndividualSubOptimOutput(NamedTuple):
    """An output from a sub optimization."""

    f_min: RealArray
    """The value of f at the design value minimizing f_i."""

    x_min: RealArray
    """The value of the design variables minimizing f_i."""

    database: Database
    """The database of the main problem."""

    n_calls: int
    """The number of calls to f."""


class BetaSubOptimOutput(NamedTuple):
    """An output from a beta sub optimization."""

    f_min: RealArray
    """The coordinates in the objective space of the sub-optimization result."""

    x_min: RealArray
    """The coordinates in the design space of the sub-optimization result."""

    w: RealArray
    """The vector w used to compute the values of beta that can be skipped in the
    following sub-optimizations."""

    database: Database
    """The database of the main problem."""

    n_calls: int
    """The number of calls to the main objective function by the optimizer during the
    sub-optimization."""


@dataclass
class MNBIAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of the MNBI optimization algorithm."""

    handle_equality_constraints: bool = True

    handle_inequality_constraints: bool = True

    handle_integer_variables: bool = True

    handle_multiobjective: bool = True

    library_name: str = "MNBI"


class MNBI(OptimizationLibrary):
    r"""MNBI optimizer.

    This algorithm computes the Pareto front of a multi-objective optimization problem
    by decomposing it into a series of constrained single-objective problems.
    Considering the following problem:

    .. math::

        \begin{align}
        & \min_{x \in D} f(x),\\
        & g(x) \leq 0,\\
        & h(x) = 0
        \end{align}

    the algorithm first finds the individual optima :math:`(x_i^\ast)_{i=1..m}`
    of the :math:`m` components of the objective function :math:`f`.
    The corresponding anchor points :math:`(f(x_i^\ast))_{i=1..m}`
    are stored in a matrix :math:`\Phi`.

    The simplex formed by the convex hull of the anchor points
    can be expressed as :math:`\Phi \beta`,
    where :math:`\beta = \{ (b_1, ..., b_m)^T | \sum_{i=1}^m b_i =1 \}`.

    Given a list of vectors :math:`\beta`,
    mNBI will solve the following single-objective problems:

    .. math::
        \begin{align}
        & \max_{x \in D, t \in \mathbb{R}} t,\\
        & \Phi \beta + t \hat{n} \geq f(x),\\
        & g(x) \leq 0,\\
        & h(x) = 0
        \end{align}

    where :math:`\hat{n}` is a quasi-normal vector to the :math:`\Phi \beta` simplex
    pointing towards the origin of the objective space.
    If :math:`(x^{*}, t^{*})` is a solution of this problem,
    :math:`x^{*}` is proven to be at least weakly Pareto-dominant.

    Let :math:`w = \Phi \beta + t^{*} \hat{n}`,
    and :math:`\pi` denote the projection (in the direction of :math:`\hat{n}`)
    on the simplex formed by the convex hull of the anchor points.
    If not all constraints :math:`\Phi \beta + t^{*} \hat{n} \geq f(x^{*})` are active,
    :math:`x^{*}` will weakly dominate the solution of the sub-problem
    for all values :math:`\beta_{dom}` that verify:

    .. math::
        \Phi \beta_{dom} \in \pi[(f(x^{*}) + \mathbb{R}_m^{+})
        \cap (w - \mathbb{R}_m^{+})]

    Therefore, the corresponding sub-optimizations are redundant
    and can be skipped to reduce run time.
    """

    _debug_results: Database = Database()
    """The results of the sub-optimizations in debug mode."""

    _RESULT_CLASS: ClassVar[type[OptimizationResult]] = MultiObjectiveOptimizationResult
    """The class used to present the result of the optimization."""

    __beta_sub_optim: OptimizationProblem | None = None
    """The sub-optimization problem that must be run for each value of beta."""

    __debug: bool = False
    """Whether the algorithm is running in debug mode."""

    __debug_file_path: str = "debug_Pareto.h5"
    """The output file for the Pareto front in debug mode.

    mNBI algorithm normally returns one point of the Pareto front per sub-optimization.
    The ``get_optimum_from_database`` method returns all Pareto optimal points in the
    database. In debug mode, the optima actually returned by mNBI are saved to an hdf5
    file, in order to verify that the algorithm has worked as intended.
    """

    __ineq_tolerance: float = 1e-4
    """The tolerance on the inequality constraints."""

    __sub_optim_max_iter: int
    """The maximum number of iterations for each run of the sub-optimization problem."""

    __n_obj: int
    """The number of objectives of the optimization problem."""

    __n_processes: int
    """The number of processes to use when running the sub-optimizations."""

    __n_sub_optim: int
    """The number of runs of the sub-optimization problem."""

    __n_vect: RealArray
    """The quasi-normal vector to the phi simplex."""

    __phi: RealArray
    r"""The matrix :math:`\Phi=(f(x^{(1),*}) | \ldots | f(x^{(d),*}))`.

    Where :math:`f=(f_1,\ldots,f_d)` is the multi-objective function
    and :math:`x^{(i),*}` the design point minimizing :math:`f_i`.

    It has the shape ``(n_obj, n_obj)``.
    """

    __skip_betas: bool
    """Whether to skip the sub-optimizations corresponding to values of beta for which
    the theoretical result has already been found.

    This can accelerate the main optimization by avoiding redundant sub-optimizations.
    But in cases where some sub-optimizations do not properly converge, some values of
    betas will be skipped based on false assumptions, and some parts of the Pareto front
    can be incorrectly resolved.
    """

    __skippable_domains: list[RealArray]
    """The regions of the phi simplex that can be skipped."""

    __sub_optim_algo: str
    """The algorithm used for the sub-optimizations."""

    __sub_optim_algo_options: Mapping[str, DriverLibOptionType]
    """The options for the sub-optimization algorithm."""

    __utopia: RealArray
    r"""The utopia :math:`(f_1(x^{(1),*}), \ldots, f_d(x^{(d),*}))`.

    Where :math:`x^{(i),*}` the design point minimizing :math:`f_i`.

    It has the shape ``(n_obj, 1)`` and it is the diagonal of :attr:`.__phi`.
    """

    __SUB_OPTIM_CONSTRAINT_NAME: Final[str] = "beta_sub_optim_constraint"

    def __init__(self) -> None:  # noqa:D107
        super().__init__()
        self.descriptions = {
            "MNBI": MNBIAlgorithmDescription(
                algorithm_name="mNBI",
                internal_algorithm_name="mNBI",
                description="Modified Normal Boundary Intersection (mNBI) method",
            )
        }

    def _get_options(
        self,
        max_iter: int,
        sub_optim_algo: str,
        normalize_design_space: bool = False,
        n_sub_optim: int = 1,
        sub_optim_algo_options: Mapping[
            str, DriverLibOptionType
        ] = READ_ONLY_EMPTY_DICT,
        sub_optim_max_iter: int | None = None,
        doe_algo: str = "fullfact",
        doe_algo_options: Mapping[str, DOELibraryOptionType] = READ_ONLY_EMPTY_DICT,
        n_processes: int = 1,
        debug_file_path: str | Path = "debug_history.h5",
        skip_betas: bool = True,
        debug: bool = False,
    ) -> dict[str, MNBIOptionsType]:
        r"""Set the options default values.

        Args:
            max_iter: The maximum number of iterations.
            sub_optim_algo: The optimization algorithm used to solve
                the generated sub optimization problems.
            sub_optim_algo_options: The options for the optimization
                algorithm.
            n_sub_optim: The number of sub optimization in addition to
                the individual minimums of the objectives. mNBI will
                generate n_sub_optim points on the Pareto front
                between the "number of objective" individual minimas. This value must be
                strictly greater than the number of objectives of the problem.
            normalize_design_space: Whether to normalize the design
                variables between 0 and 1.
            sub_optim_max_iter: Maximum number of iterations of the
                sub optimization algorithms. If ``None``, the ``max_iter`` value is
                used.
            doe_algo: The design of experiments algorithm for the
                target points on the Pareto front in problems with more than 2
                objectives. A ``fullfactorial`` DOE is used default as these tend to be
                low dimensions, usually not more than 3 objectives for a given problem.
            doe_algo_options: The options for the DOE algorithm.
            n_processes: The maximum simultaneous number of processes
                used to parallelize the sub optimizations.
            debug: Whether to output a database hdf file containing the
                sub optimization optimas only.
            debug_file_path: The path to the debug file if debug mode is
                active.

        Returns:
            The mNBI library options with their values.
        """
        sub_optim_algo_options = (
            sub_optim_algo_options if sub_optim_algo_options else {}
        )
        sub_optim_max_iter = sub_optim_max_iter if sub_optim_max_iter else max_iter
        doe_algo_options = doe_algo_options if doe_algo_options else {}
        return self._process_options(
            max_iter=max_iter,
            sub_optim_algo=sub_optim_algo,
            sub_optim_max_iter=sub_optim_max_iter,
            normalize_design_space=normalize_design_space,
            n_sub_optim=n_sub_optim,
            sub_optim_algo_options=sub_optim_algo_options,
            doe_algo=doe_algo,
            doe_algo_options=doe_algo_options,
            n_processes=n_processes,
            skip_betas=skip_betas,
            debug=debug,
            debug_file_path=debug_file_path,
        )

    def __minimize_objective_components_separately(self) -> None:
        """Minimize the component of the multi-objective function separately.

        The results are used to create the __phi and __utopia arrays.
        """
        LOGGER.info("Searching for the individual optimum of each objective")
        optima = execute(
            self._minimize_objective_component,
            [self.__copy_database_save_minimum],
            self.__n_processes,
            list(range(self.__n_obj)),
        )
        self.__phi = column_stack([optimum[0] for optimum in optima])
        self.__utopia = self.__phi.diagonal()[:, newaxis]

    def _minimize_objective_component(self, i: int) -> IndividualSubOptimOutput:
        """Minimize the i-th component f_i of the objective function f=(f1,...,f_d).

        Args:
            i: The index of the component of the objective function.

        Returns:
            The value of f at the design value minimizing f_i.
            The value of the design variables minimizing f_i.
            The database of the main problem. This is returned so that it can be copied
            in the database of the main process, to store the evaluations done in each
            sub-process.
            The number of calls to f.

        Raises:
            RuntimeError: If no optimum is found for one of the objectives.
        """
        n_calls_start = self.problem.objective.n_calls
        design_space = DesignSpace()
        design_space.extend(self.problem.design_space)
        opt_problem = OptimizationProblem(design_space)
        opt_problem.constraints = list(self.problem.constraints)
        objective = FunctionComponentExtractor(self.problem.objective, i)
        jac = (
            None
            if self.problem.objective.jac is NotImplementedCallable
            else objective.compute_jacobian
        )
        opt_problem.objective = MDOFunction(objective.compute_output, f"f_{i}", jac=jac)
        opt_result = OptimizersFactory().execute(
            opt_problem,
            self.__sub_optim_algo,
            max_iter=self.__sub_optim_max_iter,
            normalize_design_space=self._normalize_design_space,
            activate_progress_bar=False,
            **self.__sub_optim_algo_options,
        )
        if not opt_result.is_feasible:
            msg = f"No feasible optimum found for the {i}-th objective function."
            raise RuntimeError(msg)

        x_min = opt_result.x_opt
        f_min = self.problem.objective(x_min)
        n_calls = self.problem.objective.n_calls - n_calls_start
        return IndividualSubOptimOutput(f_min, x_min, self.problem.database, n_calls)

    def __copy_database_save_minimum(
        self, index: int, outputs: IndividualSubOptimOutput
    ) -> None:
        """Update the database of the optimization problem with that of the sub-problem.

        The sub-problem aims to minimize a component f_i of f=(f_1,...,f_d).

        In debug mode,
        this function stores the objective vector f(x*) and the design value x*
        in a debug history file,
        where x* minimizes f_i(x*).

        Args:
            index: The index of the worker used to run the sub-optimization.
                Provided by |g| but unused here.
            outputs: The outputs of the sub-optimization
                returned by ``_minimize_objective_component``.
        """
        if self.__n_processes > 1:
            # Store the sub-process database in the main database
            objective = self.problem.objective
            objective.n_calls += outputs.n_calls
            for functions in [
                [objective],
                self.problem.constraints,
                self.problem.observables,
            ]:
                for f in functions:
                    f_hist, x_hist = outputs.database.get_function_history(
                        f.name, with_x_vect=True
                    )
                    for x_value, f_value in zip(x_hist, f_hist):
                        self.problem.database.store(x_value, {f.name: f_value})

        if self.__debug:
            self._debug_results.store(outputs.x_min, {"obj": outputs.f_min})

    @staticmethod
    def _t_extraction_func(x_t: RealArray) -> RealArray:
        """Return the value of ``t`` from a vector concatenating ``x`` and ``t``.

        Args:
            x_t: A vector concatenating ``x`` and ``t``.

        Returns:
            The value of ``t``.
        """
        return x_t[-1]

    @staticmethod
    def _t_extraction_jac(x_t: RealArray) -> RealArray:
        """Compute the Jacobian of `_t_extraction_func`.

        Args:
            x_t: A vector concatenating ``x`` and ``t``.

        Returns:
            The Jacobian of ``_t_extraction_func`` at ``x_t``.
        """
        jac = zeros_like(x_t)[newaxis]
        jac[0][-1] = 1
        return jac

    def __create_beta_sub_optim(self) -> None:
        """Create the beta sub-optimization problem.

        The goal is to maximize ``t``
        w.r.t. the design variables ``x`` and the real variable ``t``.
        """
        design_space = deepcopy(self.problem.design_space)
        design_space.add_variable("t", value=0.0)
        self.__beta_sub_optim = OptimizationProblem(design_space)
        self.__beta_sub_optim.objective = MDOFunction(
            self._t_extraction_func,
            name="t_extraction",
            jac=self._t_extraction_jac,
        )
        self.__beta_sub_optim.change_objective_sign()

    def _run_beta_sub_optim(self, beta: RealArray) -> BetaSubOptimOutput | tuple[()]:
        """Run the sub-optimization problem for a given value of beta.

        If the main problem has two objectives, the sub-optimization starts from the
        previous sub-problem result to accelerate convergence, since the optima of two
        consecutive sub-problems are probably close to each other.
        Otherwise, nothing guaranties that the two optima are close to each other since
        the betas are spread randomly, therefore the sub-optimization starts from the
        initial point.

        Args:
            beta: The coordinates of the considered point in the phi simplex.

        Returns:
            The coordinates in the objective space of the sub-optimization result.
            The coordinates in the design space of the sub-optimization result.
            The vector w used to compute the values of beta that can be skipped in the
                following sub-optimizations. This is computed only if one component of
                the beta sub-constraint is inactive. Otherwise, returns None.
            The database of the main problem. This is returned so that it can be copied
                in the database of the main process, to store the evaluations done in
                each sub-process.
            The number of calls to the main objective function by the optimizer during
                the sub-optimization.
            Or an empty tuple if the resulting solution of the sub-optimization is
                for the given value of beta is already known.
        """
        f = self.problem.objective
        n_calls_start = f.n_calls
        beta = append(beta, 1 - beta.sum())
        phi_beta = dot(self.__phi, beta)

        # Check if phi_beta is in the skippable domains.
        if self.__skip_betas and self.__is_skippable(phi_beta):
            LOGGER.info(
                "Skipping beta = %s because the resulting solution is already known.",
                beta,
            )
            return ()

        LOGGER.info("Solving mNBI sub-problem for beta = %s", beta)
        beta_sub_optim_constraint = SubOptimConstraint(phi_beta, self.__n_vect, f)
        jac = (
            beta_sub_optim_constraint.compute_jacobian
            if not isinstance(f.jac, NotImplementedCallable)
            else None
        )
        beta_sub_cstr = MDOFunction(
            beta_sub_optim_constraint.compute_output,
            name=self.__SUB_OPTIM_CONSTRAINT_NAME,
            jac=jac,
        )

        # Reset the design space if there are more than two objective, since two
        # successive values of beta are not necessarily close
        self.__beta_sub_optim.reset(design_space=self.__n_obj != 2)
        wrapped_constraints = []
        for g in self.problem.constraints:
            wrapped_g = ConstraintFunctionWrapper(g)
            jac = (
                wrapped_g.compute_jacobian
                if not isinstance(g.jac, NotImplementedCallable)
                else None
            )
            wrapped_constraints.append(
                MDOFunction(
                    wrapped_g.compute_output,
                    name=f"wrapped_{g.name}",
                    jac=jac,
                    f_type=g.f_type,
                )
            )
        self.__beta_sub_optim.constraints = wrapped_constraints
        self.__beta_sub_optim.add_constraint(beta_sub_cstr, cstr_type="ineq")
        opt_res = OptimizersFactory().execute(
            self.__beta_sub_optim,
            self.__sub_optim_algo,
            max_iter=self.__sub_optim_max_iter,
            normalize_design_space=self._normalize_design_space,
            activate_progress_bar=False,
            **self.__sub_optim_algo_options,
        )
        if not opt_res.is_feasible:
            LOGGER.warning("No feasible optimum has been found for beta=%s", beta)
        x_min = opt_res.x_opt[:-1]
        f_min = f(x_min)
        n_calls = f.n_calls - n_calls_start

        # If some components of the sub-optim constraint are inactive, return the vector
        # w to find the values of beta that can be skipped for the next sub-optims
        if self.__skip_betas and any(
            beta_sub_optim_constraint.compute_output(opt_res.x_opt)
            < self.__ineq_tolerance
        ):
            w = phi_beta + opt_res.x_opt[-1] * self.__n_vect
        else:
            w = None

        return BetaSubOptimOutput(f_min, x_min, w, self.problem.database, n_calls)

    def __beta_sub_optim_callback(
        self, index: int, outputs: BetaSubOptimOutput
    ) -> None:
        """The callback function called after running a beta sub-optimization.

        This function main goal is to copy the sub-problem database into the main
        problem's one when running in parallel.
        If one component of the beta constraint is inactive, this function will also
        find the values of beta that can be skipped in the following sub-optimizations.
        In debug mode, this function stores the result of the sub-optimization in a
        debug history file.

        Args:
            index: The index of the worker used to run the sub-optimization. Provided by
                GEMSEO but unused here.
            outputs: The outputs of the sub-optimization.
        """
        if not outputs:
            return

        database = outputs.database

        if self.__n_processes > 1 and database is not None:
            # Store the sub-process database in the main process database.
            self.problem.objective.n_calls += outputs.n_calls
            f_hist, x_hist = database.get_function_history(
                self.problem.objective.name, with_x_vect=True
            )
            for xi, fi in zip(x_hist, f_hist):
                self.problem.database.store(xi, {self.problem.objective.name: fi})

            for functions in [self.problem.constraints, self.problem.observables]:
                for f in functions:
                    f_hist, x_hist = database.get_function_history(
                        f.name, with_x_vect=True
                    )
                    for xi, fi in zip(x_hist, f_hist):
                        self.problem.database.store(xi, {f.name: fi})

        f_min = outputs.f_min

        if outputs.w is not None:
            self.__find_skippable_domain(f_min, outputs.w)

        if self.__debug and f_min is not None:
            self._debug_results.store(outputs.x_min, {"obj": f_min})

    def __find_skippable_domain(self, f: RealArray, w: RealArray) -> None:
        """Find the domain in the phi simplex that can be skipped based on f and w.

        Project the hypervolume of the solution space formed by the intersection of the
        two subspaces y >= f and y <= w on the phi simplex.

        Args:
            f: The values of the objective functions at the current optimum, and first
                extremity of the hypervolume to be projected.
            w: The other extremity of the hypervolume.
        """
        fw = column_stack((f, w))
        # Find all the corners of the hypervolume
        proj_points = []
        inds = argwhere(abs(f - w) > self.__ineq_tolerance)
        for c in combinations([0, 1], len(inds)):
            y = zeros((f.size, 1))
            for i, j in enumerate(inds):
                y[j, 0] = fw[j, c[i]]
                proj_y = self.__project_on_phi_simplex(y)
                if proj_y is not None:
                    proj_points.append(proj_y)

        if proj_points:
            self.__skippable_domains.append(hstack(proj_points))

    def __project_on_phi_simplex(self, y: RealArray) -> RealArray | None:
        """Project a point on the phi simplex.

        Solves the linear system |phi   -I_m    0| |alpha|   |0|
                                 | 0     I_m   -n| |  z  | = |y|
                                 |1_m     0     0| |  k  |   |1|
        where:
            m is the number of objective functions
            I_m is the identity matrix of size m
            1_m is a matrix of ones of shape (1, m)
            n is the quasi-normal vector to the phi simplex
            z is the coordinates of the projection of y on the phi simplex
            alpha is the coordinates of z in the phi simplex
            k is the coordinates of z on the y + n line

        Args:
            y: The point to project.

        Returns:
            The coordinates in the objective space of the projection of y on the phi
            simplex. ``None`` when the projection on the phi simplex failed.
        """
        m = self.__n_obj
        eye_m = eye(m)
        zeros_m1 = zeros((m, 1))
        lhs = block([
            [self.__phi, -eye_m, zeros_m1],
            [zeros((m, m)), eye_m, -self.__n_vect.reshape(m, 1)],
            [ones((1, m)), zeros((1, m)), 0],
        ])
        rhs = vstack((zeros_m1, y, array([[1]])))
        try:
            z = solve(lhs, rhs)[m : 2 * m, [0]]
        except LinAlgError:
            LOGGER.warning(
                "Could not solve the projection system, projection on the phi simplex "
                "failed."
            )
            return None

        return z

    def __is_skippable(self, phi_beta: RealArray) -> bool:
        """Check whether the point phi_beta is in a skippable domain of the phi simplex.

        Args:
            phi_beta: The coordinates of the point of the phi simplex.

        Returns:
            Whether the point lies in a skippable domain.
        """
        for domain in self.__skippable_domains:
            if self.__is_in_convex_hull(phi_beta, domain):
                return True
        return False

    @staticmethod
    def __is_in_convex_hull(y: RealArray, z: RealArray) -> bool:
        """Check whether the point y is in the convex hull of the points z.

        y lies in the convex hull of z if it can be expressed as a convex combination of
        z, i.e. if here exists a positive vector w solution of | z | | |   |y|
                                                               |   | |w| = | |
                                                               |1_n| | |   |1|
        where 1_n is a matrix of ones of shape (1, n) and n is the number of points in
        z.

        The linprog call is used to check if the constraint A_eq w = b_eq is satisfied
        (the objective is always zero). The positivity of w's components is enforced by
        default in linprog.

        Args:
            y: The coordinates of the point to be checked, of shape (dim, 1).
            z: The matrix of coordinates of the points z, of shape (dim, n).

        Returns:
            Whether y lies in the convex hull of the points of z.
        """
        n_pts = z.shape[1]
        a = concatenate([z, ones((1, n_pts))], axis=0)
        b = concatenate([y, array([1])], axis=0).flatten()
        c = zeros(n_pts)
        lp = linprog(c, A_eq=a, b_eq=b, method="highs")
        return lp.success

    def _pre_run(
        self, problem: OptimizationProblem, algo_name: str, **options: Any
    ) -> None:
        """Processes the options and setups the optimization.

        Raises:
            ValueError:
                - If the algorithm is being used to solve a mono-objective problem.
                - If the given `n_sub_optim` is not strictly greater than the number of
                  objectives.
                - If the name of one of the constraints of the problem coincides with
                  the protected name for the sub optimization problems used by mNBI.
        """
        super()._pre_run(problem, algo_name, **options)
        self.__n_obj = self.problem.objective.dim
        self.__n_sub_optim = options.pop("n_sub_optim")
        if self.__n_obj == 1:
            msg = "MNBI optimizer is not suitable for mono-objective problems."
            raise ValueError(msg)

        if self.__n_sub_optim <= self.__n_obj:
            msg = (
                "The number of sub-optimization problems must be "
                f"strictly greater than the number of objectives {self.__n_obj}; "
                f"got {self.__n_sub_optim}."
            )
            raise ValueError(msg)

        if any(
            c.name == self.__SUB_OPTIM_CONSTRAINT_NAME for c in self.problem.constraints
        ):
            msg = (
                f"The constraint name {self.__SUB_OPTIM_CONSTRAINT_NAME} is protected "
                f"when using MNBI optimizer."
            )
            raise ValueError(msg)

    def _run(self, **options: Any) -> None:
        self.__n_processes = options.pop("n_processes")
        self._normalize_design_space = options.pop("normalize_design_space")
        self.__sub_optim_algo = options.pop("sub_optim_algo")
        self.__sub_optim_algo_options = options.pop("sub_optim_algo_options")
        self.__debug = options.pop("debug")
        self.__debug_file_path = options.pop("debug_file_path")
        self.__skip_betas = options.pop("skip_betas")
        self.__sub_optim_max_iter = options.pop("sub_optim_max_iter")
        self._doe_algo_options = options.pop("doe_algo_options")
        self.__n_obj = self.problem.objective.dim
        self.__ineq_tolerance = options.get(
            self.INEQ_TOLERANCE, self.problem.ineq_tolerance
        )
        self.__skippable_domains = Manager().list() if self.__n_processes > 1 else []
        if self.__debug:
            self._debug_results.clear()

        if self.__n_processes > 1:
            LOGGER.info("Running mNBI on %s processes", self.__n_processes)

        # Find the individual optimum phi of each objective function and the utopia
        self.__minimize_objective_components_separately()

        # Compute the quasi-normal vector to the simplex formed by the points in phi
        self.__n_vect = -dot(self.__phi - self.__utopia, ones(self.__n_obj))

        # Create the MDOFunction that runs the beta sub-optimization problem
        self.__create_beta_sub_optim()

        # Run the beta sub-problem for different values of beta corresponding to
        # different points of the phi simplex. The number of runs accounts for the
        # individual optimizations already performed.
        n_samples = self.__n_sub_optim - self.__n_obj
        if self.__n_obj == 2:
            betas = linspace(0, 1, n_samples + 2)[1:-1, newaxis]
        else:
            library = DOEFactory().create(options["doe_algo"])
            beta_design_space = DesignSpace()
            beta_design_space.add_variable(
                "beta", size=self.__n_obj - 1, l_b=0.0, u_b=1.0
            )
            betas = library.compute_doe(
                beta_design_space,
                size=n_samples,
                unit_sampling=True,
                **self._doe_algo_options,
            )

        execute(
            self._run_beta_sub_optim,
            [self.__beta_sub_optim_callback],
            self.__n_processes,
            betas,
        )

        if self.__debug:
            self._debug_results.to_hdf(self.__debug_file_path)
        return self.get_optimum_from_database()

    def _log_result(self) -> None:
        LOGGER.info("%s", self.problem.solution)
