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
#                        documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#        :author: Pierre-Jean Barjhoux, Benoit Pauwels - MDOScenarioAdapter
#                                                        Jacobian computation
"""A discipline executing a scenario solving an optimization problem."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final
from typing import cast

from numpy import atleast_1d
from numpy import zeros
from numpy.linalg import norm

from gemseo.core.discipline import Discipline
from gemseo.core.parallel_execution.discipline_linearization import (
    DiscParallelLinearization,
)
from gemseo.optimization.lagrange_multipliers import LagrangeMultipliers
from gemseo.optimization.post_optimal_analysis import PostOptimalAnalysis
from gemseo.optimization.problem import OptimizationProblem
from gemseo.scenario.adapter.evaluation import EvaluationScenarioAdapter
from gemseo.util.name_generator import NameGenerator
from gemseo.util.string import pretty_repr

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from numpy import ndarray

    from gemseo.scenario.evaluation import EvaluationScenario

_SINGLE_VALUED_OBJECTIVE_MESSAGE: Final[str] = "The objective must be single-valued."
"""The message of the error raised when the objective is not single-valued."""


class MDOScenarioAdapter(EvaluationScenarioAdapter):
    """A discipline executing a scenario solving an optimization problem.

    Unlike
    [EvaluationScenarioAdapter][gemseo.scenario.adapter.evaluation.EvaluationScenarioAdapter],
    this adapter requires a scenario solving an
    [OptimizationProblem][gemseo.optimization.problem.OptimizationProblem],
    e.g. an [MDOScenario][gemseo.scenario.mdo.MDOScenario],
    and raises a `TypeError` at instantiation otherwise.

    Its output data are those of the optimum,
    it can report the optimal objective value recorded by the scenario,
    it can add the Lagrange multipliers of the optimal solution to its outputs
    and it can be linearized by post-optimal analysis.
    """

    post_optimal_analysis: PostOptimalAnalysis | None
    """The post-optimal analysis, if the adapter has been linearized."""

    MULTIPLIER_SUFFIX: ClassVar[str] = "_multiplier"

    def __init__(
        self,
        scenario: EvaluationScenario,
        input_names: Sequence[str],
        output_names: Sequence[str],
        reset_x0_before_exec: bool = False,
        set_x0_before_exec: bool = False,
        set_bounds_before_exec: bool = False,
        output_multipliers: bool = False,
        name: str = "",
        keep_databases: bool = False,
        save_databases: bool = False,
        database_file_prefix: str = "",
        scenario_log_level: int | None = None,
        naming: NameGenerator.Naming = NameGenerator.Naming.NUMBERED,
        output_optimal_objective: bool = False,
    ) -> None:
        """
        Args:
            output_multipliers: Whether to compute
                the Lagrange multipliers of the scenario optimal solution
                and add them to the outputs.
            output_optimal_objective: Whether to report
                the optimal value of the objective recorded by the scenario,
                instead of the value that its disciplines computed
                at the optimal design point.
                The objective must then be single-valued.
                The post-optimal Jacobian of the objective is then
                the Lagrange multipliers times the Jacobian of the constraints,
                as the objective is assumed independent
                of the input variables of this adapter.

        Raises:
            TypeError: If the scenario does not solve an optimization problem.
            ValueError: If the optimal objective value is required
                while the objective is already known to be multi-valued.
        """  # noqa: D205, D212, D415
        problem = scenario.formulation.problem
        if not isinstance(problem, OptimizationProblem):
            msg = (
                f"The scenario {scenario.name} defines a problem of type "
                f"{type(problem).__name__}; an OptimizationProblem is required, "
                "e.g. that of an MDOScenario."
            )
            raise TypeError(msg)

        if output_optimal_objective:
            is_mono_objective = True
            with contextlib.suppress(ValueError):
                # The dimension of the objective is unknown while the latter is unset,
                # e.g. when the adapter is built before the call to add_objective;
                # _retrieve_top_level_outputs checks it again after the execution.
                is_mono_objective = problem.is_mono_objective

            if not is_mono_objective:
                raise ValueError(_SINGLE_VALUED_OBJECTIVE_MESSAGE)

        self._output_multipliers = output_multipliers
        self._output_optimal_objective = output_optimal_objective
        self.post_optimal_analysis = None
        super().__init__(
            scenario,
            input_names,
            output_names,
            reset_x0_before_exec=reset_x0_before_exec,
            set_x0_before_exec=set_x0_before_exec,
            set_bounds_before_exec=set_bounds_before_exec,
            name=name,
            keep_databases=keep_databases,
            save_databases=save_databases,
            database_file_prefix=database_file_prefix,
            scenario_log_level=scenario_log_level,
            naming=naming,
        )

    @property
    def _optimization_problem(self) -> OptimizationProblem:
        """The optimization problem attached to the scenario."""
        return cast("OptimizationProblem", self.scenario.formulation.problem)

    def _update_grammars(self) -> None:
        super()._update_grammars()
        # Add the Lagrange multipliers to the output grammar
        if self._output_multipliers:
            self._add_output_multipliers()

    def _add_output_multipliers(self) -> None:
        """Add the Lagrange multipliers of the scenario optimal solution as outputs."""
        # Fill a dictionary with data of typical shapes
        name_to_value = {}
        problem = self._optimization_problem
        # bound-constraints multipliers
        current_value = problem.design_space.get_current_value(as_dict=True)
        name_to_value.update({
            self.get_bnd_mult_name(variable_name, False): variable_value
            for variable_name, variable_value in current_value.items()
        })
        name_to_value.update({
            self.get_bnd_mult_name(variable_name, True): variable_value
            for variable_name, variable_value in current_value.items()
        })
        # equality- and inequality-constraints multipliers
        name_to_value.update({
            self.get_cstr_mult_name(constraint_name): zeros(1)
            for constraint_name in problem.constraints.get_names()
        })

        # Update the output grammar
        multipliers_grammar = self.io.output_grammar.__class__("multipliers")
        multipliers_grammar.update_from_data(name_to_value)
        self.io.output_grammar.update(multipliers_grammar)

    @classmethod
    def get_bnd_mult_name(
        cls,
        variable_name: str,
        is_upper: bool,
    ) -> str:
        """Return the name of the lower bound-constraint multiplier of a variable.

        Args:
            variable_name: The name of the variable.
            is_upper: If `True`, return name of the upper bound-constraint multiplier.
                Otherwise, return the name of the lower bound-constraint multiplier.

        Returns:
            The name of a bound-constraint multiplier.
        """
        upp_or_low = "upp" if is_upper else "low"
        return f"{variable_name}_{upp_or_low}-bnd{cls.MULTIPLIER_SUFFIX}"

    @classmethod
    def get_cstr_mult_name(
        cls,
        constraint_name: str,
    ) -> str:
        """Return the name of the multiplier of a constraint.

        Args:
            constraint_name: The name of the constraint.

        Returns:
            The name of the multiplier.
        """
        return constraint_name + cls.MULTIPLIER_SUFFIX

    def _post_run(self) -> None:
        super()._post_run()
        # Compute the Lagrange multipliers and store them in the local data
        if self._output_multipliers:
            self._compute_lagrange_multipliers()

    def _evaluate_design_point_of_interest(self) -> None:
        """Evaluate the functions of the problem at the design point of interest.

        The design point of interest is the optimum,
        which the driver has set as the current value of the design space.

        The functions of an optimization problem include its objective,
        so evaluating them does execute the disciplines;
        the observables are not needed for that.

        The evaluation is skipped when the last evaluation of the scenario
        already is that of the design point of interest,
        except when that evaluation left the disciplines of this process
        without the data of this single design point.
        """
        problem = self._optimization_problem
        design_point = problem.design_space.get_current_value()
        last_x = problem.database.get_x_vect(-1)
        if (
            self.__is_last_evaluation_unusable()
            or norm(design_point - last_x) / (1.0 + norm(last_x)) > 1e-14
        ):
            self._evaluate_functions(design_point, None)

    def __is_last_evaluation_unusable(self) -> bool:
        """Check whether the last evaluation of the scenario left no data to retrieve.

        Three kinds of drivers do not leave the disciplines of this process
        with the data of the last design point they evaluated:
        those evaluating in sub-processes,
        be they the samples of a DOE or the sub-optimizations of a multi-start,
        and those subdividing their population over a pool of worker processes,
        e.g. DIFFERENTIAL_EVOLUTION and SHGO,
        which both leave these disciplines without any data,
        and a DOE evaluating the whole sample set in a single vectorized call,
        which leaves them with one value per sample.

        Returns:
            Whether the disciplines of this process do not hold the data
            of the last design point evaluated by the scenario.
        """
        # The driver library resets its settings at the end of the execution,
        # while the scenario keeps the ones it was given.
        # A number of processes is declared by the settings of a DOE algorithm
        # and by those of the optimizers parallelizing their sub-optimizations,
        # e.g. MultiStart and MNBI,
        # a number of workers by those of the SciPy global optimizers subdividing
        # their population over a process pool, e.g. DIFFERENTIAL_EVOLUTION and SHGO,
        # while a vectorization flag is declared by the DOE settings only;
        # the number of processes and the vectorization flag are mutually exclusive.
        # The number of workers is compared with 1 instead of being bounded from below,
        # as that of the differential evolution is a plain integer and so also accepts
        # -1, the SciPy shorthand for every core.
        # Every other algorithm evaluates the design points one at a time in this
        # process, which the defaults of these three lookups report.
        settings = self.scenario._algorithm_settings
        return (
            getattr(settings, "n_processes", 1) > 1
            or getattr(settings, "workers", 1) != 1
            or getattr(settings, "vectorize", False)
        )

    def _retrieve_top_level_outputs(self) -> None:
        super()._retrieve_top_level_outputs()
        if not self._output_optimal_objective:
            return

        problem = self._optimization_problem
        # The dimension of the objective may only be known once it has been evaluated;
        # __init__ raises for a problem already known to be multi-objective.
        if not problem.is_mono_objective:
            raise ValueError(_SINGLE_VALUED_OBJECTIVE_MESSAGE)

        objective_name = problem.objective.output_names[0]
        if objective_name in self._output_names:
            # The optimum of the problem is its standardized, i.e. minimized, objective.
            f_opt = problem.optimum[0]
            if not problem.minimize_objective:
                f_opt = -f_opt

            self.io.output_data[objective_name] = atleast_1d(f_opt)

    def _compute_lagrange_multipliers(self) -> None:
        """Compute the Lagrange multipliers for the optimal solution of the scenario.

        This method stores the multipliers in the local data.
        """
        # Compute the Lagrange multipliers
        problem = self._optimization_problem
        x_opt = problem.solution.x_opt
        lagrange = LagrangeMultipliers(problem)
        lagrange.compute(x_opt, problem.tolerances.inequality)

        # Store the Lagrange multipliers in the local data
        multipliers = lagrange.get_multipliers_arrays()
        self.io.output_data.update({
            self.get_bnd_mult_name(name, False): mult
            for name, mult in multipliers[lagrange.LOWER_BOUNDS].items()
        })
        self.io.output_data.update({
            self.get_bnd_mult_name(name, True): mult
            for name, mult in multipliers[lagrange.UPPER_BOUNDS].items()
        })
        self.io.output_data.update({
            self.get_cstr_mult_name(name): mult
            for name, mult in multipliers[lagrange.EQUALITY].items()
        })
        self.io.output_data.update({
            self.get_cstr_mult_name(name): mult
            for name, mult in multipliers[lagrange.INEQUALITY].items()
        })

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """Compute the Jacobian of the adapted scenario outputs.

        The Jacobian is stored as a dictionary of numpy arrays:
        jac = {name: { input_name: ndarray(output_dim, input_dim) } }

        The bound-constraints on the scenario optimization variables
        are assumed independent of the other scenario inputs.

        Args:
            input_names: The names of the inputs
                with respect to which to differentiate the outputs.
            output_names: The names of the outputs to be differentiated.

        Raises:
            ValueError: Either
                if the objective is not single-valued,
                if a specified input is not an input of the adapter,
                if a specified output is not an output of the adapter,
                or if there is non-differentiable outputs.
        """
        optimization_problem = self._optimization_problem
        if not optimization_problem.is_mono_objective:
            raise ValueError(_SINGLE_VALUED_OBJECTIVE_MESSAGE)

        objective_names = optimization_problem.objective.output_names

        # Check the required inputs
        if input_names:
            if names := (
                set(input_names) - set(self._input_names) - set(self._bound_names)
            ):
                msg = (
                    "The following are not inputs of the adapter: "
                    f"{pretty_repr(names)}."
                )
                raise ValueError(msg)
        else:
            input_names = set(self._input_names + self._bound_names)

        # N.B the adapter is assumed constant w.r.t. bounds
        bound_inputs = set(input_names) & set(self._bound_names)

        # Check the required outputs
        if output_names:
            if names := (set(output_names).difference(self._output_names)):
                msg = (
                    "The following are not outputs of the adapter: "
                    f"{pretty_repr(names)}."
                )
                raise ValueError(msg)
        else:
            output_names = objective_names

        if names := (set(output_names).difference(objective_names)):
            msg = (
                f"The post-optimal Jacobians of {pretty_repr(names)} "
                f"cannot be computed."
            )
            raise ValueError(msg)

        # Initialize the Jacobian
        diff_inputs = [name for name in input_names if name not in bound_inputs]
        # N.B. there may be only bound inputs
        self._init_jacobian(
            diff_inputs, output_names, init_type=Discipline.InitJacobianType.EMPTY
        )

        # Compute the Jacobians of the optimization functions
        jacobians = self._compute_auxiliary_jacobians(diff_inputs)

        # Perform the post-optimal analysis
        self.post_optimal_analysis = PostOptimalAnalysis(
            optimization_problem, optimization_problem.tolerances.inequality
        )
        post_opt_jac = self.post_optimal_analysis.execute(
            output_names, diff_inputs, jacobians
        )
        self.jac.update(post_opt_jac)

        # Fill the Jacobian blocks w.r.t. bounds with zeros
        defaults = self.io.input_grammar.defaults
        for output_derivatives in self.jac.values():
            for bound_input_name in bound_inputs:
                bound_input_size = defaults[bound_input_name].size
                output_derivatives[bound_input_name] = zeros((1, bound_input_size))

        if self._output_optimal_objective:
            # The objective output is the optimal value recorded by the scenario,
            # which the disciplines do not compute
            # as a function of the input variables of this adapter:
            # the partial derivative of the objective w.r.t. them is assumed zero
            # and its total derivative reduces to the multiplier term.
            self.jac[objective_names[0]] = dict(
                self.jac[PostOptimalAnalysis.MULT_DOT_CONSTR_JAC]
            )

    def _compute_auxiliary_jacobians(
        self,
        input_names: Iterable[str],
        func_names: Iterable[str] = (),
        use_threading: bool = True,
    ) -> dict[str, dict[str, ndarray]]:
        """Compute the Jacobians of the optimization functions.

        Args:
            input_names: The names of the inputs w.r.t. which differentiate.
            func_names: The names of the functions to differentiate
                If empty, then all the optimizations functions are differentiated.
            use_threading: Whether to use threads instead of processes
                to parallelize the execution;
                multiprocessing will copy (serialize) all the disciplines,
                while threading will share all the memory.
                This is important to note
                if you want to execute the same discipline multiple times,
                you shall use multiprocessing.

        Returns:
            The Jacobians of the optimization functions.
        """
        # Gather the names of the functions to differentiate
        opt_problem = self._optimization_problem
        if not func_names:
            func_names = opt_problem.objective.output_names + [
                output_name
                for constraint in opt_problem.constraints
                for output_name in constraint.output_names
            ]

        # Identify the disciplines that compute the functions
        disciplines = {}
        for func_name in func_names:
            for discipline in self.scenario.formulation.get_top_level_disciplines():
                if func_name in discipline.io.output_grammar:
                    disciplines[func_name] = discipline
                    break

        # Linearize the required disciplines
        unique_disciplines = list(set(disciplines.values()))
        for discipline in unique_disciplines:
            diff_inputs = set(discipline.io.input_grammar) & set(input_names)
            diff_outputs = set(discipline.io.output_grammar) & set(func_names)
            if diff_inputs and diff_outputs:
                discipline.add_differentiated_inputs(list(diff_inputs))
                discipline.add_differentiated_outputs(list(diff_outputs))

        parallel_linearization = DiscParallelLinearization(
            unique_disciplines, use_threading=use_threading
        )
        # Update the local data with the optimal design parameters
        # [The adapted scenario is assumed to have been run beforehand.]
        post_opt_data = self.io.get_merged_data()
        post_opt_data.update(opt_problem.design_space.get_current_value(as_dict=True))
        parallel_linearization.execute([post_opt_data] * len(unique_disciplines))

        # Store the Jacobians
        jacobians = {}
        for func_name in func_names:
            jacobians[func_name] = {}
            func_jacobian = disciplines[func_name].jac[func_name]
            for input_name in input_names:
                jacobians[func_name][input_name] = func_jacobian[input_name]

        return jacobians
