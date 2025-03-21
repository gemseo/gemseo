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
"""A discipline running a scenario."""

from __future__ import annotations

from copy import copy
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import zeros
from numpy.linalg import norm

from gemseo.algos.lagrange_multipliers import LagrangeMultipliers
from gemseo.algos.post_optimal_analysis import PostOptimalAnalysis
from gemseo.core._process_flow.base_process_flow import BaseProcessFlow
from gemseo.core.discipline import Discipline
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.core.parallel_execution.disc_parallel_linearization import (
    DiscParallelLinearization,
)
from gemseo.core.process_discipline import ProcessDiscipline
from gemseo.utils.logging_tools import LOGGING_SETTINGS
from gemseo.utils.logging_tools import LoggingContext
from gemseo.utils.name_generator import NameGenerator

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from numpy.core.multiarray import ndarray

    from gemseo.algos.database import Database
    from gemseo.algos.design_space import DesignSpace
    from gemseo.core._process_flow.execution_sequences.loop import LoopExecSequence
    from gemseo.core.discipline.base_discipline import BaseDiscipline
    from gemseo.scenarios.base_scenario import BaseScenario


class _ProcessFlow(BaseProcessFlow):
    """The process data and execution flow."""

    def get_execution_flow(self) -> LoopExecSequence:  # noqa: D102
        return self._node.scenario.get_process_flow().get_execution_flow()

    def get_disciplines_in_data_flow(self) -> list[BaseDiscipline]:
        """Return the disciplines that must be shown as blocks in the XDSM.

        By default, only the discipline itself is shown.
        This function can be differently implemented for any type of inherited
        discipline.

        Returns:
            The disciplines shown in the XDSM chain.
        """
        return [self._node]


class MDOScenarioAdapter(ProcessDiscipline):
    """An adapter class for MDO Scenario.

    The specified input variables update the default input data of the top level
    discipline while the output ones filter the output data from the top level
    discipline outputs.
    """

    databases: list[Database]
    """The copies of the scenario databases after execution."""

    keep_opt_history: bool
    """Whether to keep databases copies after each execution."""

    post_optimal_analysis: PostOptimalAnalysis
    """The post-optimal analysis."""

    save_opt_history: bool
    """Whether to save the optimization history after each execution."""

    scenario: BaseScenario
    """The scenario to be adapted."""

    _process_flow_class: ClassVar[type[BaseProcessFlow]] = _ProcessFlow

    LOWER_BND_SUFFIX: ClassVar[str] = "_lower_bnd"
    UPPER_BND_SUFFIX: ClassVar[str] = "_upper_bnd"
    MULTIPLIER_SUFFIX: ClassVar[str] = "_multiplier"
    DEFAULT_DATABASE_FILE_PREFIX: ClassVar[str] = "database"

    _ATTR_NOT_TO_SERIALIZE = Discipline._ATTR_NOT_TO_SERIALIZE.union([
        "_MDOScenarioAdapter__name_generator"
    ])

    __name_generator: NameGenerator
    """A name generator used to get unique file names when exporting databases."""

    __naming: NameGenerator.Naming
    """The way of naming the files when exporting databases."""

    __scenario_log_level: int | None
    """The level of the root logger during the scenario execution.

    If ``None``, do not change the level of the root logger.
    """

    def __init__(
        self,
        scenario: BaseScenario,
        input_names: Sequence[str],
        output_names: Sequence[str],
        reset_x0_before_opt: bool = False,
        set_x0_before_opt: bool = False,
        set_bounds_before_opt: bool = False,
        output_multipliers: bool = False,
        name: str = "",
        keep_opt_history: bool = False,
        save_opt_history: bool = False,
        opt_history_file_prefix: str = "",
        scenario_log_level: int | None = None,
        naming: NameGenerator.Naming = NameGenerator.Naming.NUMBERED,
    ) -> None:
        """
        Args:
            scenario: The scenario to adapt.
            input_names: The inputs to overload at sub-scenario execution.
            output_names: The outputs to get from the sub-scenario execution.
            reset_x0_before_opt: If ``True``, reset the initial guess
                before running the sub optimization.
            set_x0_before_opt: If ``True``, set the initial guess of the sub-scenario.
                This is useful for multi-start optimization.
            set_bounds_before_opt: If ``True``, set the bounds of the design space.
                This is useful for trust regions.
            output_multipliers: If ``True``,
                the Lagrange multipliers of the scenario optimal solution are computed
                and added to the outputs.
            name: The name of the scenario adapter.
                If empty,
                use the name of the scenario adapter suffixed by ``"_adapter"``.
            keep_opt_history: Whether to keep database copies after each execution.
                Depending on the size of the databases
                and the number of consecutive executions,
                this can be very memory consuming. If the adapter will be executed in
                parallel, the databases will not be saved to the main process by the
                sub-processes, so this argument should be set to ``False`` to avoid
                unnecessary memory use in the sub-processes.
            save_opt_history: Whether to save the optimization history
                to an HDF5 file after each execution.
            opt_history_file_prefix: The base name for the databases to be exported.
                The full names of the databases are built from
                the provided base name suffixed by ``"_identifier.h5"``
                where ``identifier`` is replaced by an identifier according to the
                ``naming_method``.
                If empty, use :attr:`.DEFAULT_DATABASE_FILE_PREFIX`.
            scenario_log_level: The level of the root logger
                during the scenario execution.
                If ``None``, do not change the level of the root logger.
            naming: The way of naming the database files.
                When the adapter will be executed in parallel, this method shall be set
                to ``UUID`` because this method is multiprocess-safe.

        Raises:
            ValueError: If both `reset_x0_before_opt` and `set_x0_before_opt` are True.
        """  # noqa: D205, D212, D415
        if reset_x0_before_opt and set_x0_before_opt:
            msg = "Inconsistent options for MDOScenarioAdapter."
            raise ValueError(msg)
        self.scenario = scenario
        self._set_x0_before_opt = set_x0_before_opt
        self._set_bounds_before_opt = set_bounds_before_opt
        self._input_names = input_names
        self._output_names = output_names
        self._reset_x0_before_opt = reset_x0_before_opt
        self._output_multipliers = output_multipliers
        self.__naming = naming
        self.keep_opt_history = keep_opt_history
        self.save_opt_history = save_opt_history
        self.databases = []
        self.__opt_history_file_prefix = (
            opt_history_file_prefix or self.DEFAULT_DATABASE_FILE_PREFIX
        )

        name = name or f"{scenario.name}_adapter"
        super().__init__((), name=name)

        self._update_grammars()
        self._dv_in_names = None
        if set_x0_before_opt:
            self._dv_in_names = list(
                set(self._input_names).intersection(self.scenario.design_space)
            )

        # Set the initial bounds as default bounds
        self._bound_names = []
        if set_bounds_before_opt:
            dspace = scenario.design_space
            lower_bounds = dspace.convert_array_to_dict(dspace.get_lower_bounds())
            lower_suffix = MDOScenarioAdapter.LOWER_BND_SUFFIX
            upper_bounds = dspace.convert_array_to_dict(dspace.get_upper_bounds())
            upper_suffix = MDOScenarioAdapter.UPPER_BND_SUFFIX
            for bounds, suffix in [
                (lower_bounds, lower_suffix),
                (upper_bounds, upper_suffix),
            ]:
                bounds = {name + suffix: val for name, val in bounds.items()}
                self.io.input_grammar.defaults.update(bounds)
                self._bound_names.extend(bounds.keys())

        # Optimization functions are redefined at each run
        # since default inputs of top
        # level discipline change
        # History must be erased otherwise the wrong values are retrieved
        # between two runs
        scenario.clear_history_before_execute = True
        self._initial_x = deepcopy(
            scenario.design_space.get_current_value(as_dict=True)
        )
        self.post_optimal_analysis = None
        self.__scenario_log_level = scenario_log_level
        self._init_shared_memory_attrs_after()

    def _update_grammars(self) -> None:
        """Update the input and output grammars.

        Raises:
            ValueError: Either if a specified input is missing from the input grammar
                or if a specified output is missing from the output grammar.
        """
        formulation = self.scenario.formulation
        opt_problem = formulation.optimization_problem
        top_leveld = formulation.get_top_level_disciplines()
        for disc in top_leveld:
            self.io.input_grammar.update(disc.io.input_grammar)
            self.io.output_grammar.update(disc.io.output_grammar)
            # The output may also be the optimum value of the design
            # variables, so the output grammar may contain inputs
            # of the disciplines. All grammars are filtered just after
            # this loop
            self.io.output_grammar.update(disc.io.input_grammar)
            self.io.input_grammar.defaults.update(disc.io.input_grammar.defaults)

        try:
            self.io.input_grammar.restrict_to(self._input_names)
        except KeyError:
            missing_inputs = set(self._input_names) - set(self.io.input_grammar)

            if missing_inputs:
                msg = "Can't compute inputs from scenarios: {}.".format(
                    ", ".join(sorted(missing_inputs))
                )
                raise ValueError(msg) from None

        # Add the design variables bounds to the input grammar
        if self._set_bounds_before_opt:
            current_x = self.scenario.design_space.get_current_value(as_dict=True)
            names_to_values = {}
            for suffix in [
                MDOScenarioAdapter.LOWER_BND_SUFFIX,
                MDOScenarioAdapter.UPPER_BND_SUFFIX,
            ]:
                names_to_values.update({k + suffix: v for k, v in current_x.items()})
            bounds_grammar = JSONGrammar("bounds")
            bounds_grammar.update_from_data(names_to_values)
            self.io.input_grammar.update(bounds_grammar)

        # If a DV is not an input of the top level disciplines:
        missing_outputs = set(self._output_names) - set(self.io.output_grammar)

        if missing_outputs:
            miss_dvs = set(missing_outputs).intersection(opt_problem.design_space)
            if miss_dvs:
                dv_gram = JSONGrammar("dvs")
                dv_gram.update_from_names(miss_dvs)
                self.io.output_grammar.update(dv_gram)

        try:
            self.io.output_grammar.restrict_to(self._output_names)
        except KeyError:
            missing_outputs = set(self._output_names) - set(self.io.output_grammar)

            if missing_outputs:
                msg = "Can't compute outputs from scenarios: {}.".format(
                    ", ".join(sorted(missing_outputs))
                )
                raise ValueError(msg) from None

        # Add the Lagrange multipliers to the output grammar
        if self._output_multipliers:
            self._add_output_multipliers()

    def _add_output_multipliers(self) -> None:
        """Add the Lagrange multipliers of the scenario optimal solution as outputs."""
        # Fill a dictionary with data of typical shapes
        base_dict = {}
        problem = self.scenario.formulation.optimization_problem
        # bound-constraints multipliers
        current_x = problem.design_space.get_current_value(as_dict=True)
        base_dict.update({
            self.get_bnd_mult_name(var_name, False): val
            for var_name, val in current_x.items()
        })
        base_dict.update({
            self.get_bnd_mult_name(var_name, True): val
            for var_name, val in current_x.items()
        })
        # equality- and inequality-constraints multipliers
        base_dict.update({
            self.get_cstr_mult_name(cstr_name): zeros(1)
            for cstr_name in problem.constraints.get_names()
        })

        # Update the output grammar
        multipliers_grammar = JSONGrammar("multipliers")
        multipliers_grammar.update_from_data(base_dict)
        self.io.output_grammar.update(multipliers_grammar)

    def _init_shared_memory_attrs_after(self) -> None:
        self.__name_generator = NameGenerator(naming_method=self.__naming)

    @staticmethod
    def get_bnd_mult_name(
        variable_name: str,
        is_upper: bool,
    ) -> str:
        """Return the name of the lower bound-constraint multiplier of a variable.

        Args:
            variable_name: The name of the variable.
            is_upper: If ``True``, return name of the upper bound-constraint multiplier.
                Otherwise, return the name of the lower bound-constraint multiplier.

        Returns:
            The name of a bound-constraint multiplier.
        """
        mult_name = variable_name
        mult_name += "_upp-bnd" if is_upper else "_low-bnd"
        mult_name += MDOScenarioAdapter.MULTIPLIER_SUFFIX
        return mult_name

    @staticmethod
    def get_cstr_mult_name(
        constraint_name: str,
    ) -> str:
        """Return the name of the multiplier of a constraint.

        Args:
            constraint_name: The name of the constraint.

        Returns:
            The name of the multiplier.
        """
        return constraint_name + MDOScenarioAdapter.MULTIPLIER_SUFFIX

    def _execute(self) -> None:
        self._pre_run()
        with LoggingContext(LOGGING_SETTINGS.logger, level=self.__scenario_log_level):
            self.scenario.execute()
        self._post_run()

    def _pre_run(self) -> None:
        """Pre-run the scenario."""
        formulation = self.scenario.formulation
        design_space: DesignSpace = formulation.optimization_problem.design_space
        top_leveld = formulation.get_top_level_disciplines()

        # Update the top level discipline default inputs with adapter inputs
        # This is the key role of the adapter
        for indata in self._input_names:
            for disc in top_leveld:
                if indata in disc.io.input_grammar:
                    disc.io.input_grammar.defaults[indata] = self.io.data[indata]

        self._reset_optimization_problem()

        # Set the starting point of the sub scenario with current dv names
        if self._set_x0_before_opt:
            dv_values = {dv_n: self.io.data[dv_n] for dv_n in self._dv_in_names}
            self.scenario.formulation.design_space.set_current_value(dv_values)

        # Set the bounds of the sub-scenario
        if self._set_bounds_before_opt:
            for name in design_space:
                # Set the lower bound
                lower_suffix = MDOScenarioAdapter.LOWER_BND_SUFFIX
                lower_bound = self.io.data[name + lower_suffix]
                design_space.set_lower_bound(name, lower_bound)
                # Set the upper bound
                upper_suffix = MDOScenarioAdapter.UPPER_BND_SUFFIX
                upper_bound = self.io.data[name + upper_suffix]
                design_space.set_upper_bound(name, upper_bound)

    def _reset_optimization_problem(self) -> None:
        """Reset the optimization problem."""
        self.scenario.formulation.optimization_problem.reset(
            design_space=self._reset_x0_before_opt, database=False, preprocessing=False
        )

    def _post_run(self) -> None:
        """Post-process the scenario."""
        formulation = self.scenario.formulation
        opt_problem = formulation.optimization_problem
        design_space = opt_problem.design_space

        database = opt_problem.database
        if self.keep_opt_history:
            self.databases.append(deepcopy(database))
        if self.save_opt_history:
            database.to_hdf(
                f"{self.__opt_history_file_prefix}_{self.__name_generator.generate_name()}.h5"
            )

        # Test if the last evaluation is the optimum
        x_opt = design_space.get_current_value()
        last_x = database.get_x_vect(-1)
        last_eval_not_opt = norm(x_opt - last_x) / (1.0 + norm(last_x)) > 1e-14
        if last_eval_not_opt:
            # Revaluate all functions at optimum
            # To re-execute all disciplines and get the right data
            output_functions, jacobian_functions = opt_problem.get_functions(
                no_db_no_norm=True
            )
            opt_problem.evaluate_functions(
                design_vector=x_opt,
                design_vector_is_normalized=False,
                output_functions=output_functions or None,
                jacobian_functions=jacobian_functions or None,
            )

        # Retrieves top-level discipline outputs
        self._retrieve_top_level_outputs()

        # Compute the Lagrange multipliers and store them in the local data
        if self._output_multipliers:
            self._compute_lagrange_multipliers()

    def _retrieve_top_level_outputs(self) -> None:
        """Retrieve the top-level outputs.

        This method overwrites the adapter outputs with the top-level discipline outputs
        and the optimal design parameters.
        """
        formulation = self.scenario.formulation
        top_level_disciplines = formulation.get_top_level_disciplines()
        current_x = formulation.optimization_problem.design_space.get_current_value(
            as_dict=True
        )
        for name in self._output_names:
            for discipline in top_level_disciplines:
                if name in discipline.io.output_grammar and name not in current_x:
                    self.io.data[name] = discipline.io.data[name]

            output_value_in_current_x = current_x.get(name)
            if output_value_in_current_x is not None:
                self.io.data[name] = output_value_in_current_x

    def _compute_lagrange_multipliers(self) -> None:
        """Compute the Lagrange multipliers for the optimal solution of the scenario.

        This method stores the multipliers in the local data.
        """
        # Compute the Lagrange multipliers
        problem = self.scenario.formulation.optimization_problem
        x_opt = problem.solution.x_opt
        lagrange = LagrangeMultipliers(problem)
        lagrange.compute(x_opt, problem.tolerances.inequality)

        # Store the Lagrange multipliers in the local data
        multipliers = lagrange.get_multipliers_arrays()
        self.io.data.update({
            self.get_bnd_mult_name(name, False): mult
            for name, mult in multipliers[lagrange.LOWER_BOUNDS].items()
        })
        self.io.data.update({
            self.get_bnd_mult_name(name, True): mult
            for name, mult in multipliers[lagrange.UPPER_BOUNDS].items()
        })
        self.io.data.update({
            self.get_cstr_mult_name(name): mult
            for name, mult in multipliers[lagrange.EQUALITY].items()
        })
        self.io.data.update({
            self.get_cstr_mult_name(name): mult
            for name, mult in multipliers[lagrange.INEQUALITY].items()
        })

    def _compute_jacobian(
        self,
        input_names: Sequence[str] = (),
        output_names: Sequence[str] = (),
    ) -> None:
        """Compute the Jacobian of the adapted scenario outputs.

        The Jacobian is stored as a dictionary of numpy arrays:
        jac = {name: { input_name: ndarray(output_dim, input_dim) } }

        The bound-constraints on the scenario optimization variables
        are assumed independent of the other scenario inputs.

        Raises:
            ValueError: Either
                if the dimension of the objective function is greater than 1,
                if a specified input is not an input of the adapter,
                if a specified output is not an output of the adapter,
                or if there is non-differentiable outputs.
        """
        opt_problem = self.scenario.formulation.optimization_problem
        objective_names = (
            self.scenario.formulation.optimization_problem.objective.output_names
        )
        if len(objective_names) != 1:
            msg = "The objective must be single-valued."
            raise ValueError(msg)

        # Check the required inputs
        if not input_names:
            input_names = set(self._input_names + self._bound_names)
        else:
            not_inputs = (
                set(input_names) - set(self._input_names) - set(self._bound_names)
            )
            if not_inputs:
                msg = "The following are not inputs of the adapter: {}.".format(
                    ", ".join(sorted(not_inputs))
                )
                raise ValueError(msg)
        # N.B the adapter is assumed constant w.r.t. bounds
        bound_inputs = set(input_names) & set(self._bound_names)

        # Check the required outputs
        if not output_names:
            output_names = objective_names
        else:
            not_outputs = sorted(set(output_names) - set(self._output_names))
            if not_outputs:
                msg = "The following are not outputs of the adapter: {}.".format(
                    ", ".join(not_outputs)
                )
                raise ValueError(msg)

        non_differentiable_outputs = sorted(set(output_names) - set(objective_names))
        if non_differentiable_outputs:
            msg = "Post-optimal Jacobians of {} cannot be computed.".format(
                ", ".join(non_differentiable_outputs)
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
        ineq_tolerance = opt_problem.tolerances.inequality
        self.post_optimal_analysis = PostOptimalAnalysis(opt_problem, ineq_tolerance)
        post_opt_jac = self.post_optimal_analysis.execute(
            output_names, diff_inputs, jacobians
        )
        self.jac.update(post_opt_jac)

        # Fill the Jacobian blocks w.r.t. bounds with zeros
        for output_derivatives in self.jac.values():
            for bound_input_name in bound_inputs:
                bound_input_size = self.io.input_grammar.defaults[bound_input_name].size
                output_derivatives[bound_input_name] = zeros((1, bound_input_size))

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
        opt_problem = self.scenario.formulation.optimization_problem
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
        post_opt_data = copy(self.io.data)
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

    def add_outputs(
        self,
        output_names: Iterable[str],
    ) -> None:
        """Add outputs to the scenario adapter.

        Args:
            output_names: The names of the outputs to be added.
        """
        names_to_add = [name for name in output_names if name not in self._output_names]
        self._output_names.extend(names_to_add)
        self._update_grammars()
