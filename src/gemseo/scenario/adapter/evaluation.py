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
"""A discipline executing an evaluation scenario."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.core._process_flow.base_process_flow import BaseProcessFlow
from gemseo.core.discipline import Discipline
from gemseo.core.discipline.process_discipline import ProcessDiscipline
from gemseo.util.discipline import update_default_input_values
from gemseo.util.logging import LoggingContext
from gemseo.util.name_generator import NameGenerator
from gemseo.util.string import pretty_repr

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from numpy import ndarray

    from gemseo.core._process_flow.execution_sequence.loop import LoopExecSequence
    from gemseo.core.discipline.base_discipline import BaseDiscipline
    from gemseo.core.problem.database import Database
    from gemseo.scenario.evaluation import EvaluationScenario


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


class EvaluationScenarioAdapter(ProcessDiscipline):
    """A discipline executing an evaluation scenario.

    Its execution is in three stages:

    1. update the default input data of the top-level disciplines of the formulation
       from its own input data,
    2. execute the scenario,
    3. update its output data from the output data of the top-level disciplines.

    The output data are those of the last design point evaluated by the scenario,
    e.g. the last sample of a DOE algorithm.

    This adapter has no analytic Jacobian,
    as an [EvaluationProblem][gemseo.core.problem.evaluation.EvaluationProblem]
    has no optimum with respect to which differentiate.
    Use [MDOScenarioAdapter][gemseo.scenario.adapter.mdo.MDOScenarioAdapter]
    to adapt a scenario solving an
    [OptimizationProblem][gemseo.optimization.problem.OptimizationProblem],
    e.g. an [MDOScenario][gemseo.scenario.mdo.MDOScenario],
    and get its optimum, its Lagrange multipliers and its post-optimal Jacobian.
    """

    databases: list[Database]
    """The copies of the scenario databases after execution."""

    keep_databases: bool
    """Whether to keep copies of the database of the scenario after each execution."""

    save_databases: bool
    """Whether to save the database of the scenario after each execution."""

    scenario: EvaluationScenario
    """The scenario to be adapted."""

    _process_flow_class: ClassVar[type[BaseProcessFlow]] = _ProcessFlow

    LOWER_BND_SUFFIX: ClassVar[str] = "_lower_bnd"
    UPPER_BND_SUFFIX: ClassVar[str] = "_upper_bnd"

    DEFAULT_DATABASE_FILE_PREFIX: ClassVar[str] = "database"
    """The default file prefix for the databases to be exported."""

    _ATTR_NOT_TO_SERIALIZE = Discipline._ATTR_NOT_TO_SERIALIZE.union([
        "_EvaluationScenarioAdapter__name_generator"
    ])

    __name_generator: NameGenerator
    """A name generator used to get unique file names when exporting databases."""

    __naming: NameGenerator.Naming
    """The way of naming the files when exporting databases."""

    __scenario_log_level: int | None
    """The level of the root logger during the scenario execution.

    If `None`, do not change the level of the root logger.
    """

    def __init__(
        self,
        scenario: EvaluationScenario,
        input_names: Sequence[str],
        output_names: Sequence[str],
        reset_x0_before_exec: bool = False,
        set_x0_before_exec: bool = False,
        set_bounds_before_exec: bool = False,
        name: str = "",
        keep_databases: bool = False,
        save_databases: bool = False,
        database_file_prefix: str = "",
        scenario_log_level: int | None = None,
        naming: NameGenerator.Naming = NameGenerator.Naming.NUMBERED,
    ) -> None:
        """
        Args:
            scenario: The scenario to adapt.
            input_names: The names of the inputs of the top-level disciplines
                to overload before executing the scenario.
                These are the adapter's input variables.
            output_names: The names of the outputs of the top-level disciplines
                to get after executing the scenario.
                These are the adapter's output variables.
            reset_x0_before_exec: Whether to reset the current value
                of the design space of the scenario before executing it,
                to the value it had when the problem of the scenario was created.
            set_x0_before_exec: Whether to set the current value
                of the design space of the scenario before executing it,
                from the input data of this adapter.
                This is useful for multi-start optimization.
            set_bounds_before_exec: Whether to set the bounds
                of the design space of the scenario before executing it,
                from the input data of this adapter.
                This is useful for trust regions.
            name: The name of the scenario adapter.
                If empty,
                use the name of the scenario adapter suffixed by `"_adapter"`.
            keep_databases: Whether to keep copies
                of the database of the scenario after each execution.
                Depending on the size of the databases
                and the number of consecutive executions,
                this can be very memory consuming. If the adapter will be executed in
                parallel, the databases will not be saved to the main process by the
                sub-processes, so this argument should be set to `False` to avoid
                unnecessary memory use in the sub-processes.
            save_databases: Whether to save the database of the scenario
                to an HDF5 file after each execution.
            database_file_prefix: The base name for the databases to be exported.
                The full names of the databases are built from
                the provided base name suffixed by `"_identifier.h5"`
                where `identifier` is replaced by an identifier according to the
                `naming` convention.
                If empty, use
                [DEFAULT_DATABASE_FILE_PREFIX][gemseo.scenario.adapter.evaluation.EvaluationScenarioAdapter.DEFAULT_DATABASE_FILE_PREFIX].
            scenario_log_level: The level of the root logger
                during the scenario execution.
                If `None`, do not change the level of the root logger.
            naming: The way of naming the database files.
                When the adapter will be executed in parallel, this method shall be set
                to `UUID` because this method is multiprocess-safe.

        Raises:
            ValueError: If both `reset_x0_before_exec` and `set_x0_before_exec`
                are `True`.
        """  # noqa: D205, D212, D415
        if reset_x0_before_exec and set_x0_before_exec:
            msg = (
                "The options reset_x0_before_exec and set_x0_before_exec "
                f"of {type(self).__name__} cannot both be True."
            )
            raise ValueError(msg)
        self.scenario = scenario
        self._set_x0_before_exec = set_x0_before_exec
        self._set_bounds_before_exec = set_bounds_before_exec
        self._input_names = input_names
        self._output_names = output_names
        self._reset_x0_before_exec = reset_x0_before_exec
        self.__naming = naming
        self.keep_databases = keep_databases
        self.save_databases = save_databases
        self.databases = []
        self.__database_file_prefix = (
            database_file_prefix or self.DEFAULT_DATABASE_FILE_PREFIX
        )
        super().__init__((), name=name or f"{scenario.name}_adapter")

        self._update_grammars()
        self._dv_in_names = None
        if set_x0_before_exec:
            self._dv_in_names = list(
                set(self._input_names).intersection(self.scenario.design_space)
            )

        # Set the initial bounds as default bounds
        self._bound_names = []
        design_space = scenario.design_space
        if set_bounds_before_exec:
            defaults = self.io.input_grammar.defaults
            for bounds, suffix in [
                (
                    design_space.get_lower_bounds(as_dict=True),
                    self.LOWER_BND_SUFFIX,
                ),
                (
                    design_space.get_upper_bounds(as_dict=True),
                    self.UPPER_BND_SUFFIX,
                ),
            ]:
                bounds = {name + suffix: val for name, val in bounds.items()}
                defaults.update(bounds)
                self._bound_names.extend(bounds.keys())

        # Optimization functions are redefined at each run
        # since default inputs of top
        # level discipline change
        # History must be erased otherwise the wrong values are retrieved
        # between two runs
        scenario.clear_database_before_execute = True
        self.__scenario_log_level = scenario_log_level
        self._init_shared_memory_attrs_after()

    def _update_grammars(self) -> None:
        """Update the input and output grammars.

        Raises:
            ValueError: Either if a specified input is missing from the input grammar
                or if a specified output is missing from the output grammar.
        """
        formulation = self.scenario.formulation
        input_grammar = self.io.input_grammar
        output_grammar = self.io.output_grammar
        for discipline in formulation.get_top_level_disciplines():
            input_grammar.update(
                discipline.io.input_grammar, allow_namespace_nesting=True
            )
            output_grammar.update(
                discipline.io.output_grammar, allow_namespace_nesting=True
            )
            # The output may also be the optimum value of the design
            # variables, so the output grammar may contain inputs
            # of the disciplines. All grammars are filtered just after
            # this loop
            output_grammar.update(
                discipline.io.input_grammar, allow_namespace_nesting=True
            )
            input_grammar.defaults.update(discipline.io.input_grammar.defaults)

        try:
            input_grammar.restrict_to(self._input_names)
        except KeyError:
            missing_inputs = set(self._input_names).difference(input_grammar)
            if missing_inputs:
                msg = (
                    "Cannot compute inputs from scenarios: "
                    f"{pretty_repr(missing_inputs)}."
                )
                raise ValueError(msg) from None

        # Add the design variables bounds to the input grammar
        if self._set_bounds_before_exec:
            current_value = self.scenario.design_space.get_current_value(as_dict=True)
            bounds_grammar = input_grammar.__class__("bounds")
            bounds_grammar.update_from_data({
                variable_name + suffix: variable_value
                for variable_name, variable_value in current_value.items()
                for suffix in {
                    self.LOWER_BND_SUFFIX,
                    self.UPPER_BND_SUFFIX,
                }
            })
            input_grammar.update(bounds_grammar)

        # If a design variable is not an input of the top-level disciplines:
        missing_outputs = set(self._output_names).difference(output_grammar)
        if missing_outputs:
            missing_design_variables = set(missing_outputs).intersection(
                formulation.problem.design_space
            )
            if missing_design_variables:
                dv_grammar = output_grammar.__class__("dvs")
                dv_grammar.update_from_names(missing_design_variables)
                output_grammar.update(dv_grammar)

        try:
            output_grammar.restrict_to(self._output_names)
        except KeyError:
            missing_outputs = set(self._output_names).difference(output_grammar)
            if missing_outputs:
                msg = (
                    "Cannot compute outputs from scenarios: "
                    f"{pretty_repr(missing_outputs)}."
                )
                raise ValueError(msg) from None

    def _init_shared_memory_attrs_after(self) -> None:
        self.__name_generator = NameGenerator(naming=self.__naming)

    def _execute(self) -> None:
        self._pre_run()
        with LoggingContext(level=self.__scenario_log_level):
            self.scenario.execute()
        self._post_run()

    def _pre_run(self) -> None:
        """Pre-run the scenario."""
        # Adapter inputs and any previously-set outputs may both be needed here.
        data = self.io.get_merged_data()
        design_space = self.scenario.formulation.problem.design_space

        # Update the top level discipline default inputs with adapter inputs
        # This is the key role of the adapter
        update_default_input_values(
            self.scenario.formulation.get_top_level_disciplines(),
            data,
            self._input_names,
        )

        self._reset_problem()

        # Set the starting point of the sub scenario with current dv names
        if self._set_x0_before_exec:
            dv_values = {dv_name: data[dv_name] for dv_name in self._dv_in_names}
            design_space.set_current_value(dv_values)

        # Set the bounds of the sub-scenario
        if self._set_bounds_before_exec:
            lower_bound_suffix = self.LOWER_BND_SUFFIX
            upper_bound_suffix = self.UPPER_BND_SUFFIX
            for name in design_space:
                design_space.set_lower_bound(name, data[f"{name}{lower_bound_suffix}"])
                design_space.set_upper_bound(name, data[f"{name}{upper_bound_suffix}"])

    def _reset_problem(self) -> None:
        """Reset the problem attached to the scenario."""
        self.scenario.formulation.problem.reset(
            design_space=self._reset_x0_before_exec, database=False, preprocessing=False
        )

    def _post_run(self) -> None:
        """Post-process the scenario."""
        database = self.scenario.formulation.problem.database
        if self.keep_databases:
            self.databases.append(deepcopy(database))

        if self.save_databases:
            database.to_hdf(
                f"{self.__database_file_prefix}_{self.__name_generator.generate_name()}.h5"
            )

        self._evaluate_design_point_of_interest()
        self._retrieve_top_level_outputs()

    def _evaluate_design_point_of_interest(self) -> None:
        """Evaluate the functions of the problem at the design point of interest.

        An evaluation problem has no optimum;
        the design point of interest is the last one evaluated by the scenario,
        which is set as the current value of the design space.

        The only functions of an evaluation problem are its observables;
        they must be evaluated even though the design point is the last one evaluated,
        as a parallel DOE evaluates the samples in sub-processes
        and so leaves the disciplines of this process without their data.
        """
        problem = self.scenario.formulation.problem
        design_point = problem.database.get_x_vect(-1)
        problem.design_space.set_current_value(design_point)
        self._evaluate_functions(design_point, ())

    def _evaluate_functions(
        self,
        design_point: ndarray,
        observable_names: Iterable[str] | None,
    ) -> None:
        """Evaluate the functions of the problem at a design point.

        This re-executes all the disciplines and so provides them with the right data.

        Args:
            design_point: The design point at which to evaluate the functions.
            observable_names: The names of the observables to be evaluated.
                If empty, evaluate all the observables.
                If `None`, do not evaluate any observable.
        """
        problem = self.scenario.formulation.problem
        output_functions, jacobian_functions = problem.get_functions(
            no_db_no_norm=True, observable_names=observable_names
        )
        problem.evaluate_functions(
            design_vector=design_point,
            design_vector_is_normalized=False,
            output_functions=output_functions or None,
            jacobian_functions=jacobian_functions or None,
        )

    def _retrieve_top_level_outputs(self) -> None:
        """Retrieve the top-level outputs.

        This method overwrites the adapter outputs with the top-level discipline outputs
        and the current design values.
        """
        data = self.io.output_data
        formulation = self.scenario.formulation
        top_level_disciplines = formulation.get_top_level_disciplines()
        current_value = formulation.problem.design_space.get_current_value(as_dict=True)
        for output_name in self._output_names:
            for discipline in top_level_disciplines:
                if (
                    output_name in discipline.io.output_grammar
                    and output_name not in current_value
                ):
                    data[output_name] = discipline.io.output_data[output_name]

            if (output_value := current_value.get(output_name)) is not None:
                data[output_name] = output_value

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """Raise an error, as this adapter has no analytic Jacobian.

        Args:
            input_names: The names of the inputs
                with respect to which to differentiate the outputs.
            output_names: The names of the outputs to be differentiated.

        Raises:
            NotImplementedError: When called,
                as the scenario has no optimum with respect to which differentiate.
        """
        msg = (
            f"{type(self).__name__} has no analytic Jacobian; "
            "use MDOScenarioAdapter to adapt a scenario "
            "solving an OptimizationProblem, e.g. an MDOScenario."
        )
        raise NotImplementedError(msg)

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
