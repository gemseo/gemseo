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
"""Scenario for evaluating disciplinary outputs from disciplinary inputs."""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from numpy import array
from numpy import complex128
from numpy import float64
from numpy import ndarray

from gemseo.algos.doe.base_doe_settings import BaseDOESettings
from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.algos.evaluation_problem import EvaluationProblem
from gemseo.core._base_monitored_process import BaseMonitoredProcess
from gemseo.core._process_flow.base_process_flow import BaseProcessFlow
from gemseo.core._process_flow.execution_sequences.loop import LoopExecSequence
from gemseo.core._process_flow.execution_sequences.parallel import ParallelExecSequence
from gemseo.core._process_flow.execution_sequences.sequential import (
    SequentialExecSequence,
)
from gemseo.core.execution_statistics import ExecutionStatistics
from gemseo.formulations.factory import MDO_FORMULATION_FACTORY
from gemseo.formulations.mdf_settings import MDF_Settings
from gemseo.utils.discipline import get_all_outputs
from gemseo.utils.discipline import get_sub_disciplines
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import convert_strings_to_iterable
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.algos.base_algo_factory import BaseAlgoFactory
    from gemseo.algos.base_driver_settings import BaseDriverSettings
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.doe.base_doe_library import BaseDOELibrary
    from gemseo.algos.driver_library import DriverLibraryFactory
    from gemseo.core.discipline.base_discipline import BaseDiscipline
    from gemseo.core.functions.array_function import ArrayFunction
    from gemseo.datasets.dataset import Dataset
    from gemseo.formulations.base import BaseFormulation
    from gemseo.formulations.base_settings import BaseFormulationSettings
    from gemseo.formulations.factory import MDOFormulationFactory
    from gemseo.typing import RealArray
    from gemseo.utils.xdsm.xdsm import XDSM

LOGGER = logging.getLogger(__name__)


class _ScenarioProcessFlow(BaseProcessFlow):
    """The process data and execution flow."""

    def get_data_flow(  # noqa:D102
        self,
    ) -> list[tuple[BaseDiscipline, BaseDiscipline, list[str]]]:
        top_level_discs = self._node.formulation.get_top_level_disciplines()
        if len(top_level_discs) == 1:
            return top_level_discs[0].get_process_flow().get_data_flow()
        data_flow = []
        for disc in top_level_discs:
            data_flow.extend(disc.get_process_flow().get_data_flow())
        return data_flow

    def get_execution_flow(self) -> LoopExecSequence:  # noqa:D102
        top_level_discs = self._node.formulation.get_top_level_disciplines()
        sequence = (
            SequentialExecSequence()
            if len(top_level_discs) == 1
            else ParallelExecSequence()
        )
        for disc in top_level_discs:
            sequence.extend(disc.get_process_flow().get_execution_flow())
        return LoopExecSequence(self._node, sequence)

    def get_disciplines_in_data_flow(self) -> list[BaseDiscipline]:
        return [self._node]


class EvaluationScenario(BaseMonitoredProcess):
    """Scenario for evaluating disciplinary outputs from disciplinary inputs.

    The outputs of interest are declared as observables
    using the
    [add_observable()][gemseo.scenarios.evaluation.EvaluationScenario.add_observable]
    method.
    These observables are attached to
    an [EvaluationProblem][gemseo.algos.evaluation_problem.EvaluationProblem],
    built over the [DesignSpace][gemseo.algos.design_space.DesignSpace],
    that is passed at instantiation.
    """

    _ALGO_FACTORY_CLASS: ClassVar[type[DriverLibraryFactory]] = DOELibraryFactory
    """The type of algorithm factory."""

    _algo_factory: BaseAlgoFactory
    """The algorithm factory."""

    clear_database_before_execute: bool
    """Whether to clear the database before execute."""

    DifferentiationMethod = EvaluationProblem.DifferentiationMethod
    """The enumeration of differentiation methods."""

    _evaluation_problem_class: type[EvaluationProblem] = EvaluationProblem
    """The type of evaluation problem."""

    _execution_result: Any
    """The result of the last execution."""

    _formulation_factory: ClassVar[MDOFormulationFactory] = MDO_FORMULATION_FACTORY
    """The factory of MDO formulations."""

    formulation: BaseFormulation
    """The MDO formulation."""

    _backup_evaluations: bool
    """Whether to back-up evaluations during the execution."""

    _backup_file_path: Path
    """The backup file path."""

    _process_flow_class: ClassVar[type[BaseProcessFlow]] = _ScenarioProcessFlow

    __algorithm_settings: BaseDriverSettings | None
    """The algorithm settings once they have been specified."""

    def __init__(
        self,
        disciplines: Sequence[BaseDiscipline],
        design_space: DesignSpace,
        name: str = "",
        formulation_settings: BaseFormulationSettings | None = None,
    ) -> None:
        """
        Args:
            disciplines: The disciplines.
            design_space: The input space on which to evaluate the disciplines.
            name: The name to be given to the scenario.
                If empty, use the name of the class.
            formulation_settings: The MDO formulation settings
                to generate the multidisciplinary evaluation process.
                If `None`,
                use [MDF_Settings][gemseo.formulations.mdf_settings.MDF_Settings].
        """  # noqa: D205, D212
        super().__init__(name)
        self._algo_factory = self._ALGO_FACTORY_CLASS(use_cache=True)
        self.clear_database_before_execute = False
        self._execution_result = None
        self._backup_evaluations = False
        self._backup_file_path = Path()
        self.__algorithm_settings = None
        if formulation_settings is None:
            formulation_settings = MDF_Settings()

        evaluation_problem = self._evaluation_problem_class(design_space)
        self.formulation = self._formulation_factory.create(
            formulation_settings.target_class_name,
            evaluation_problem,
            disciplines,
            settings=formulation_settings,
        )
        evaluation_problem.database.name = self.name
        for constraint in self.formulation.extra_constraint_functions:
            self._add_extra_constraint(constraint)

    def _add_extra_constraint(self, constraint: ArrayFunction) -> None:
        """Add an extra constraint as observable.

        An extra constraint is a constraint that is automatically
        built by the formulation (e.g. consistency constraints in IDF).
        In case of an `EvaluationScenario`,
        these extra constraints are transformed to observables.

        Args:
            constraint: The extra constraint.
        """
        self.formulation.problem.add_observable(constraint)

    @property
    def disciplines(self) -> tuple[BaseDiscipline, ...]:
        """The disciplines."""
        return self.formulation.disciplines

    @property
    def design_space(self) -> DesignSpace:
        """The design space."""
        return self.formulation.problem.design_space

    def add_observable(
        self,
        output_names: str | Iterable[str],
        observable_name: str = "",
        discipline: BaseDiscipline | None = None,
    ) -> None:
        """Add output variables to be observed.

        Args:
            output_names: The names of the outputs.
                If multiple names are passed,
                the observable will be a vector
                and a single top-level discipline must provide all outputs.
            observable_name: The name of the observable to be stored.
            discipline: The discipline used to build the observable function.
                If `None`, detect the discipline from the top-level disciplines.
        """  # noqa: D205, D212
        output_names = convert_strings_to_iterable(output_names)

        self.formulation.add_observable(output_names, observable_name, discipline)

    @property
    def formulation_name(self) -> str:
        """The name of the MDO formulation."""
        return self.formulation.__class__.__name__

    def set_differentiation_method(
        self,
        method: EvaluationProblem.DifferentiationMethod = EvaluationProblem.DifferentiationMethod.USER_GRAD,  # noqa: E501
        step: float = 1e-6,
        cast_default_inputs_to_complex: bool = False,
    ) -> None:
        """Set the differentiation method for computing the Jacobian.

        When the selected method to differentiate the process is `complex_step`,
        the [DesignSpace][gemseo.algos.design_space.DesignSpace] current value
        will be cast to `complex128`;
        additionally, if the option `cast_default_inputs_to_complex` is `True`,
        the default inputs of the scenario's disciplines will be cast as well provided
        that they are `ndarray` with `dtype` `float64`.

        Args:
            method: The differentiation method.
            step: The finite difference step.
            cast_default_inputs_to_complex: Whether to cast all float default inputs
                of the scenario's disciplines if the selected method is
                `"complex_step"`.
        """
        if method == self.DifferentiationMethod.COMPLEX_STEP:
            self.formulation.problem.design_space.to_complex()
            if cast_default_inputs_to_complex:
                self.__cast_default_inputs_to_complex()

        self.formulation.problem.differentiation_method = method
        self.formulation.problem.differentiation_step = step

    def __cast_default_inputs_to_complex(self) -> None:
        """Cast the float default inputs of all disciplines to complex."""
        for discipline in get_sub_disciplines(
            self.formulation.disciplines, recursive=True
        ):
            defaults = discipline.io.input_grammar.defaults
            for key, value in defaults.items():
                if isinstance(value, ndarray) and value.dtype == float64:
                    defaults[key] = array(value, dtype=complex128)

    def to_dataset(
        self,
        name: str = "",
        categorize: bool = True,
        export_gradients: bool = False,
    ) -> Dataset:
        """Export the database of the evaluation problem to a dataset.

        Args:
            name: The name to be given to the dataset.
                If empty, use the name of the database.
            categorize: Whether to distinguish
                between the different groups of variables.
                Otherwise, put all the variables in the same group.
            export_gradients: Whether to export the gradients of the functions
                if the latter are available in the database.

        Returns:
            The dataset.
        """
        return self.formulation.problem.to_dataset(
            name=name,
            categorize=categorize,
            export_gradients=export_gradients,
            input_values=self._get_input_values(),
        )

    def _get_input_values(self) -> Iterable[RealArray]:
        """Return the input values for exporting the database into a dataset.

        Returns:
            The input values.
        """
        if isinstance(self.__algorithm_settings, BaseDOESettings):
            # The algo is not instantiated again since it is in the factory cache.
            algo = self._algo_factory.create(
                self.__algorithm_settings.target_class_name
            )
            algo: BaseDOELibrary
            return algo.samples

        return ()

    def set_algorithm(self, algorithm_settings: BaseDriverSettings) -> None:
        """Define the algorithm settings.

        Args:
            algorithm_settings: The algorithm settings
        """
        self.__algorithm_settings = algorithm_settings

    def __get_execution_metrics(self) -> MultiLineString:
        """Return the string representation of the execution metrics of the scenario.

        Returns:
            The string representation of the execution metrics of the scenario.
        """
        total_n_linearizations = 0
        total_n_executions = 0
        mls = MultiLineString()
        mls.add("Scenario execution statistics")
        mls.indent()
        for discipline in self.formulation.disciplines:
            statistics = discipline.execution_statistics
            n_executions = statistics.n_executions
            n_linearizations = statistics.n_linearizations
            total_n_executions += n_executions
            total_n_linearizations += n_linearizations
            mls.add("Discipline: {}", discipline.name)
            mls.indent()
            mls.add("Executions number: {}", n_executions)
            mls.add("Execution time: {} s", statistics.duration)
            mls.add("Linearizations number: {}", statistics.n_linearizations)
            mls.dedent()

        mls.add("Total number of executions calls: {}", total_n_executions)
        mls.add("Total number of linearizations: {}", total_n_linearizations)
        return mls

    def print_execution_metrics(self) -> None:
        """Print the total number of executions and cumulated runtime by discipline."""
        if ExecutionStatistics.is_enabled:
            LOGGER.info("%s", self.__get_execution_metrics())
        else:
            LOGGER.info("The discipline counters are disabled.")

    def execute(
        self,
        algorithm_settings: BaseDriverSettings | None = None,
    ) -> Any:
        """Apply an algorithm to the scenario.

        Args:
            algorithm_settings: The algorithm settings.
                If `None`,
                the method will use the settings
                defined by the `algorithm_settings` attribute.

        Returns:
            The result of the algorithm's execution, if there is one.

        Raises:
            ValueError: If the algorithm settings are not defined.
        """
        LOGGER.info("*** Start %s execution ***", self.name)
        LOGGER.info("%r", self)
        initial_duration = self.execution_statistics.duration

        if algorithm_settings is not None:
            self.set_algorithm(algorithm_settings)

        if self.__algorithm_settings is None:
            msg = (
                "Algorithm settings are necessary for executing a scenario. "
                "Pass the settings in the execute method "
                "or use the set_algorithm method."
            )
            raise ValueError(msg)

        # DOE algorithms do not normalize the input data
        # but if an optimization algorithm was used in the previous execution,
        # the functions attached to the OptimizationProblem
        # expect normalized input data.
        # So the original functions must be used.
        # As it is possible that other types of driver do the same as optimizers,
        # the original functions are restored each time a DOE is used.
        if isinstance(self.__algorithm_settings, BaseDOESettings):
            self.formulation.problem.reset(
                database=False,
                current_iter=False,
                design_space=False,
                function_calls=False,
            )

        if self.clear_database_before_execute:
            # Clear the database when multiple runs are performed,
            # see MDOScenarioAdapter.
            self.formulation.problem.database.clear()

        n_x = len(self.formulation.problem.database)

        self._execute_monitored()

        # The last call to the functions may not trigger the callback
        # so some values may be missing in the database.
        # This ensures that the callback is called after the last iteration.
        if self._backup_evaluations:
            database = self.formulation.problem.database
            n_x_a = len(database)
            if 0 < n_x < n_x_a:
                x_vect = database.get_x_vect(n_x_a)
                self._execute_backup_callback(x_vect)

        execution_statistics = self.execution_statistics
        if execution_statistics.is_enabled:
            time_ = timedelta(seconds=execution_statistics.duration - initial_duration)
            LOGGER.info("*** End %s execution (time: %s) ***", self.name, time_)
        else:
            LOGGER.info("*** End %s execution ***", self.name)

        return self._execution_result

    def _execute(self) -> None:
        self._execution_result = self._algo_factory.execute(
            self.formulation.problem, settings=self.__algorithm_settings
        )

    def to_ggobi(self, file_path: str | Path) -> None:
        """Export the database to an XML file for ggobi tool.

        Args:
            file_path: The XML file path.
        """
        self.formulation.problem.database.to_ggobi(file_path=file_path)

    def to_hdf(self, file_path: str | Path, append: bool = False) -> None:
        """Export the evaluations and results to an HDF file.

        Args:
            file_path: The HDF file path.
            append: Whether to append the evaluations to the file if not empty.
        """
        self.formulation.problem.to_hdf(file_path=file_path, append=append)

    def set_backup_settings(
        self,
        file_path: str | Path,
        at_each_iteration: bool = False,
        at_each_function_call: bool = True,
        erase: bool = False,
        load: bool = False,
    ) -> None:
        """Set the backup file to store the evaluations of the functions during the run.

        Args:
            file_path: The backup file path.
            at_each_iteration: Whether the backup file is updated
                at every iteration of the driver.
            at_each_function_call: Whether the backup file is updated
                at every function call.
            erase: Whether the backup file is erased before the run.
            load: Whether the backup file is loaded before run,
                useful after a crash.

        Raises:
            ValueError: If both `erase` and `pre_load` are `True`.
        """
        problem = self.formulation.problem
        self._backup_evaluations = True
        self._backup_file_path = Path(file_path)

        if self._backup_file_path.exists():
            if erase and load:
                msg = (
                    "Conflicting options for evaluation backup, "
                    "cannot pre-load and erase the backup file at the same time."
                )
                raise ValueError(msg)
            if erase:
                LOGGER.warning(
                    "Erasing evaluation backup in %s",
                    self._backup_file_path,
                )
                self._backup_file_path.unlink()
            elif load:
                problem.database.update_from_hdf(self._backup_file_path)
                max_iteration = len(problem.database)
                if max_iteration != 0:
                    problem.evaluation_counter.current = max_iteration

        problem.add_listener(
            self._execute_backup_callback,
            at_each_iteration=at_each_iteration,
            at_each_function_call=at_each_function_call,
        )

    def _execute_backup_callback(self, x_vect: ndarray) -> None:
        """A callback function to back up the evaluations.

        Args:
            x_vect: The input value.
        """
        self.to_hdf(self._backup_file_path, append=True)

    def _get_string_representation(self) -> MultiLineString:
        mls = MultiLineString()
        mls.add(self.name)
        mls.indent()
        mls.add(
            "Disciplines: {}", pretty_str(self.formulation.disciplines, delimiter=" ")
        )
        mls.add("MDO formulation: {}", self.formulation.__class__.__name__)
        return mls

    def xdsmize(
        self,
        monitor: bool = False,
        directory_path: str | Path = ".",
        log_workflow_status: bool = False,
        file_name: str = "xdsm",
        show_html: bool = False,
        save_html: bool = True,
        save_json: bool = False,
        save_pdf: bool = False,
        pdf_build: bool = True,
        pdf_cleanup: bool = True,
        pdf_batchmode: bool = True,
    ) -> XDSM | None:
        """Create a XDSM diagram of the scenario.

        Args:
            monitor: Whether to update the generated file
                at each discipline status change.
            directory_path: The path of the directory to save the files.
            log_workflow_status: Whether to log the evolution of the workflow's status.
            file_name: The file name without the file extension.
            show_html: Whether to open the web browser and display the XDSM.
            save_html: Whether to save the XDSM as a HTML file.
            save_json: Whether to save the XDSM as a JSON file.
            save_pdf: Whether to save the XDSM as
                a TikZ file `"{file_name}.tikz"` containing its definition and
                a LaTeX file `"{file_name}.tex"` including this TikZ file.
                The LaTeX file can be compiled to a PDF file.
            pdf_build: Whether to compile the LaTeX file `"{file_name}.tex"`
                to a PDF file using pdflatex.
            pdf_cleanup: Whether to clean up the pdflatex built files after compilation.
            pdf_batchmode: Whether to suppress compilation logs.

        Returns:
            A view of the XDSM if `monitor` is `False`.
        """
        from gemseo.utils.xdsm.xdsmizer import XDSMizer

        if log_workflow_status:
            monitor = True

        xdsmizer = XDSMizer(self)
        if monitor:
            xdsmizer.monitor(
                directory_path=directory_path, log_workflow_status=log_workflow_status
            )
            return None

        return xdsmizer.run(
            directory_path=directory_path,
            save_pdf=save_pdf,
            show_html=show_html,
            save_html=save_html,
            save_json=save_json,
            file_name=file_name,
            pdf_build=pdf_build,
            pdf_cleanup=pdf_cleanup,
            pdf_batchmode=pdf_batchmode,
        )

    def observe_all_outputs(self) -> None:
        """Add all the outputs of the disciplines as observables."""
        all_output_names = get_all_outputs(self.disciplines)
        function_names = self.formulation.problem.function_names
        new_output_names = set(all_output_names).difference(function_names)
        for output_name in sorted(new_output_names):
            self.add_observable(output_name)
