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
"""The base class for the scenarios."""

from __future__ import annotations

import logging
import timeit
from collections.abc import Mapping
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Union

from numpy import array
from numpy import complex128
from numpy import float64
from numpy import ndarray

from gemseo import create_scenario_result
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import ExecutionSequenceFactory
from gemseo.core.execution_sequence import LoopExecSequence
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.disciplines.utils import check_disciplines_consistency
from gemseo.formulations.formulations_factory import MDOFormulationsFactory
from gemseo.scenarios.scenario_results.scenario_result import ScenarioResult
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.opt_result import OptimizationResult
    from gemseo.core.formulation import MDOFormulation
    from gemseo.datasets.dataset import Dataset
    from gemseo.post.opt_post_processor import OptPostProcessor
    from gemseo.post.opt_post_processor import OptPostProcessorOptionType
    from gemseo.post.post_factory import PostFactory
    from gemseo.utils.xdsm import XDSM


LOGGER = logging.getLogger(__name__)

ScenarioInputDataType = Mapping[str, Union[str, int, Mapping[str, Union[int, float]]]]


class Scenario(MDODiscipline):
    """Base class for the scenarios.

    The instantiation of a :class:`.Scenario` creates an :class:`.OptimizationProblem`,
    by linking :class:`.MDODiscipline` objects with an :class:`.MDOFormulation` and
    defining both the objective to minimize or maximize and the :class:`.DesignSpace` on
    which to solve the problem. Constraints can also be added to the
    :class:`.OptimizationProblem` with the :meth:`.Scenario.add_constraint` method, as
    well as observables with the :meth:`.Scenario.add_observable` method.

    Then, the :meth:`.Scenario.execute` method takes a driver (see
    :class:`.DriverLibrary`) with options as input data and uses it to solve the
    optimization problem. This driver is in charge of executing the multidisciplinary
    process.

    To view the results, use the :meth:`.Scenario.post_process` method after execution
    with one of the available post-processors that can be listed by
    :attr:`.Scenario.posts`.
    """

    formulation: MDOFormulation
    """The MDO formulation."""

    formulation_name: str
    """The name of the MDO formulation."""

    optimization_result: OptimizationResult | None
    """The optimization result if the scenario has been executed; otherwise ``None``."""

    post_factory: PostFactory | None
    """The factory for post-processors if any."""

    DifferentiationMethod = OptimizationProblem.DifferentiationMethod
    # Constants for input variables in json schema
    X_0 = "x_0"
    U_BOUNDS = "u_bounds"
    L_BOUNDS = "l_bounds"
    ALGO = "algo"
    ALGO_OPTIONS = "algo_options"
    activate_input_data_check = True
    activate_output_data_check = True
    _opt_hist_backup_path: Path

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        formulation: str,
        objective_name: str | Sequence[str],
        design_space: DesignSpace,
        name: str | None = None,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        maximize_objective: bool = False,
        **formulation_options: Any,
    ) -> None:
        """
        Args:
            disciplines: The disciplines
                used to compute the objective, constraints and observables
                from the design variables.
            formulation: The class name of the :class:`.MDOFormulation`,
                e.g. ``"MDF"``, ``"IDF"`` or ``"BiLevel"``.
            objective_name: The name(s) of the discipline output(s) used as objective.
                If multiple names are passed, the objective will be a vector.
            design_space: The search space including at least the design variables
                (some formulations requires additional variables,
                e.g. :class:`.IDF` with the coupling variables).
            name: The name to be given to this scenario.
                If ``None``, use the name of the class.
            grammar_type: The grammar for the scenario and the MDO formulation.
            maximize_objective: Whether to maximize the objective.
            **formulation_options: The options of the :class:`.MDOFormulation`.
        """  # noqa: D205, D212, D415
        self.optimization_result = None
        self._algo_factory = None
        self._gen_opt_backup_plot = False
        self._algo_name = None
        self._lib = None

        self._init_algo_factory()
        self._form_factory = self._formulation_factory
        super().__init__(
            name=name, grammar_type=grammar_type, auto_detect_grammar_files=True
        )
        self._disciplines = disciplines
        self._check_disciplines()

        self._init_formulation(
            formulation,
            objective_name,
            design_space,
            maximize_objective,
            grammar_type=grammar_type,
            **formulation_options,
        )
        self.formulation.opt_problem.database.name = self.name
        self._update_input_grammar()
        self.clear_history_before_run = False

    @property
    def use_standardized_objective(self) -> bool:
        """Whether to use the standardized objective for logging and post-processing.

        The objective is :attr:`.OptimizationProblem.objective`.
        """
        return self.formulation.opt_problem.use_standardized_objective

    @use_standardized_objective.setter
    def use_standardized_objective(self, value: bool) -> None:
        self.formulation.opt_problem.use_standardized_objective = value

    @property
    def post_factory(self) -> PostFactory:
        """The factory of post-processors."""
        return ScenarioResult.POST_FACTORY

    @property
    def _formulation_factory(self) -> MDOFormulationsFactory:
        """The factory of MDO formulations."""
        return MDOFormulationsFactory()

    def _check_disciplines(self) -> None:
        """Check that two disciplines do not compute the same output."""
        check_disciplines_consistency(self.disciplines, False, True)

    @property
    def design_space(self) -> DesignSpace:
        """The design space on which the scenario is performed."""
        return self.formulation.design_space

    def set_differentiation_method(
        self,
        method: DifferentiationMethod = DifferentiationMethod.USER_GRAD,
        step: float = 1e-6,
        cast_default_inputs_to_complex: bool = False,
    ) -> None:
        """Set the differentiation method for the process.

        When the selected method to differentiate the process is ``complex_step`` the
        :class:`.DesignSpace` current value will be cast to ``complex128``;
        additionally, if the option ``cast_default_inputs_to_complex`` is ``True``,
        the default inputs of the scenario's disciplines will be cast as well provided
        that they are ``ndarray`` with ``dtype`` ``float64``.

        Args:
            method: The method to use to differentiate the process.
            step: The finite difference step.
            cast_default_inputs_to_complex: Whether to cast all float default inputs
                of the scenario's disciplines if the selected method is
                ``"complex_step"``.
        """
        if method == self.DifferentiationMethod.COMPLEX_STEP:
            self.formulation.design_space.to_complex()
            if cast_default_inputs_to_complex:
                self.__cast_default_inputs_to_complex()

        self.formulation.opt_problem.differentiation_method = method
        self.formulation.opt_problem.fd_step = step

    def __cast_default_inputs_to_complex(self) -> None:
        """Cast the float default inputs of all disciplines to complex."""
        for discipline in self.formulation.get_sub_disciplines(recursive=True):
            for key, value in discipline.default_inputs.items():
                if isinstance(value, ndarray) and value.dtype == float64:
                    discipline.default_inputs[key] = array(value, dtype=complex128)

    def add_constraint(
        self,
        output_name: str | Sequence[str],
        constraint_type: MDOFunction.ConstraintType = MDOFunction.ConstraintType.EQ,
        constraint_name: str | None = None,
        value: float | None = None,
        positive: bool = False,
        **kwargs,
    ) -> None:
        """Add a design constraint.

        This constraint is in addition to those created by the formulation,
        e.g. consistency constraints in IDF.

        The strategy of repartition of the constraints is defined by the formulation.

        Args:
            output_name: The names of the outputs to be used as constraints.
                For instance, if `"g_1"` is given and `constraint_type="eq"`,
                `g_1=0` will be added as constraint to the optimizer.
                If several names are given,
                a single discipline must provide all outputs.
            constraint_type: The type of constraint.
            constraint_name: The name of the constraint to be stored.
                If ``None``,
                the name of the constraint is generated from the output name.
            value: The value for which the constraint is active.
                If ``None``, this value is 0.
            positive: If ``True``, the inequality constraint is positive.

        Raises:
            ValueError: If the constraint type is neither 'eq' nor 'ineq'.
        """
        self.formulation.add_constraint(
            output_name,
            constraint_type,
            constraint_name=constraint_name,
            value=value,
            positive=positive,
            **kwargs,
        )

    def add_observable(
        self,
        output_names: Sequence[str],
        observable_name: Sequence[str] | None = None,
        discipline: MDODiscipline | None = None,
    ) -> None:
        """Add an observable to the optimization problem.

        The repartition strategy of the observable is defined in the formulation class.
        When more than one output name is provided,
        the observable function returns a concatenated array of the output values.

        Args:
            output_names: The names of the outputs to observe.
            observable_name: The name to be given to the observable.
                If ``None``, the output name is used by default.
            discipline: The discipline used to build the observable function.
                If ``None``, detect the discipline from the inner disciplines.
        """
        self.formulation.add_observable(output_names, observable_name, discipline)

    def _init_formulation(
        self,
        formulation: str,
        objective_name: str,
        design_space: DesignSpace,
        maximize_objective: bool,
        **formulation_options: Any,
    ) -> None:
        """Initialize the MDO formulation.

        Args:
            formulation: The name of the MDO formulation,
                also the name of a class inheriting from :class:`.MDOFormulation`.
            objective_name: The name of the objective.
            design_space: The design space.
            maximize_objective: Whether to maximize the objective.
            **formulation_options: The options
                to be passed to the :class:`.MDOFormulation`.
        """
        if not isinstance(formulation, str):
            raise TypeError(
                "Formulation must be specified by its name; "
                "please use GEMSEO_PATH to specify custom formulations."
            )
        self.formulation = self._form_factory.create(
            formulation,
            disciplines=self.disciplines,
            objective_name=objective_name,
            design_space=design_space,
            maximize_objective=maximize_objective,
            **formulation_options,
        )
        self.formulation_name = formulation

    def get_optim_variable_names(self) -> list[str]:
        """A convenience function to access the optimization variables.

        Returns:
            The optimization variables of the scenario.
        """
        return self.formulation.get_optim_variable_names()

    def save_optimization_history(
        self,
        file_path: str | Path,
        file_format: str = OptimizationProblem.HDF5_FORMAT,
        append: bool = False,
    ) -> None:
        """Save the optimization history of the scenario to a file.

        Args:
            file_path: The path of the file to save the history.
            file_format: The format of the file, either "hdf5" or "ggobi".
            append: If ``True``, the history is appended to the file if not empty.

        Raises:
            ValueError: If the file format is not correct.
        """
        opt_pb = self.formulation.opt_problem
        if file_format == OptimizationProblem.HDF5_FORMAT:
            opt_pb.to_hdf(file_path=file_path, append=append)
        elif file_format == OptimizationProblem.GGOBI_FORMAT:
            opt_pb.database.to_ggobi(file_path=file_path)
        else:
            raise ValueError(
                f"Cannot export optimization history to file format: {file_format}."
            )

    def set_optimization_history_backup(
        self,
        file_path: str | Path,
        each_new_iter: bool = False,
        each_store: bool = True,
        erase: bool = False,
        pre_load: bool = False,
        generate_opt_plot: bool = False,
    ) -> None:
        """Set the backup file for the optimization history during the run.

        Args:
            file_path: The path to the file to save the history.
            each_new_iter: If ``True``, callback at every iteration.
            each_store: If ``True``, callback at every call to store() in the database.
            erase: If ``True``, the backup file is erased before the run.
            pre_load: If ``True``, the backup file is loaded before run,
                useful after a crash.
            generate_opt_plot: If ``True``, generate the optimization history view
                at backup.

        Raises:
            ValueError: If both erase and pre_load are ``True``.
        """
        opt_pb = self.formulation.opt_problem
        self._opt_hist_backup_path = Path(file_path)
        self._gen_opt_backup_plot = generate_opt_plot

        if self._opt_hist_backup_path.exists():
            if erase and pre_load:
                raise ValueError(
                    "Conflicting options for history backup, "
                    "cannot pre load optimization history and erase it!"
                )
            if erase:
                LOGGER.warning(
                    "Erasing optimization history in %s",
                    self._opt_hist_backup_path,
                )
                self._opt_hist_backup_path.unlink()
            elif pre_load:
                opt_pb.database.update_from_hdf(self._opt_hist_backup_path)
                max_iteration = len(opt_pb.database)
                if max_iteration != 0:
                    opt_pb.current_iter = max_iteration

        opt_pb.add_callback(
            self._execute_backup_callback,
            each_new_iter=each_new_iter,
            each_store=each_store,
        )

    def _execute_backup_callback(self, option: Any = None) -> None:
        """A callback function to back up optimization history.

        Args:
            option: Any input value which is not used within the current function,
               but need to be defined for the generic mechanism of the
               callback functions usage in :class:`.OptimizationProblem`.
        """
        self.save_optimization_history(self._opt_hist_backup_path, append=True)
        if self._gen_opt_backup_plot and self.formulation.opt_problem.database:
            self.post_process(
                "OptHistoryView",
                save=True,
                show=False,
                file_path=self._opt_hist_backup_path.stem,
            )

    @property
    def posts(self) -> list[str]:
        """The available post-processors."""
        return self.post_factory.posts

    def post_process(
        self,
        post_name: str,
        **options: OptPostProcessorOptionType | Path,
    ) -> OptPostProcessor:
        """Post-process the optimization history.

        Args:
            post_name: The name of the post-processor,
                i.e. the name of a class inheriting from :class:`.OptPostProcessor`.
            **options: The options for the post-processor.

        Returns:
            The post-processing instance related to the optimization scenario.
        """
        return self.post_factory.execute(
            self.formulation.opt_problem, post_name, **options
        )

    def _run(self) -> None:
        t_0 = timeit.default_timer()
        LOGGER.info(" ")
        LOGGER.info("*** Start %s execution ***", self.name)
        LOGGER.info("%s", repr(self))
        # Clear the database when multiple runs are performed, see MDOScenarioAdapter.
        if self.clear_history_before_run:
            self.formulation.opt_problem.database.clear()

        self._run_algorithm()
        LOGGER.info(
            "*** End %s execution (time: %s) ***",
            self.name,
            timedelta(seconds=timeit.default_timer() - t_0),
        )

    def _run_algorithm(self) -> OptimizationResult:
        """Run the driver algorithm."""
        raise NotImplementedError

    def __repr__(self) -> str:
        msg = MultiLineString()
        msg.add(self.name)
        msg.indent()
        msg.add("Disciplines: {}", pretty_str(self.disciplines, delimiter=" "))
        msg.add("MDO formulation: {}", self.formulation.__class__.__name__)
        return str(msg)

    def get_disciplines_statuses(self) -> dict[str, str]:
        """Retrieve the statuses of the disciplines.

        Returns:
            The statuses of the disciplines.
        """
        statuses = {}
        for disc in self.disciplines:
            statuses[disc.__class__.__name__] = disc.status
        return statuses

    def __get_execution_metrics(self) -> MultiLineString:
        """Return the execution metrics of the scenarios."""
        n_lin = 0
        n_calls = 0
        msg = MultiLineString()
        msg.add("Scenario Execution Statistics")
        msg.indent()
        for disc in self.disciplines:
            msg.add("Discipline: {}", disc.name)
            msg.indent()
            msg.add("Executions number: {}", disc.n_calls)
            msg.add("Execution time: {} s", disc.exec_time)
            msg.add("Linearizations number: {}", disc.n_calls_linearize)
            msg.dedent()

            n_calls += disc.n_calls
            n_lin += disc.n_calls_linearize

        msg.add("Total number of executions calls: {}", n_calls)
        msg.add("Total number of linearizations: {}", n_lin)
        return msg

    def print_execution_metrics(self) -> None:
        """Print the total number of executions and cumulated runtime by discipline."""
        if MDODiscipline.activate_counters:
            LOGGER.info("%s", self.__get_execution_metrics())
        else:
            LOGGER.info("The discipline counters are disabled.")

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
    ) -> XDSM | None:
        """Create a XDSM diagram of the scenario.

        Args:
            monitor: Whether to update the generated file
                at each discipline status change.
            log_workflow_status: Whether to log the evolution of the workflow's status.
            directory_path: The path of the directory to save the files.
                If ``show_html=True`` and ``output_directory_path=None``,
                the HTML file is stored in a temporary directory.
            file_name: The file name without the file extension.
            show_html: Whether to open the web browser and display the XDSM.
            save_html: Whether to save the XDSM as a HTML file.
            save_json: Whether to save the XDSM as a JSON file.
            save_pdf: Whether to save the XDSM as a PDF file.

        Returns:
            A view of the XDSM if ``monitor`` is ``False``.
        """
        from gemseo.utils.xdsmizer import XDSMizer

        if log_workflow_status:
            monitor = True

        if monitor:
            XDSMizer(self).monitor(
                directory_path=directory_path, log_workflow_status=log_workflow_status
            )
            return None

        return XDSMizer(self).run(
            directory_path=directory_path,
            save_pdf=save_pdf,
            show_html=show_html,
            save_html=save_html,
            save_json=save_json,
            file_name=file_name,
        )

    def get_expected_dataflow(  # noqa:D102
        self,
    ) -> list[tuple[MDODiscipline, MDODiscipline, list[str]]]:
        return self.formulation.get_expected_dataflow()

    def get_expected_workflow(self) -> LoopExecSequence:  # noqa:D102
        exp_wf = self.formulation.get_expected_workflow()
        return ExecutionSequenceFactory.loop(self, exp_wf)

    def _init_algo_factory(self) -> None:
        """Initialize the factory of algorithms."""
        raise NotImplementedError

    def get_available_driver_names(self) -> list[str]:
        """The available drivers."""
        return self._algo_factory.algorithms

    def _update_input_grammar(self) -> None:
        """Update the input grammar from the names of available drivers."""
        if self.grammar_type == MDODiscipline.GrammarType.JSON:
            self.input_grammar.update_from_schema({
                "properties": {
                    "algo": {
                        "type": "string",
                        "enum": self.get_available_driver_names(),
                    }
                }
            })
        else:
            self.input_grammar.update_from_types({"algo": str})
        self.input_grammar.required_names.add("algo")

    @staticmethod
    def is_scenario() -> bool:
        """Indicate if the current object is a :class:`.Scenario`.

        Returns:
            ``True`` if the current object is a :class:`.Scenario`.
        """
        return True

    def to_dataset(
        self,
        name: str = "",
        categorize: bool = True,
        opt_naming: bool = True,
        export_gradients: bool = False,
    ) -> Dataset:
        """Export the database of the optimization problem to a :class:`.Dataset`.

        The variables can be classified into groups:
        :attr:`.Dataset.DESIGN_GROUP` or :attr:`.Dataset.INPUT_GROUP`
        for the design variables
        and :attr:`.Dataset.FUNCTION_GROUP` or :attr:`.Dataset.OUTPUT_GROUP`
        for the functions
        (objective, constraints and observables).

        Args:
            name: The name to be given to the dataset.
                If empty, use the name of the :attr:`.OptimizationProblem.database`.
            categorize: Whether to distinguish
                between the different groups of variables.
                Otherwise, group all the variables in :attr:`.Dataset.PARAMETER_GROUP``.
            opt_naming: Whether to use
                :attr:`.Dataset.DESIGN_GROUP` and :attr:`.Dataset.FUNCTION_GROUP`
                as groups.
                Otherwise,
                use :attr:`.Dataset.INPUT_GROUP` and :attr:`.Dataset.OUTPUT_GROUP`.
            export_gradients: Whether to export the gradients of the functions
                (objective function, constraints and observables)
                if the latter are available in the database of the optimization problem.

        Returns:
            A dataset built from the database of the optimization problem.
        """
        return self.formulation.opt_problem.to_dataset(
            name=name,
            categorize=categorize,
            opt_naming=opt_naming,
            export_gradients=export_gradients,
        )

    def get_result(self, name: str = "", **options: Any) -> ScenarioResult:
        """Return the result of the scenario execution.

        Args:
            name: The class name of the :class:`.ScenarioResult`.
                If empty, use a default one (see :func:`create_scenario_result`).
            **options: The options of the :class:`.ScenarioResult`.

        Returns:
            The result of the scenario execution.
        """
        return create_scenario_result(self, name, **options)
