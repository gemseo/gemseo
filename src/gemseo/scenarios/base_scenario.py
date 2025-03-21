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
from collections.abc import Mapping
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Union

from numpy import array
from numpy import complex128
from numpy import float64
from numpy import ndarray
from pydantic import BaseModel
from pydantic import Field
from strenum import StrEnum

from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core._base_monitored_process import BaseMonitoredProcess
from gemseo.core._process_flow.base_process_flow import BaseProcessFlow
from gemseo.core._process_flow.execution_sequences.loop import LoopExecSequence
from gemseo.core._process_flow.execution_sequences.parallel import ParallelExecSequence
from gemseo.core._process_flow.execution_sequences.sequential import (
    SequentialExecSequence,
)
from gemseo.core.execution_statistics import ExecutionStatistics
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.formulations.factory import MDOFormulationFactory
from gemseo.scenarios.scenario_results.factory import ScenarioResultFactory
from gemseo.scenarios.scenario_results.scenario_result import ScenarioResult
from gemseo.utils.discipline import get_sub_disciplines
from gemseo.utils.pydantic import get_class_name
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from gemseo.algos.base_driver_settings import BaseDriverSettings
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.driver_library import DriverLibraryFactory
    from gemseo.algos.optimization_result import OptimizationResult
    from gemseo.core.discipline import Discipline
    from gemseo.core.discipline.base_discipline import BaseDiscipline
    from gemseo.datasets.dataset import Dataset
    from gemseo.formulations.base_formulation_settings import BaseFormulationSettings
    from gemseo.formulations.base_mdo_formulation import BaseMDOFormulation
    from gemseo.post.base_post import BasePost
    from gemseo.post.base_post_settings import BasePostSettings
    from gemseo.post.factory import PostFactory
    from gemseo.utils.xdsm import XDSM

LOGGER = logging.getLogger(__name__)

ScenarioInputDataType = Mapping[str, Union[str, int, Mapping[str, Union[int, float]]]]


class _ScenarioProcessFlow(BaseProcessFlow):
    """The process data and execution flow."""

    def get_data_flow(  # noqa:D102
        self,
    ) -> list[tuple[Discipline, Discipline, list[str]]]:
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

    def get_disciplines_in_data_flow(self) -> list[Discipline]:
        return [self._node]


class BaseScenario(BaseMonitoredProcess):
    """Base class for the scenarios.

    The instantiation of a :class:`.Scenario` creates an :class:`.OptimizationProblem`,
    by linking :class:`.Discipline` objects with an :class:`.BaseMDOFormulation` and
    defining both the objective to minimize or maximize and the :class:`.DesignSpace` on
    which to solve the problem. Constraints can also be added to the
    :class:`.OptimizationProblem` with the :meth:`.Scenario.add_constraint` method, as
    well as observables with the :meth:`.Scenario.add_observable` method.

    Then, the :meth:`.Scenario.execute` method takes a driver (see
    :class:`.BaseDriverLibrary`) with options as input data and uses it to solve the
    optimization problem. This driver is in charge of executing the multidisciplinary
    process.

    To view the results, use the :meth:`.Scenario.post_process` method after execution
    with one of the available post-processors that can be listed by
    :attr:`.Scenario.posts`.
    """

    class _BaseSettings(BaseModel):
        """Scenario base settings passed to :meth:`.execute`.

        This class can be derived in Scenario's derived classes to add fields.
        At import time, this class is derived a final time to override the `algo` field
        which possible values depends on the :class:`._ALGO_FACTORY`.
        The final class is assigned to :attr:`.Settings`.
        """

        algo_name: str = Field(..., description="The name of the algorithm.")

        algo_settings: dict[str, Any] = Field(
            default_factory=dict, description="The settings for the algorithm."
        )

    _algo_enum: ClassVar[type[StrEnum]]
    """The possible algorithm class names, this attribute is solely necessary for
    pickling."""

    _ALGO_FACTORY_CLASS: ClassVar[type[DriverLibraryFactory]]
    """The driver factory."""

    Settings: ClassVar[type[_BaseSettings]] = _BaseSettings
    """The class used to validate the arguments of :meth:`.execute`."""

    _settings: Settings | None
    """The algorithm name and settings (``None`` before execution)."""

    _process_flow_class: ClassVar[type[BaseProcessFlow]] = _ScenarioProcessFlow

    clear_history_before_execute: bool
    """Whether to clear the history before execute."""

    formulation: BaseMDOFormulation
    """The MDO formulation."""

    formulation_name: str
    """The name of the MDO formulation."""

    optimization_result: OptimizationResult | None
    """The optimization result if the scenario has been executed; otherwise ``None``."""

    post_factory: PostFactory | None
    """The factory for post-processors if any."""

    DifferentiationMethod = OptimizationProblem.DifferentiationMethod

    _opt_hist_backup_path: Path

    __history_backup_is_set: bool
    """Whether the history backup database option is set."""

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        objective_name: str | Sequence[str],
        design_space: DesignSpace,
        name: str = "",
        maximize_objective: bool = False,
        formulation_settings_model: BaseFormulationSettings | None = None,
        **formulation_settings: Any,
    ) -> None:
        """
        Args:
            disciplines: The disciplines
                used to compute the objective, constraints and observables
                from the design variables.
            objective_name: The name(s) of the discipline output(s) used as objective.
                If multiple names are passed, the objective will be a vector.
            design_space: The search space including at least the design variables
                (some formulations requires additional variables,
                e.g. :class:`.IDF` with the coupling variables).
            name: The name to be given to this scenario.
                If empty, use the name of the class.
            maximize_objective: Whether to maximize the objective.
            formulation_settings_model: The formulation settings as a Pydantic model.
                If ``None``, use ``**settings``.
            **formulation_settings: The formulation settings,
                including the formulation name (use the keyword ``"formulation_name"``).
                These arguments are ignored when ``settings_model`` is not ``None``.
        """  # noqa: D205, D212, D415
        super().__init__(name)
        self._form_factory = self._formulation_factory
        self._algo_factory = self._ALGO_FACTORY_CLASS(use_cache=True)

        self.optimization_result = None
        self.clear_history_before_execute = False
        formulation_name = get_class_name(
            formulation_settings_model,
            formulation_settings,
            class_name_arg="formulation_name",
        )

        self._init_formulation(
            disciplines,
            formulation_name,
            objective_name,
            design_space,
            formulation_settings_model,
            **formulation_settings,
        )
        if maximize_objective:
            self.formulation.optimization_problem.minimize_objective = False

        self.formulation.optimization_problem.database.name = self.name
        self.clear_history_before_run = False
        self.__history_backup_is_set = False
        self._settings = None

    def set_algorithm(
        self,
        algo_settings_model: BaseDriverSettings | None = None,
        **algo_settings: Any,
    ) -> None:
        """Define the algorithm to execute the scenario.

        Args:
            algo_settings_model: The algorithm settings as a Pydantic model.
                If ``None``, use ``**settings``.
            **algo_settings: The algorithm settings,
                including the algorithm name (use the keyword ``"algo_name"``).
                These arguments are ignored when ``settings_model`` is not ``None``.
        """
        if algo_settings_model is None:
            algo_name = algo_settings.pop("algo_name", None)
            if algo_name is None:
                msg = 'The algorithm name is missing; use the argument "algo_name".'
                raise ValueError(msg)
        else:
            algo_settings = {"settings_model": algo_settings_model}
            algo_name = algo_settings_model._TARGET_CLASS_NAME

        self._settings = self.Settings(algo_name=algo_name, algo_settings=algo_settings)

    @property
    def disciplines(self) -> tuple[BaseDiscipline, ...]:
        """The disciplines."""
        return self.formulation.disciplines

    @classmethod
    def __init_subclass__(cls) -> None:
        """Initialize the attributes :attr:`_algo_enum` and :attr:`.Settings`.

        This method is necessary for pickling  :attr:`.Settings` because the
        classes used for unpickling shall be accessible with a qualified name in a
        module, which is not the case of a method's body.
        Thus, the classes created at runtime (import time actually) are modified to
        pretend that they were created in the class body.
        """
        cls._algo_enum = StrEnum(
            "algo_enum",
            names=cls._ALGO_FACTORY_CLASS().algorithms,
            module=cls.__module__,
            qualname=cls.__qualname__ + "._algo_enum",
        )

        class Settings(cls._BaseSettings):
            algo_name: cls._algo_enum = Field(
                ..., description="The name of the algorithm."
            )

        Settings.__module__ = cls.__module__
        Settings.__qualname__ = cls.__qualname__ + ".Settings"
        cls.Settings = Settings

    @property
    def use_standardized_objective(self) -> bool:
        """Whether to use the standardized objective for logging and post-processing.

        The objective is :attr:`.OptimizationProblem.objective`.
        """
        return self.formulation.optimization_problem.use_standardized_objective

    @use_standardized_objective.setter
    def use_standardized_objective(self, value: bool) -> None:
        self.formulation.optimization_problem.use_standardized_objective = value

    # TODO: API: the factory is a global object, remove this property.
    @property
    def post_factory(self) -> PostFactory:
        """The factory of post-processors."""
        return ScenarioResult.POST_FACTORY

    # TODO: API: the factory is a global object, remove this property.
    @property
    def _formulation_factory(self) -> MDOFormulationFactory:
        """The factory of MDO formulations."""
        return MDOFormulationFactory()

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

        self.formulation.optimization_problem.differentiation_method = method
        self.formulation.optimization_problem.differentiation_step = step

    def __cast_default_inputs_to_complex(self) -> None:
        """Cast the float default inputs of all disciplines to complex."""
        for discipline in get_sub_disciplines(
            self.formulation.disciplines, recursive=True
        ):
            for key, value in discipline.io.input_grammar.defaults.items():
                if isinstance(value, ndarray) and value.dtype == float64:
                    discipline.io.input_grammar.defaults[key] = array(
                        value, dtype=complex128
                    )

    def add_constraint(
        self,
        output_name: str | Sequence[str],
        constraint_type: MDOFunction.ConstraintType = MDOFunction.ConstraintType.EQ,
        constraint_name: str = "",
        value: float = 0,
        positive: bool = False,
        **kwargs,
    ) -> None:
        r"""Add an equality or inequality constraint to the optimization problem.

        An equality constraint is written as :math:`c(x)=a`,
        a positive inequality constraint is written as :math:`c(x)\geq a`
        and a negative inequality constraint is written as :math:`c(x)\leq a`.

        This constraint is in addition to those created by the formulation,
        e.g. consistency constraints in IDF.

        The strategy of repartition of the constraints is defined by the formulation.

        Args:
            output_name: The name(s) of the outputs computed by :math:`c(x)`.
                If several names are given,
                a single discipline must provide all outputs.
            constraint_type: The type of constraint.
            constraint_name: The name of the constraint to be stored.
                If empty,
                the name of the constraint is generated
                from ``output_name``, ``constraint_type``, ``value`` and ``positive``.
            value: The value :math:`a`.
            positive: Whether the inequality constraint is positive.

        Raises:
            ValueError: If the constraint type is neither 'eq' nor 'ineq'.
        """
        self.formulation.add_constraint(
            output_name,
            constraint_type=constraint_type,
            constraint_name=constraint_name,
            value=value,
            positive=positive,
            **kwargs,
        )

    def add_observable(
        self,
        output_names: Sequence[str],
        observable_name: str = "",
        discipline: Discipline | None = None,
    ) -> None:
        """Add an observable to the optimization problem.

        The repartition strategy of the observable is defined in the formulation class.
        When more than one output name is provided,
        the observable function returns a concatenated array of the output values.

        Args:
            output_names: The names of the outputs to observe.
            observable_name: The name to be given to the observable.
                If empty, the output name is used by default.
            discipline: The discipline used to build the observable function.
                If ``None``, detect the discipline from the inner disciplines.
        """
        self.formulation.add_observable(output_names, observable_name, discipline)

    def _init_formulation(
        self,
        disciplines: Sequence[Discipline],
        formulation_name: str,
        objective_name: str,
        design_space: DesignSpace,
        formulation_settings_model: BaseFormulationSettings | None,
        **formulation_settings: Any,
    ) -> None:
        """Initialize the MDO formulation.

        Args:
            disciplines: The disciplines.
            formulation_name: The name of the MDO formulation,
                also the name of a class inheriting from :class:`.BaseMDOFormulation`.
            objective_name: The name of the objective.
            design_space: The design space.
            formulation_settings_model: The formulation settings as a Pydantic model.
                If ``None``, use ``**settings``.
            **formulation_settings: The formulation settings.
                These arguments are ignored when ``settings_model`` is not ``None``.
        """
        self.formulation = self._form_factory.create(
            formulation_name,
            disciplines=disciplines,
            objective_name=objective_name,
            design_space=design_space,
            settings_model=formulation_settings_model,
            **formulation_settings,
        )
        self.formulation_name = formulation_name

    def get_optim_variable_names(self) -> list[str]:
        """A convenience function to access the optimization variables.

        Returns:
            The optimization variables of the scenario.
        """
        return self.formulation.get_optim_variable_names()

    def save_optimization_history(
        self,
        file_path: str | Path,
        file_format: OptimizationProblem.HistoryFileFormat = OptimizationProblem.HistoryFileFormat.HDF5,  # noqa: E501
        # noqa: E501
        append: bool = False,
    ) -> None:
        """Save the optimization history of the scenario to a file.

        Args:
            file_path: The path of the file to save the history.
            file_format: The format of the file.
            append: If ``True``, the history is appended to the file if not empty.
        """
        optimization_problem = self.formulation.optimization_problem
        if file_format == optimization_problem.HistoryFileFormat.HDF5:
            optimization_problem.to_hdf(file_path=file_path, append=append)
        elif file_format == optimization_problem.HistoryFileFormat.GGOBI:
            optimization_problem.database.to_ggobi(file_path=file_path)

    def set_optimization_history_backup(
        self,
        file_path: str | Path,
        at_each_iteration: bool = False,
        at_each_function_call: bool = True,
        erase: bool = False,
        load: bool = False,
        plot: bool = False,
    ) -> None:
        """Set the backup file to store the evaluations of the functions during the run.

        Args:
            file_path: The backup file path.
            at_each_iteration: Whether the backup file is updated
                at every iteration of the optimization.
            at_each_function_call: Whether the backup file is updated
                at every function call.
            erase: Whether the backup file is erased before the run.
            load: Whether the backup file is loaded before run,
                useful after a crash.
            plot: Whether to plot the optimization history view at each iteration.
                The plots will be generated only after the first two iterations.

        Raises:
            ValueError: If both ``erase`` and ``pre_load`` are ``True``.
        """
        opt_pb = self.formulation.optimization_problem
        self.__history_backup_is_set = True
        self._opt_hist_backup_path = Path(file_path)

        if self._opt_hist_backup_path.exists():
            if erase and load:
                msg = (
                    "Conflicting options for history backup, "
                    "cannot pre load optimization history and erase it!"
                )
                raise ValueError(msg)
            if erase:
                LOGGER.warning(
                    "Erasing optimization history in %s",
                    self._opt_hist_backup_path,
                )
                self._opt_hist_backup_path.unlink()
            elif load:
                opt_pb.database.update_from_hdf(self._opt_hist_backup_path)
                max_iteration = len(opt_pb.database)
                if max_iteration != 0:
                    opt_pb.evaluation_counter.current = max_iteration

        opt_pb.add_listener(
            self._execute_backup_callback,
            at_each_iteration=at_each_iteration,
            at_each_function_call=at_each_function_call,
        )

        if plot:
            opt_pb.add_listener(
                self._execute_plot_callback,
                at_each_iteration=True,
                at_each_function_call=False,
            )

    def _execute_backup_callback(self, x_vect: ndarray) -> None:
        """A callback function to back up optimization history.

        Args:
            x_vect: The input value.
        """
        self.save_optimization_history(self._opt_hist_backup_path, append=True)

    def _execute_plot_callback(self, x_vect: ndarray) -> None:
        """A callback function to plot the OptHistoryView of the current history.

        Args:
            x_vect: The input value.
        """
        if len(self.formulation.optimization_problem.database) > 2:
            self.post_process(
                post_name="OptHistoryView",
                save=True,
                show=False,
                file_path=self._opt_hist_backup_path.stem,
            )

    # TODO: use class attr.
    @property
    def posts(self) -> list[str]:
        """The available post-processors."""
        return self.post_factory.class_names

    def post_process(
        self, settings_model: BasePostSettings | None = None, **settings: Any
    ) -> BasePost:
        """Post-process the optimization history.

        Args:
            settings_model: The post-processor settings as a Pydantic model.
                If ``None``, use ``**settings``.
            **settings: The post-processor settings,
                including the algorithm name (use the keyword ``"post_name"``).
                These arguments are ignored when ``settings_model`` is not ``None``.

        Returns:
            The post-processor.
        """
        return self.post_factory.execute(
            self.formulation.optimization_problem,
            settings_model=settings_model,
            **settings,
        )

    def execute(
        self,
        algo_settings_model: BaseDriverSettings | None = None,
        **algo_settings: Any,
    ) -> None:
        """Execute a scenario.

        Args:
            algo_settings_model: The algorithm settings as a Pydantic model.
                If ``None``, use ``**settings`` if any.
                If ``None`` and no settings,
                the method will use the settings defined by :meth:`.set_algorithm`.
            **algo_settings: The algorithm settings,
                including the algorithm name (use the keyword ``"algo_name"``).
                These arguments are ignored when ``settings_model`` is not ``None``.
        """
        LOGGER.info("*** Start %s execution ***", self.name)
        LOGGER.info("%s", repr(self))
        initial_duration = self.execution_statistics.duration

        if algo_settings_model is not None or algo_settings:
            self.set_algorithm(algo_settings_model=algo_settings_model, **algo_settings)

        # DOE algorithms do not normalize the input data
        # but if an optimization algorithm was used in the previous execution,
        # the functions attached to the OptimizationProblem
        # expect normalized input data.
        # So the original functions must be used.
        # As it is possible that other types of driver do the same as optimizers,
        # the original functions are restored each time a DOE is used.
        if DOELibraryFactory().is_available(self._settings.algo_name):
            self.formulation.optimization_problem.reset(
                database=False,
                current_iter=False,
                design_space=False,
                function_calls=False,
            )

        if self.clear_history_before_execute:
            # Clear the database when multiple runs are performed,
            # see MDOScenarioAdapter.
            self.formulation.optimization_problem.database.clear()
        database = self.formulation.optimization_problem.database
        n_x = len(database)
        self._execute_monitored()
        # The last call to the functions may not trigger the callback
        # so some values may be missing in the database.
        # This ensures that the callback is called after the last iteration.
        if self.__history_backup_is_set:
            n_x_a = len(database)
            if 0 < n_x < n_x_a:
                x_vect = database.get_x_vect(n_x_a)
                self._execute_backup_callback(x_vect)

        execution_statistics = self.execution_statistics
        if execution_statistics.is_enabled:
            LOGGER.info(
                "*** End %s execution (time: %s) ***",
                self.name,
                timedelta(seconds=execution_statistics.duration - initial_duration),
            )
        else:
            LOGGER.info("*** End %s execution ***", self.name)

    def _execute(self) -> None:
        self.optimization_result = self._algo_factory.execute(
            self.formulation.optimization_problem,
            algo_name=self._settings.algo_name,
            **self._settings.algo_settings,
        )

    def _get_string_representation(self) -> str:
        msg = MultiLineString()
        msg.add(self.name)
        msg.indent()
        msg.add(
            "Disciplines: {}", pretty_str(self.formulation.disciplines, delimiter=" ")
        )
        msg.add("MDO formulation: {}", self.formulation.__class__.__name__)
        return str(msg)

    def __get_execution_metrics(self) -> MultiLineString:
        """Return the execution metrics of the scenarios."""
        n_lin = 0
        n_calls = 0
        msg = MultiLineString()
        msg.add("Scenario Execution Statistics")
        msg.indent()
        for disc in self.formulation.disciplines:
            msg.add("Discipline: {}", disc.name)
            msg.indent()
            msg.add("Executions number: {}", disc.execution_statistics.n_executions)
            msg.add("Execution time: {} s", disc.execution_statistics.duration)
            msg.add(
                "Linearizations number: {}", disc.execution_statistics.n_linearizations
            )
            msg.dedent()

            n_calls += disc.execution_statistics.n_executions
            n_lin += disc.execution_statistics.n_linearizations

        msg.add("Total number of executions calls: {}", n_calls)
        msg.add("Total number of linearizations: {}", n_lin)
        return msg

    def print_execution_metrics(self) -> None:
        """Print the total number of executions and cumulated runtime by discipline."""
        if ExecutionStatistics.is_enabled:
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
        pdf_build: bool = True,
        pdf_cleanup: bool = True,
        pdf_batchmode: bool = True,
    ) -> XDSM | None:
        """Create a XDSM diagram of the scenario.

        Args:
            monitor: Whether to update the generated file
                at each discipline status change.
            log_workflow_status: Whether to log the evolution of the workflow's status.
            directory_path: The path of the directory to save the files.
            file_name: The file name without the file extension.
            show_html: Whether to open the web browser and display the XDSM.
            save_html: Whether to save the XDSM as a HTML file.
            save_json: Whether to save the XDSM as a JSON file.
            save_pdf: Whether to save the XDSM as a PDF file.
            pdf_build: Whether the standalone pdf of the XDSM will be built.
            pdf_cleanup: Whether pdflatex built files will be cleaned up
                after build is complete.
            pdf_batchmode: Whether pdflatex is run in `batchmode`.

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
            pdf_build=pdf_build,
            pdf_cleanup=pdf_cleanup,
            pdf_batchmode=pdf_batchmode,
        )

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
        return self.formulation.optimization_problem.to_dataset(
            name=name,
            categorize=categorize,
            opt_naming=opt_naming,
            export_gradients=export_gradients,
        )

    def get_result(self, name: str = "", **options: Any) -> ScenarioResult | None:
        """Return the result of the scenario execution.

        Args:
            name: The class name of the :class:`.ScenarioResult`.
                If empty, use a default one (see :func:`create_scenario_result`).
            **options: The options of the :class:`.ScenarioResult`.

        Returns:
            The result of the scenario execution.
        """
        if self.optimization_result is None:
            return None

        return ScenarioResultFactory().create(
            name or self.formulation.DEFAULT_SCENARIO_RESULT_CLASS_NAME,
            scenario=self,
            **options,
        )
