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
from datetime import timedelta
from os import remove
from os.path import basename
from os.path import exists
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence
from typing import Union

from numpy import array
from numpy import complex128
from numpy import float64
from numpy import ndarray

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.dataset import Dataset
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import ExecutionSequenceFactory
from gemseo.core.execution_sequence import LoopExecSequence
from gemseo.core.formulation import MDOFormulation
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.formulations.formulations_factory import MDOFormulationsFactory
from gemseo.post.opt_post_processor import OptPostProcessor
from gemseo.post.opt_post_processor import OptPostProcessorOptionType
from gemseo.post.post_factory import PostFactory
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

LOGGER = logging.getLogger(__name__)

ScenarioInputDataType = Mapping[str, Union[str, int, Mapping[str, Union[int, float]]]]


class Scenario(MDODiscipline):
    """Base class for the scenarios.

    The instantiation of a :class:`.Scenario`
    creates an :class:`.OptimizationProblem`,
    by linking :class:`.MDODiscipline` objects with an :class:`.MDOFormulation`
    and defining both the objective to minimize or maximize
    and the :class:`.DesignSpace` on which to solve the problem.
    Constraints can also be added to the :class:`.OptimizationProblem`
    with the :meth:`.Scenario.add_constraint` method,
    as well as observables with the :meth:`.Scenario.add_observable` method.

    Then,
    the :meth:`.Scenario.execute` method takes
    a driver (see :class:`.DriverLib`) with options as input data
    and uses it to solve the optimization problem.
    This driver is in charge of executing the multidisciplinary process.

    To view the results,
    use the :meth:`.Scenario.post_process` method after execution
    with one of the available post-processors
    that can be listed by :attr:`.Scenario.posts`.
    """

    formulation: MDOFormulation
    """The MDO formulation."""

    formulation_name: str
    """The name of the MDO formulation."""

    optimization_result: OptimizationResult
    """The optimization result."""

    post_factory: PostFactory | None
    """The factory for post-processors if any."""

    # Constants for input variables in json schema
    X_0 = "x_0"
    U_BOUNDS = "u_bounds"
    L_BOUNDS = "l_bounds"
    ALGO = "algo"
    ALGO_OPTIONS = "algo_options"
    activate_input_data_check = True
    activate_output_data_check = True

    _ATTR_TO_SERIALIZE = MDODiscipline._ATTR_TO_SERIALIZE + (
        "_algo_name",
        "_algo_factory",
        "clear_history_before_run",
        "formulation",
    )

    def __init__(
        self,
        disciplines: list[MDODiscipline],
        formulation: str,
        objective_name: str | Sequence[str],
        design_space: DesignSpace,
        name: str | None = None,
        grammar_type: str = MDODiscipline.JSON_GRAMMAR_TYPE,
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
            grammar_type: The type of grammar to declare the input and output variables
                either :attr:`~.MDODiscipline.JSON_GRAMMAR_TYPE`
                or :attr:`~.MDODiscipline.SIMPLE_GRAMMAR_TYPE`.
            maximize_objective: Whether to maximize the objective.
            **formulation_options: The options of the :class:`.MDOFormulation`.
        """  # noqa: D205, D212, D415
        self.formulation = None
        self.formulation_name = None
        self.optimization_result = None
        self._algo_factory = None
        self._opt_hist_backup_path = None
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
        self.__post_factory = None
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
    def post_factory(self) -> PostFactory | None:
        """The factory of post-processors."""
        if self.__post_factory is None:
            self.__post_factory = PostFactory()

        return self.__post_factory

    @property
    def _formulation_factory(self) -> MDOFormulationsFactory:
        """The factory of MDO formulations."""
        return MDOFormulationsFactory()

    def _check_disciplines(self) -> None:
        """Check that two disciplines do not compute the same output.

        Raises:
            ValueError: If two disciplines compute the same output.
        """
        all_outs = set()
        for disc in self.disciplines:
            outs = set(disc.get_output_data_names())
            common = outs & all_outs
            if len(common) > 0:
                raise ValueError(
                    f"Two disciplines, among which {disc.name}, "
                    f"compute the same output: {common}"
                )
            all_outs |= outs

    @property
    def design_space(self) -> DesignSpace:
        """The design space on which the scenario is performed."""
        return self.formulation.design_space

    def set_differentiation_method(
        self,
        method: str | None = "user",
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
            method: The method to use to differentiate the process,
                either ``"user"``, ``"finite_differences"``, ``"complex_step"`` or
                ``"no_derivatives"``, which is equivalent to ``None``.
            step: The finite difference step.
            cast_default_inputs_to_complex: Whether to cast all float default inputs
                of the scenario's disciplines if the selected method is
                ``"complex_step"``.
        """
        if method is None:
            method = OptimizationProblem.NO_DERIVATIVES

        elif method == OptimizationProblem.COMPLEX_STEP:
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
        constraint_type: str = MDOFunction.TYPE_EQ,
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
            constraint_type: The type of constraint,
                `"eq"` for equality constraint and
                `"ineq"` for inequality constraint.
            constraint_name: The name of the constraint to be stored.
                If ``None``, the name of the constraint is generated from the output name.
            value: The value for which the constraint is active.
                If ``None``, this value is 0.
            positive: If ``True``, the inequality constraint is positive.

        Raises:
            ValueError: If the constraint type is neither 'eq' or 'ineq'.
        """
        if constraint_type not in [MDOFunction.TYPE_EQ, MDOFunction.TYPE_INEQ]:
            raise ValueError(
                "Constraint type must be either 'eq' or 'ineq'; "
                "got '{}' instead.".format(constraint_type)
            )

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

    def get_optim_variables_names(self) -> list[str]:
        """A convenience function to access the optimization variables.

        Returns:
            The optimization variables of the scenario.
        """
        return self.formulation.get_optim_variables_names()

    def get_optimum(self) -> OptimizationResult | None:
        """Return the optimization results.

        Returns:
            The optimal solution found by the scenario if executed,
            ``None`` otherwise.
        """
        return self.optimization_result

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
            opt_pb.export_hdf(file_path=file_path, append=append)
        elif file_format == OptimizationProblem.GGOBI_FORMAT:
            opt_pb.database.export_to_ggobi(file_path=file_path)
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
        self._opt_hist_backup_path = file_path
        self._gen_opt_backup_plot = generate_opt_plot

        if exists(self._opt_hist_backup_path):
            if erase and pre_load:
                raise ValueError(
                    "Conflicting options for history backup, "
                    "cannot pre load optimization history and erase it!"
                )
            if erase:
                LOGGER.warning(
                    "Erasing optimization history in %s",
                    str(self._opt_hist_backup_path),
                )
                remove(self._opt_hist_backup_path)
            elif pre_load:
                opt_pb.database.import_hdf(self._opt_hist_backup_path)

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
            basepath = basename(self._opt_hist_backup_path).split(".")[0]
            self.post_process(
                "OptHistoryView", save=True, show=False, file_path=basepath
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
        raise NotImplementedError()

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
        outdir: str | None = ".",
        print_statuses: bool = False,
        outfilename: str = "xdsm.html",
        latex_output: bool = False,
        open_browser: bool = False,
        html_output: bool = True,
        json_output: bool = False,
    ) -> None:
        """Create a JSON file defining the XDSM related to the current scenario.

        Args:
            monitor: If ``True``, update the generated file
                at each discipline status change.
            outdir: The directory where the JSON file is generated.
                If ``None``, the current working directory is used.
            print_statuses: If ``True``, print the statuses in the console at each update.
            outfilename: The name of the file of the output.
                The basename is used and the extension is adapted
                for the HTML / JSON / PDF outputs.
            latex_output: If ``True``, build TEX, TIKZ and PDF files.
            open_browser: If ``True``, open the web browser and display the XDSM.
            html_output: If ``True``, output a self-contained HTML file.
            json_output: If ``True``, output a JSON file for XDSMjs.
        """
        from gemseo.utils.xdsmizer import XDSMizer

        if print_statuses:
            monitor = True

        if monitor:
            XDSMizer(self).monitor(outdir=outdir, print_statuses=print_statuses)
        else:
            XDSMizer(self).run(
                output_directory_path=outdir,
                latex_output=latex_output,
                open_browser=open_browser,
                html_output=html_output,
                json_output=json_output,
                outfilename=outfilename,
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
        raise NotImplementedError()

    def get_available_driver_names(self) -> list[str]:
        """The available drivers."""
        return self._algo_factory.algorithms

    def _update_input_grammar(self) -> None:
        """Update the input grammar from the names of available drivers."""
        if self.grammar_type == MDODiscipline.JSON_GRAMMAR_TYPE:
            self.input_grammar.update(
                {"algo": {"type": "string", "enum": self.get_available_driver_names()}}
            )
        else:
            self._update_grammar_input()

    def _update_grammar_input(self) -> None:
        """Update the inputs of a Grammar."""
        raise NotImplementedError()

    @staticmethod
    def is_scenario() -> bool:
        """Indicate if the current object is a :class:`.Scenario`.

        Returns:
            ``True`` if the current object is a :class:`.Scenario`.
        """
        return True

    def export_to_dataset(
        self,
        name: str | None = None,
        by_group: bool = True,
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
                If ``None``, use the name of the :attr:`.OptimizationProblem.database`.
            by_group: Whether to store the data by group in :attr:`.Dataset.data`,
                in the sense of one unique NumPy array per group.
                If ``categorize`` is ``False``,
                there is a unique group: :attr:`.Dataset.PARAMETER_GROUP``.
                If ``categorize`` is ``True``,
                the groups can be either
                :attr:`.Dataset.DESIGN_GROUP` and :attr:`.Dataset.FUNCTION_GROUP`
                if ``opt_naming`` is ``True``,
                or :attr:`.Dataset.INPUT_GROUP` and :attr:`.Dataset.OUTPUT_GROUP`.
                If ``by_group`` is ``False``, store the data by variable names.
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
        return self.formulation.opt_problem.export_to_dataset(
            name=name,
            by_group=by_group,
            categorize=categorize,
            opt_naming=opt_naming,
            export_gradients=export_gradients,
        )
