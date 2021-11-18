# -*- coding: utf-8 -*-
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

from __future__ import division, unicode_literals

import inspect
import logging
from os import remove
from os.path import basename, exists
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from six import string_types

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import ExecutionSequenceFactory, LoopExecSequence
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.formulations.formulations_factory import MDOFormulationsFactory
from gemseo.post.opt_post_processor import OptPostProcessor, OptPostProcessorOptionType
from gemseo.post.post_factory import PostFactory
from gemseo.utils.py23_compat import Path
from gemseo.utils.string_tools import MultiLineString, pretty_repr

LOGGER = logging.getLogger(__name__)

ScenarioInputDataType = Mapping[str, Union[str, int, Mapping[str, Union[int, float]]]]


class Scenario(MDODiscipline):
    """Base class for the scenarios.

    The instantiation of a :class:`Scenario`
    creates an :class:`.OptimizationProblem`,
    by linking :class:`.MDODiscipline` objects with an :class:`.MDOFormulation`
    and defining both the objective to minimize or maximize
    and the :class:`.DesignSpace` on which to solve the problem.
    Constraints can also be added to the :class:`.OptimizationProblem`
    with the :meth:`add_constraint` method,
    as well as observables with the :meth:`add_observable` method.

    Then,
    the :meth:`execute` method takes
    a driver (see :class:`.DriverLib`) with options as input data
    and uses it to solve the optimization problem.
    This driver is in charge of executing the multidisciplinary process.

    To view the results,
    use the :meth:`post_process` method after execution
    with one of the available post-processors that can be listed by :attr:`posts`.

    Attributes:
        disciplines (List(MDODiscipline)): The disciplines.
        formulation (MDOFormulation): The MDO formulation.
        formulation_name (str): The name of the MDO formulation.
        optimization_result (OptimizationResult): The optimization result.
        post_factory (PostFactory): The factory for post-processors.
    """

    # Constants for input variables in json schema
    X_0 = "x_0"
    U_BOUNDS = "u_bounds"
    L_BOUNDS = "l_bounds"
    ALGO = "algo"
    ALGO_OPTIONS = "algo_options"

    _ATTR_TO_SERIALIZE = MDODiscipline._ATTR_TO_SERIALIZE

    def __init__(
        self,
        disciplines,  # type: Sequence[MDODiscipline]
        formulation,  # type: str
        objective_name,  # type: str
        design_space,  # type: DesignSpace
        name=None,  # type: Optional[str]
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,  # type: str
        **formulation_options  # type: Any
    ):  # type: (...) -> None
        """
        Args:
            disciplines: The disciplines
                used to compute the objective, constraints and observables
                from the design variables.
            formulation: The name of the MDO formulation,
                also the name of a class inheriting from :class:`.MDOFormulation`.
            objective_name: The name of the objective.
            design_space: The design space.
            name: The name to be given to this scenario.
                If None, use the name of the class.
            grammar_type: The type of grammar to use for IO declaration
                either JSON_GRAMMAR_TYPE or SIMPLE_GRAMMAR_TYPE.
            **formulation_options: The options
                to be passed to the :class:`.MDOFormulation`.
        """
        self.formulation = None
        self.formulation_name = None
        self.disciplines = disciplines
        self.optimization_result = None
        self._algo_factory = None
        self._opt_hist_backup_path = None
        self._gen_opt_backup_plot = False

        self._check_disciplines()
        self._init_algo_factory()
        self._form_factory = self._formulation_factory
        super(Scenario, self).__init__(name=name, grammar_type=grammar_type)
        self._init_base_grammar(self.__class__.__name__)
        self._init_formulation(
            formulation,
            objective_name,
            design_space,
            grammar_type=grammar_type,
            **formulation_options
        )
        self.formulation.opt_problem.database.name = self.name
        self.post_factory = PostFactory()
        self._update_input_grammar()

    @property
    def _formulation_factory(self):  # type:(...) -> MDOFormulationsFactory
        """The factory of MDO formulations."""
        return MDOFormulationsFactory()

    def _check_disciplines(self):  # type: (...) -> None
        """Check that two disciplines do not compute the same output.

        Raises:
            ValueError: If two disciplines compute the same output.
        """

        all_outs = set()
        for disc in self.disciplines:
            outs = set(disc.get_output_data_names())
            common = outs & all_outs
            if len(common) > 0:
                msg = "Two disciplines, among which {}, compute the same output: {}"
                raise ValueError(msg.format(disc.name, common))
            all_outs = all_outs | outs

    @property
    def design_space(self):  # type: (...) -> DesignSpace
        """The design space on which the scenario is performed."""
        return self.formulation.design_space

    def _init_base_grammar(
        self,
        name,  # type: str
    ):  # type: (...) -> None
        """Initialize the base grammars from the inputs and outputs of the scenario.

        This ensures that subclasses have base scenario inputs and outputs.
        This method can be overloaded by subclasses if this is not desired.

        Args:
            name: The name of the scenario,
                used as a base name for the JSON schemas to import:
                `name_input.json` and `name_output.json`.
        """
        comp_dir = str(Path(inspect.getfile(self.__class__)).parent)
        input_grammar_file = self.auto_get_grammar_file(True, name, comp_dir)
        output_grammar_file = self.auto_get_grammar_file(False, name, comp_dir)
        self._instantiate_grammars(
            input_grammar_file, output_grammar_file, grammar_type=self.grammar_type
        )

    def set_differentiation_method(
        self,
        method="user",  # type: Optional[str]
        step=1e-6,  # type: float
    ):  # type: (...) -> None
        """Set the differentiation method for the process.

        Args:
            method: The method to use to differentiate the process,
                either "user", "finite_differences", "complex_step" or "no_derivatives",
                which is equivalent to None.
            step: The finite difference step.
        """
        if method is None:
            method = "no_derivatives"
        self.formulation.opt_problem.differentiation_method = method
        self.formulation.opt_problem.fd_step = step

    def add_constraint(
        self,
        output_name,  # type: Union[str,Sequence[str]]
        constraint_type=MDOFunction.TYPE_EQ,  # type: str
        constraint_name=None,  # type: Optional[str]
        value=None,  # type: Optional[float]
        positive=False,  # type:bool
        **kwargs
    ):  # type: (...) -> None
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
                If None, the name of the constraint is generated from the output name.
            value: The value for which the constraint is active.
                If None, this value is 0.
            positive: If True, the inequality constraint is positive.

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
            **kwargs
        )

    def add_observable(
        self,
        output_names,  # type: Sequence[str]
        observable_name=None,  # type: Optional[Sequence[str]]
        discipline=None,  # type: Optional[MDODiscipline]
    ):  # type: (...) -> None
        """Add an observable to the optimization problem.

        The repartition strategy of the observable is defined in the formulation class.
        When more than one output name is provided,
        the observable function returns a concatenated array of the output values.

        Args:
            output_names: The names of the outputs to observe.
            observable_name: The name to be given to the observable.
                If None, the output name is used by default.
            discipline: The discipline used to build the observable function.
                If None, detect the discipline from the inner disciplines.
        """
        self.formulation.add_observable(output_names, observable_name, discipline)

    def _init_formulation(
        self,
        formulation,  # type: str
        objective_name,  # type: str
        design_space,  # type: DesignSpace
        **formulation_options  # type: Any
    ):  # type: (...) -> None
        """Initialize the MDO formulation.

        Args:
            formulation: The name of the MDO formulation,
                also the name of a class inheriting from :class:`.MDOFormulation`.
            objective_name: The name of the objective.
            design_space: The design space.
            **formulation_options: The options
                to be passed to the :class:`.MDOFormulation`.
        """
        if not isinstance(formulation, string_types):
            raise TypeError(
                "Formulation must be specified by its name; "
                "please use GEMSEO_PATH to specify custom formulations."
            )
        form_inst = self._form_factory.create(
            formulation,
            disciplines=self.disciplines,
            objective_name=objective_name,
            design_space=design_space,
            **formulation_options
        )
        self.formulation_name = formulation
        self.formulation = form_inst

    def get_optim_variables_names(self):  # type: (...) -> List[str]
        """A convenience function to access the optimization variables.

        Returns:
            The optimization variables of the scenario.
        """
        return self.formulation.get_optim_variables_names()

    def get_optimum(self):  # type: (...) -> Optional[OptimizationResult]
        """Return the optimization results.

        Returns:
            The optimal solution found by the scenario if executed,
            None otherwise.
        """
        return self.optimization_result

    def save_optimization_history(
        self,
        file_path,  # type: str
        file_format=OptimizationProblem.HDF5_FORMAT,  # type: str
        append=False,  # type: bool
    ):  # type: (...) -> None
        """Save the optimization history of the scenario to a file.

        Args:
            file_path: The path to the file to save the history.
            file_format: The format of the file, either "hdf5" or "ggobi".
            append: If True, the history is appended to the file if not empty.

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
                "Cannot export optimization history "
                "to file format: {}.".format(file_format)
            )

    def set_optimization_history_backup(
        self,
        file_path,  # type: str
        each_new_iter=False,  # type:bool
        each_store=True,  # type:bool
        erase=False,  # type:bool
        pre_load=False,  # type:bool
        generate_opt_plot=False,  # type:bool
    ):  # type: (...) -> None
        """Set the backup file for the optimization history during the run.

        Args:
            file_path: The path to the file to save the history.
            each_new_iter: If True, callback at every iteration.
            each_store: If True, callback at every call to store() in the database.
            erase: If True, the backup file is erased before the run.
            pre_load: If True, the backup file is loaded before run,
                useful after a crash.
            generate_opt_plot: If True, generate the optimization history view
                at backup.

        Raises:
            ValueError: If both erase and pre_load are True.
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

    def _execute_backup_callback(self):  # type: (...) -> None
        """A callback function to backup optimization history."""
        self.save_optimization_history(self._opt_hist_backup_path, append=True)
        if self._gen_opt_backup_plot and self.formulation.opt_problem.database:
            basepath = basename(self._opt_hist_backup_path).split(".")[0]
            self.post_process(
                "OptHistoryView", save=True, show=False, file_path=basepath
            )

    @property
    def posts(self):  # type: (...) -> List[str]
        """The available post-processors."""
        return self.post_factory.posts

    def post_process(
        self,
        post_name,  # type: str
        **options  # type: Union[OptPostProcessorOptionType,Path]
    ):  # type: (...) -> OptPostProcessor
        """Post-process the optimization history.

        Args:
            post_name: The name of the post-processor,
                i.e. the name of a class inheriting from :class:`.OptPostProcessor`.
            options: The options for the post-processor.
        """
        post = self.post_factory.execute(
            self.formulation.opt_problem, post_name, **options
        )
        return post

    def _run_algorithm(self):  # type: (...) -> OptimizationResult
        """Run the driver algorithm."""
        raise NotImplementedError()

    def __repr__(self):  # type: (...) -> str
        msg = MultiLineString()
        msg.add(self.name)
        msg.indent()
        msg.add("Disciplines: {}", pretty_repr(self.disciplines, delimiter=" "))
        msg.add("MDOFormulation: {}", self.formulation.__class__.__name__)
        msg.add("Algorithm: {}", self.local_data.get(self.ALGO))
        return str(msg)

    def get_disciplines_statuses(self):  # type: (...) -> Dict[str,str]
        """Retrieve the statuses of the disciplines.

        Returns:
            The statuses of the disciplines.
        """
        statuses = {}
        for disc in self.disciplines:
            statuses[disc.__class__.__name__] = disc.status
        return statuses

    def print_execution_metrics(self):  # type: (...)-> None
        """Print the total number of executions and cumulated runtime by discipline."""
        n_lin = 0
        n_calls = 0
        LOGGER.info("* Scenario Executions statistics *")
        for disc in self.disciplines:
            LOGGER.info("* Discipline: %s", disc.name)
            LOGGER.info("Executions number: %s", str(disc.n_calls))
            LOGGER.info("Execution time:  %s s", str(disc.exec_time))

            n_calls += disc.n_calls
            LOGGER.info("Linearizations number: %s", str(disc.n_calls_linearize))

            n_lin += disc.n_calls_linearize
        LOGGER.info("Total number of executions calls %s", str(n_calls))
        LOGGER.info("Total number of linearizations %s", str(n_lin))

    def xdsmize(
        self,
        monitor=False,  # type: bool
        outdir=".",  # type: Optional[str]
        print_statuses=False,  # type: bool
        outfilename="xdsm.html",  # type: str
        latex_output=False,  # type: bool
        open_browser=False,  # type: bool
        html_output=True,  # type: bool
        json_output=False,  # type: bool
    ):  # type: (...) -> None
        """Create a JSON file defining the XDSM related to the current scenario.

        Args:
            monitor: If True, update the generated file
                at each discipline status change.
            outdir: The directory where the JSON file is generated.
                If None, the current working directory is used.
            print_statuses: If True, print the statuses in the console at each update.
            outfilename: The name of the file of the output.
                The basename is used and the extension is adapted
                for the HTML / JSON / PDF outputs.
            latex_output: If True, build TEX, TIKZ and PDF files.
            open_browser: If True, open the web browser and display the the XDSM.
            html_output: If True, output a self contained HTML file.
            json_output: If True, output a JSON file for XDSMjs.
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

    def get_expected_dataflow(
        self,
    ):  # type: (...) -> List[Tuple[MDODiscipline,MDODiscipline,List[str]]]
        return self.formulation.get_expected_dataflow()

    def get_expected_workflow(self):  # type: (...) -> LoopExecSequence
        exp_wf = self.formulation.get_expected_workflow()
        return ExecutionSequenceFactory.loop(self, exp_wf)

    def _init_algo_factory(self):  # type: (...) -> None
        """Initalize the factory of algorithms."""
        raise NotImplementedError()

    def get_available_driver_names(self):  # type: (...) -> List[str]
        """The available drivers."""
        return self._algo_factory.algorithms

    def _update_input_grammar(self):  # type: (...) -> None
        """Update the input grammar from the names of available drivers."""
        available_algos = self.get_available_driver_names()
        algo_grammar = {"type": "string", "enum": available_algos}
        # TODO: Implement a cleaner solution to handle SimpleGrammar,
        #  use enum not str.

        if self.grammar_type == MDODiscipline.JSON_GRAMMAR_TYPE:
            self.input_grammar.set_item_value("algo", algo_grammar)
        else:
            self._update_grammar_input()

    def _update_grammar_input(self):  # type: (...) -> None
        """Update the inputs of a Grammar."""
        raise NotImplementedError()

    @staticmethod
    def is_scenario():  # type: (...) -> bool
        """Indicate if the current object is a :class:`.Scenario`.

        Returns:
            True if the current object is a :class:`.Scenario`.
        """
        return True
