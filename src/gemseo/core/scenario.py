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
"""
Base class for all Scenarios
****************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import inspect
from builtins import str, super
from os import remove
from os.path import abspath, basename
from os.path import dirname as pdirname
from os.path import exists

from future import standard_library
from six import string_types

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import ExecutionSequenceFactory
from gemseo.core.function import MDOFunction
from gemseo.formulations.formulations_factory import MDOFormulationsFactory
from gemseo.post.post_factory import PostFactory

standard_library.install_aliases()


from gemseo import LOGGER


class Scenario(MDODiscipline):
    """Base class for MDO and DOE scenarios.

    Multidisciplinary Design Optimization Scenario, main user interface
    Creates an optimization problem and solves it with a driver

    MDO Problem description: links the disciplines and the formulation
    to create an optimization problem.
    Use the class by instantiation.
    Create your disciplines beforehand.

    Specify the formulation by giving the class name such as the string
    "MDF"

    The reference_input_data is the typical input data dict that is provided
    to the run method of the disciplines

    Specify the objective function name, which must be an output
    of a discipline of the scenario, with the "objective_name" attribute

    If you want to add additional design constraints,
    use the add_constraint method

    To view the results, use the "post_process" method after execution.
    You can view:

    - The design variables history, the objective value, the constraints,
      by using:
      scenario.post_process("OptHistoryView", show=False, save=True)
    - Quadratic approximations of the functions close to the
      optimum, when using gradient based algorithms, by using:
      scenario.post_process("QuadApprox", method="SR1", show=False,
      save=True, function="my_objective_name",
      file_path="appl_dir")
    - Self Organizing Maps of the design space, by using:
      scenario.post_process("SOM", save=True, file_path="appl_dir")

    To list post processings on your setup,
    use the method scenario.posts
    For more detains on their options, go to the "gemseo.post" package


    """

    # Constants for input variables in json schema
    X_0 = "x_0"
    U_BOUNDS = "u_bounds"
    L_BOUNDS = "l_bounds"
    ALGO = "algo"
    ALGO_OPTIONS = "algo_options"

    def __init__(
        self,
        disciplines,
        formulation,
        objective_name,
        design_space,
        name=None,
        **formulation_options
    ):
        """
        Constructor, initializes the MDO scenario
        Objects instantiation and checks are made before run intentionally

        :param disciplines: the disciplines of the scenario
        :param formulation: the formulation name,
            the class name of the formulation in gemseo.formulations
        :param objective_name: the objective function name
        :param design_space: the design space
        :param name: scenario name
        :param formulation_options: options for creation of the formulation
        """
        self.formulation = None
        self.formulation_name = None
        self.disciplines = disciplines
        self.optimization_result = None
        self._algo_factory = None

        self._check_disciplines()
        self._init_algo_factory()
        self._form_factory = MDOFormulationsFactory()
        super(Scenario, self).__init__(name=name)
        self._init_base_grammar(self.__class__.__name__)
        self._init_formulation(
            formulation, objective_name, design_space, **formulation_options
        )
        self.post_factory = PostFactory()
        self._update_input_grammar()

    def _check_disciplines(self):
        """
        Check that two disciplines dont compute the same output
        """

        all_outs = set()
        for disc in self.disciplines:
            outs = set(disc.get_output_data_names())
            common = outs & all_outs
            if len(common) > 0:
                raise ValueError(
                    "Two disciplines, among which "
                    + disc.name
                    + " compute the same output :"
                    + str(common)
                )
            all_outs = all_outs | outs

    @property
    def design_space(self):
        """
        Proxy for formulation.design_space

        :returns: the design space
        """
        return self.formulation.design_space

    def _init_base_grammar(self, name):
        """Initializes the base grammars from MDO scenario inputs and outputs
        This ensures that subclasses have base scenario inputs and outputs
        Can be overloaded by subclasses if this is not desired.

        :param name: name of the scenario, used as base name for the json
            schema to import: name_input.json and name_output.json
        """
        comp_dir = abspath(pdirname(inspect.getfile(Scenario)))
        input_grammar_file = self.auto_get_grammar_file(True, name, comp_dir)
        output_grammar_file = self.auto_get_grammar_file(False, name, comp_dir)
        self._instantiate_grammars(input_grammar_file, output_grammar_file)

    def set_differentiation_method(self, method="user", step=1e-6):
        """Sets the differentiation method for the process

        :param method: the method to use, either "user", "finite_differences",
            or "complex_step" or "no_derivatives",
            which is equivalent to None. (Default value = "user")
        :param step: Default value = 1e-6)
        """
        if method is None:
            method = "no_derivatives"
        self.formulation.opt_problem.differentiation_method = method
        self.formulation.opt_problem.fd_step = step

    def add_constraint(
        self,
        output_name,
        constraint_type=MDOFunction.TYPE_EQ,
        constraint_name=None,
        value=None,
        positive=False,
    ):
        """Add a user constraint, i.e. a design constraint in addition to
        formulation specific constraints such as targets in IDF.
        The strategy of repartition of constraints is defined in the
        formulation class.

        :param output_name: the output name to be used as constraint
            for instance, if g_1 is given and
            constraint_type="eq",
            g_1=0 will be added as constraint to the optimizer
            If a list is given, a single discipline
            must provide all
            outputs
        :param constraint_type: the type of constraint, "eq" for equality,
            "ineq" for inequality constraint
            (Default value = MDOFunction.TYPE_EQ)
        :param constraint_name: name of the constraint to be stored,
            if None, generated from the output name
            (Default value = None)
        :param value: Default value = None)
        :param positive: Default value = False)
        :returns: the constraint function as an MDOFunction
        """
        if constraint_type not in [MDOFunction.TYPE_EQ, MDOFunction.TYPE_INEQ]:
            raise ValueError(
                "Constraint type must be either 'eq' or 'ineq',"
                + " got:"
                + str(constraint_type)
                + " instead"
            )

        self.formulation.add_constraint(
            output_name,
            constraint_type,
            constraint_name=constraint_name,
            value=value,
            positive=positive,
        )

    def _init_formulation(
        self, formulation, objective_name, design_space, **formulation_options
    ):
        """Initializes the formulation given disciplines, objective name
        and design variables names

        :param formulation: the formulation name to use
        :param design_space: the design space object
        :param objective_name: the objective function name
        :param formulation_options: options for creation of the formulation
        """
        if not isinstance(formulation, string_types):
            raise TypeError(
                "Formulation must be specified by its name!"
                + " Please use GEMSEO_PATH to specify custom formulations"
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

    def get_optim_variables_names(self):
        """A convenience function to access formulation design variables names

        :returns: the decision variables of the scenario
        :rtype: list(str)
        """
        return self.formulation.get_optim_variables_names()

    def get_optimum(self):
        """Return the optimization results

        :returns: Optimal solution found by the scenario if executed, None
                  otherwise
        :rtype: OptimizationResult
        """

        return self.optimization_result

    def save_optimization_history(
        self, file_path, file_format=OptimizationProblem.HDF5_FORMAT, append=False
    ):
        """Saves the optimization history of the scenario
        to a file

        :param file_path: The path to the file to save the history
        :param file_format: The format of the file, either "hdf5" or "ggobi"
            (Default value = "hdf5")
        :param append: if True, data is appended to the file if not empty
            (Default value = False)
        """
        opt_pb = self.formulation.opt_problem
        if file_format == OptimizationProblem.HDF5_FORMAT:
            opt_pb.export_hdf(file_path=file_path, append=append)
        elif file_format == OptimizationProblem.GGOBI_FORMAT:
            opt_pb.database.export_to_ggobi(file_path=file_path)
        else:
            raise ValueError(
                "Cannot export optimization history"
                + " to file format:"
                + str(file_format)
            )

    def set_optimization_history_backup(
        self,
        file_path,
        each_new_iter=False,
        each_store=True,
        erase=False,
        pre_load=False,
        generate_opt_plot=False,
    ):
        """
        Sets the backup file for the optimization history during the run

        :param file_path: The path to the file to save the history
        :param each_new_iter: if True, callback at every iteration
        :param each_store: if True, callback at every call to store()
            in the database
        :param erase: if True, the backup file is erased before the run
        :param pre_load: if True, the backup file is loaded before run,
            useful after a crash
        :param generate_opt_plot: generates the optimization history view
            at backup
        """
        opt_pb = self.formulation.opt_problem

        if exists(file_path):
            if erase and pre_load:
                raise ValueError(
                    "Conflicting options for history backup, "
                    + "cannot pre load optimization history"
                    + " and erase it!"
                )
            if erase:
                LOGGER.warning("Erasing optimization history in %s", str(file_path))
                remove(file_path)
            elif pre_load:
                opt_pb.database.import_hdf(file_path)

        def backup_callback():
            """
            A callback function to backup optimization history
            """
            self.save_optimization_history(file_path, append=True)
            if generate_opt_plot and opt_pb.database:
                basepath = basename(file_path).split(".")[0]
                self.post_process(
                    "OptHistoryView", save=True, show=False, file_path=basepath
                )

        opt_pb.add_callback(
            backup_callback, each_new_iter=each_new_iter, each_store=each_store
        )

    @property
    def posts(self):
        """Lists the available post processings

        :returns: the list of methods
        """
        return self.post_factory.posts

    def post_process(self, post_name, **options):
        """Finds the appropriate library and executes
        the post processing on the problem

        :param post_name: the post processing name
        :param options: options for the post method, see its package
        """
        post = self.post_factory.execute(
            self.formulation.opt_problem, post_name, **options
        )
        return post

    def _run_algorithm(self):
        """Runs the algo, either DOE or optimizer"""
        raise NotImplementedError()

    def __str__(self):
        """
        Summarize optimizations results for display

        :returns: string summarizing optimization results
        """
        msg = self.__class__.__name__ + ":\nDisciplines: "
        disc_names = [disc.name for disc in self.disciplines]  # pylint: disable=E1101
        msg += " ".join(disc_names)
        msg += "\nMDOFormulation: "
        msg += self.formulation.__class__.__name__
        msg += "\nAlgorithm: "
        msg += str(self.local_data.get(self.ALGO)) + "\n"
        return msg

    def log_me(self):
        """Logs a representation of the scenario characteristics
        logs self.__repr__ message
        """
        msg = str(self)
        for line in msg.split("\n"):
            LOGGER.info(line)

    def get_disciplines_statuses(self):
        """Retrieves the disciplines statuses

        :returns: the statuses dict, key: discipline name, value: status
        """
        statuses = {}
        for disc in self.disciplines:
            statuses[disc.__class__.__name__] = disc.status
        return statuses

    def print_execution_metrics(self):
        """Prints total number of executions and cumulated runtime
        by discipline
        """
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
        monitor=False,
        outdir=".",
        print_statuses=False,
        outfilename="xdsm.html",
        latex_output=False,
        open_browser=False,
        html_output=True,
        json_output=False,
    ):
        """
        Creates an xdsm.json file from the current scenario.
        If monitor is set to True, the xdsm.json file is updated to reflect
        discipline status update (hence monitor name).

        :param bool monitor: if True, updates the generated file at each
            discipline status change
        :param str outdir: the directory where XDSM json file is generated
        :param bool print_statuses: print the statuses in the console at
            each update
        :param outfilename: file name of the output. THe basename
            is used and the extension adapted for the HTML / JSON / PDF
            outputs
        :param bool latex_output: build .tex, .tikz and .pdf file
        :param open_browser: if True, opens the web browser with the XDSM
        :param html_output: if True, outputs a self contained HTML file
        :param json_output: if True, outputs a JSON file for XDSMjs
        """
        from gemseo.utils.xdsmizer import XDSMizer

        if print_statuses:
            monitor = True

        if monitor:
            XDSMizer(self).monitor(outdir=outdir, print_statuses=print_statuses)
        else:
            XDSMizer(self).run(
                outdir=outdir,
                latex_output=latex_output,
                open_browser=open_browser,
                html_output=html_output,
                json_output=json_output,
            )

    def get_expected_dataflow(self):
        """Overriden method from MDODiscipline base class
        delegated to formulation object

        """
        return self.formulation.get_expected_dataflow()

    def get_expected_workflow(self):
        """Overriden method from MDODiscipline base class
        delegated to formulation object

        """
        exp_wf = self.formulation.get_expected_workflow()
        return ExecutionSequenceFactory.loop(self, exp_wf)

    def _init_algo_factory(self):
        """
        Initalizes the algorithms factory
        """
        raise NotImplementedError()

    def get_available_driver_names(self):
        """
        Returns the list of available drivers
        """
        return self._algo_factory.algorithms

    def _update_input_grammar(self):
        """
        Updates input grammar from algos names
        """

        available_algos = self.get_available_driver_names()
        algo_grammar = {"type": "string", "enum": available_algos}
        self.input_grammar.set_item_value("algo", algo_grammar)

    @staticmethod
    def is_scenario():
        """
        Retuns True if self is a scenario

        :returns: True if self is a scenario
        """
        return True
