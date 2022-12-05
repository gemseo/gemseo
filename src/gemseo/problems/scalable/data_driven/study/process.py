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
#         documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Scalability study - Process
===========================

The :class:`.ScalabilityStudy` class implements
the concept of scalability study:

1. By instantiating a :class:`.ScalabilityStudy`, the user defines
   the MDO problem in terms of design parameters, objective function and
   constraints.
2. For each discipline, the user adds a dataset stored
   in a :class:`.Dataset` and select a type of
   :class:`.ScalableModel` to build the :class:`.ScalableDiscipline`
   associated with this discipline.
3. The user adds different optimization strategies, defined in terms
   of both optimization algorithms and MDO formulation.
4. The user adds different scaling strategies, in terms of sizes of
   design parameters, coupling variables and equality and inequality
   constraints. The user can also define a scaling strategies according to
   particular parameters rather than groups of parameters.
5. Lastly, the user executes the :class:`.ScalabilityStudy` and the results
   are written in several files and stored into directories
   in a hierarchical way, where names depend on both MDO formulation,
   scaling strategy and replications when it is necessary. Different kinds
   of files are stored: optimization graphs, dependency matrix plots and
   of course, scalability results by means of a dedicated class:
   :class:`.ScalabilityResult`.
"""
from __future__ import annotations

import logging
import numbers
from copy import deepcopy
from pathlib import Path

from numpy import inf

from gemseo.problems.scalable.data_driven.problem import ScalableProblem
from gemseo.problems.scalable.data_driven.study.result import ScalabilityResult
from gemseo.utils.logging_tools import LoggingContext
from gemseo.utils.string_tools import MultiLineString

LOGGER = logging.getLogger(__name__)

RESULTS_DIRECTORY = Path("results")
POST_DIRECTORY = Path("visualization")
POSTOPTIM_DIRECTORY = POST_DIRECTORY / "optimization_history"
POSTSTUDY_DIRECTORY = POST_DIRECTORY / "scalability_study"
POSTSCAL_DIRECTORY = POST_DIRECTORY / "dependency_matrix"


class ScalabilityStudy:
    """Scalability Study."""

    def __init__(
        self,
        objective,
        design_variables,
        directory="study",
        prefix="",
        eq_constraints=None,
        ineq_constraints=None,
        maximize_objective=False,
        fill_factor=0.7,
        active_probability=0.1,
        feasibility_level=0.8,
        start_at_equilibrium=True,
        early_stopping=True,
        coupling_variables=None,
    ):
        """
        The constructor of the ScalabilityStudy class requires two mandatory
        arguments:

        - the :code:`'objective'` name,
        - the list of :code:`'design_variables'` names.

        Concerning output files, we can specify:

        - the :code:`directory` which is :code:`'study'` by default,
        - the prefix of output file names (default: no prefix).

        Regarding optimization parametrization, we can specify:

        - the list of equality constraints names (:code:`eq_constraints`),
        - the list of inequality constraints names (:code:`ineq_constraints`),
        - the choice of maximizing the objective function
          (:code:`maximize_objective`).

        By default, the objective function is minimized and the MDO problem
        is unconstrained.

        Last but not least, with regard to the scalability methodology,
        we can overwrite:

        - the default fill factor of the input-output dependency matrix
          :code:`ineq_constraints`,
        - the probability to set the inequality constraints as active at
          initial step of the optimization :code:`active_probability`,
        - the offset of satisfaction for inequality constraints
          :code:`feasibility_level`,
        - the use of a preliminary MDA to start at equilibrium
          :code:`start_at_equilibrium`,
        - the post-processing of the optimization database to get results
          earlier than final step :code:`early_stopping`.

        :param str objective: name of the objective
        :param list(str) design_variables: names of the design variables
        :param str directory: working directory of the study. Default: 'study'.
        :param str prefix: prefix for the output filenames. Default: ''.
        :param list(str) eq_constraints: names of the equality constraints.
            Default: None.
        :param list(str) ineq_constraints: names of the inequality constraints
            Default: None.
        :param bool maximize_objective: maximizing objective. Default: False.
        :param float fill_factor: default fill factor of the input-output
            dependency matrix. Default: 0.7.
        :param float active_probability: probability to set the inequality
            constraints as active at initial step of the optimization.
            Default: 0.1
        :param float feasibility_level: offset of satisfaction for inequality
            constraints. Default: 0.8.
        :param bool start_at_equilibrium: start at equilibrium
            using a preliminary MDA. Default: True.
        :param bool early_stopping: post-process the optimization database
            to get results earlier than final step.
        """
        LOGGER.info("Initialize the scalability study")
        self.prefix = prefix
        self.directory = Path(directory)
        self.__create_directories()
        self.datasets = []
        self.objective = objective
        self.design_variables = design_variables
        self.eq_constraints = eq_constraints or []
        self.ineq_constraints = ineq_constraints or []
        self.coupling_variables = coupling_variables or []
        self.maximize_objective = maximize_objective
        self.formulations = []
        self.formulations_options = []
        self.algorithms = []
        self.algorithms_options = []
        self.scalings = []
        self.var_scalings = []
        self.__check_fill_factor(fill_factor)
        self._default_fill_factor = fill_factor
        self.__check_proportion(active_probability)
        self.active_probability = active_probability
        if isinstance(feasibility_level, dict):
            for _, value in feasibility_level.items():
                self.__check_proportion(value)
        else:
            self.__check_proportion(feasibility_level)
        self.feasibility_level = feasibility_level
        self.start_at_equilibrium = start_at_equilibrium
        self.results = []
        self.early_stopping = early_stopping
        self._group_dep = {}
        self._fill_factor = {}
        self.top_level_diff = []
        self._all_data = None
        optimize = "maximize" if self.maximize_objective else "minimize"
        msg = MultiLineString()
        msg.indent()
        msg.add("Optimization problem")
        msg.indent()
        msg.add("Objective: {} {}", optimize, self.objective)
        msg.add("Design variables: {}", self.design_variables)
        msg.add("Equality constraints: {}", self.eq_constraints)
        msg.add("Inequality constraints: {}", self.ineq_constraints)
        msg.dedent()
        msg.add("Study properties")
        msg.indent()
        msg.add("Default fill factor: {}", self._default_fill_factor)
        msg.add("Active probability: {}", self.active_probability)
        msg.add("Feasibility level: {}", self.feasibility_level)
        msg.add("Start at equilibrium: {}", self.start_at_equilibrium)
        msg.add("Early stopping: {}", self.early_stopping)
        LOGGER.info("%s", msg)

    def __create_directories(self):
        """Create the different directories to store results, post-processings, ..."""
        self.directory.mkdir(exist_ok=True)
        post = self.directory / POST_DIRECTORY
        post.mkdir(exist_ok=True)
        postoptim = self.directory / POSTOPTIM_DIRECTORY
        postoptim.mkdir(exist_ok=True)
        poststudy = self.directory / POSTSTUDY_DIRECTORY
        poststudy.mkdir(exist_ok=True)
        postscal = self.directory / POSTSCAL_DIRECTORY
        postscal.mkdir(exist_ok=True)
        results = self.directory / RESULTS_DIRECTORY
        results.mkdir(exist_ok=True)
        msg = MultiLineString()
        msg.indent()
        msg.add("Create directories")
        msg.indent()
        msg.add("Working directory: {}", self.directory)
        msg.add("Post-processing: {}", post)
        msg.add("Optimization history view: {}", postoptim)
        msg.add("Scalability views: {}", poststudy)
        msg.add("Dependency matrices: {}", postscal)
        msg.add("Results: {}", results)
        LOGGER.info("%s", msg)

    def add_discipline(self, data):
        """This method adds a disciplinary dataset from a dataset.

        :param Dataset data: dataset provided as a dataset.
        """
        self._group_dep[data.name] = {}
        self._all_data = data.get_all_data()
        self.datasets.append(data)
        for output_name in data.get_names(data.OUTPUT_GROUP):
            self.set_fill_factor(data.name, output_name, self._default_fill_factor)
        inputs = ", ".join(
            [f"{name}({data.sizes[name]})" for name in data.get_names(data.INPUT_GROUP)]
        )
        outputs = ", ".join(
            [
                f"{name}({data.sizes[name]})"
                for name in data.get_names(data.OUTPUT_GROUP)
            ]
        )
        msg = MultiLineString()
        msg.add("Add scalable discipline # {}", len(self.datasets))
        msg.indent()
        msg.add("Name: {}", data.name)
        msg.add("Inputs: {}", inputs)
        msg.add("Outputs: {}", outputs)
        msg.add("Built from {}", len(data))
        LOGGER.info("%s", msg)

    @property
    def disciplines_names(self):
        """Get discipline names.

        :return: list of discipline names
        :rtype: list(str)
        """
        disc_names = [discipline.name for discipline in self.datasets]
        return disc_names

    def set_input_output_dependency(self, discipline, output, inputs):
        """Set the dependency between an output and a set of inputs for a given
        discipline.

        :param str discipline: name of the discipline
        :param str output: name of the output
        :param list(str) inputs: list of inputs names
        """
        self.__check_discipline(discipline)
        self.__check_output(discipline, output)
        self.__check_inputs(discipline, inputs)
        self._group_dep[discipline][output] = inputs

    def set_fill_factor(self, discipline, output, fill_factor):
        """
        :param str discipline: name of the discipline
        :param str output: name of the output function
        :param float fill_factor: fill factor
        """
        self.__check_discipline(discipline)
        self.__check_output(discipline, output)
        self.__check_fill_factor(fill_factor)
        if discipline not in self._fill_factor:
            self._fill_factor[discipline] = {}
        self._fill_factor[discipline][output] = fill_factor

    def __check_discipline(self, discipline):
        """Check if discipline is a string comprised in the list of disciplines names."""
        if not isinstance(discipline, str):
            raise TypeError("The argument discipline should be a string")
        disciplines_names = self.disciplines_names
        if discipline not in disciplines_names:
            raise ValueError(
                "The argument discipline should be a string comprised in the list %s",
                disciplines_names,
            )

    def __check_output(self, discipline, varname):
        """Check if a variable is an output of a given discipline."""
        self.__check_discipline(discipline)
        if not isinstance(varname, str):
            raise TypeError(f"{varname} is not a string.")
        outputs_names = next(
            dataset.get_names(dataset.OUTPUT_GROUP)
            for dataset in self.datasets
            if dataset.name == discipline
        )
        if varname not in outputs_names:
            raise ValueError(
                "'{}' is not an output of {}; available outputs are: {}".format(
                    varname, discipline, outputs_names
                )
            )

    def __check_inputs(self, discipline, inputs):
        """Check if inputs is a list of inputs of discipline."""
        self.__check_discipline(discipline)
        if not isinstance(inputs, list):
            raise TypeError("The argument 'inputs' must be a list of string.")
        inputs_names = next(
            dataset.get_names(dataset.INPUT_GROUP)
            for dataset in self.datasets
            if dataset.name == discipline
        )
        for inpt in inputs:
            if not isinstance(inpt, str):
                raise TypeError(f"{inpt} is not a string.")
            if inpt not in inputs_names:
                raise ValueError(
                    "'{}' is not a discipline input; available inputs are: {}".format(
                        inpt, inputs_names
                    )
                )

    def __check_fill_factor(self, fill_factor):
        """Check if fill factor is a proportion or a number equal to -1.

        :param float fill_factor: a proportion or -1
        """
        try:
            self.__check_proportion(fill_factor)
        except ValueError:
            if fill_factor != -1:
                raise TypeError(
                    "Fill factor should be a float number comprised in 0 and 1 "
                    "or a number equal to -1."
                )

    @staticmethod
    def __check_proportion(proportion):
        """Check if a proportion is a float number comprised in 0 and 1.

        :param float proportion: proportion comprised in 0 and 1.
        """
        if not isinstance(proportion, numbers.Number):
            raise TypeError(
                "A proportion should be a float number comprised in 0 and 1."
            )
        if not 0 <= proportion <= 1:
            raise ValueError(
                "A proportion should be a float number comprised in 0 and 1."
            )

    def add_optimization_strategy(
        self,
        algo,
        max_iter,
        formulation="DisciplinaryOpt",
        algo_options=None,
        formulation_options=None,
        top_level_diff="auto",
    ):
        """Add both optimization algorithm and MDO formulation, as well as their options.

        :param str algo: name of the optimization algorithm.
        :param int max_iter: maximum number of iterations
            for the optimization algorithm.
        :param str formulation: name of the MDO formulation.
            Default: 'DisciplinaryOpt'.
        :param dict algo_options: options of the optimization algorithm.
        :param dict formulation_options: options of the MDO formulation.
        :param str top_level_diff: differentiation method
            for the top level disciplines. Default: 'auto'.
        """
        self.algorithms.append(algo)
        if algo_options is None:
            algo_options = {}
        else:
            if not isinstance(algo_options, dict):
                raise TypeError("algo_options must be a dictionary.")
        algo_options.update({"max_iter": max_iter})
        self.algorithms_options.append(algo_options)
        self.formulations.append(formulation)
        self.formulations_options.append(formulation_options)
        self.top_level_diff.append(top_level_diff)
        if algo_options is not None:
            algo_options = ", ".join(
                [f"{name}({value})" for name, value in algo_options.items()]
            )
        if formulation_options is not None:
            formulation_options = ", ".join(
                [f"{name}({value})" for name, value in formulation_options.items()]
            )
        msg = MultiLineString()
        msg.add("Add optimization strategy # {}", len(self.formulations))
        msg.indent()
        msg.add("Algorithm: {}", algo)
        msg.add("Algorithm options: {}", algo_options)
        msg.add("Formulation: {}", formulation)
        msg.add("Formulation options: {}", formulation_options)
        LOGGER.info("%s", msg)

    def add_scaling_strategies(
        self,
        design_size=None,
        coupling_size=None,
        eq_cstr_size=None,
        ineq_cstr_size=None,
        variables=None,
    ):
        """Add different scaling strategies.

        :param design_size: size of the design variables. Default: None.
        :type design_size: int or list(int)
        :param coupling_size: size of the coupling variables. Default: None.
        :type coupling_size: int or list(int)
        :param eq_cstr_size: size of the equality constraints. Default: None.
        :type eq_cstr_size: int or list(int)
        :param ineq_cstr_size: size of the inequality constraints.
            Default: None.
        :type ineq_cstr_size: int or list(int)
        """
        n_design = self.__check_varsizes_type(design_size)
        n_coupling = self.__check_varsizes_type(coupling_size)
        n_eq = self.__check_varsizes_type(eq_cstr_size)
        n_ineq = self.__check_varsizes_type(ineq_cstr_size)
        n_var = self.__check_varsizes_type(variables)
        n_scaling = max(n_design, n_coupling, n_eq, n_ineq, n_var)
        self.__check_scaling_consistency(n_design, n_scaling)
        self.__check_scaling_consistency(n_coupling, n_scaling)
        self.__check_scaling_consistency(n_eq, n_scaling)
        self.__check_scaling_consistency(n_ineq, n_scaling)
        design_size = self.__format_scaling(design_size, n_scaling)
        coupling_size = self.__format_scaling(coupling_size, n_scaling)
        eq_cstr_size = self.__format_scaling(eq_cstr_size, n_scaling)
        ineq_cstr_size = self.__format_scaling(ineq_cstr_size, n_scaling)
        for idx in range(n_scaling):
            var_scaling = {}
            self.__update_var_scaling(
                var_scaling, design_size[idx], self.design_variables
            )
            self.__update_var_scaling(
                var_scaling, coupling_size[idx], self.coupling_variables
            )
            self.__update_var_scaling(
                var_scaling, eq_cstr_size[idx], self.eq_constraints
            )
            self.__update_var_scaling(
                var_scaling, ineq_cstr_size[idx], self.ineq_constraints
            )
            scaling = {
                "design_variables": design_size[idx],
                "coupling_variables": coupling_size[idx],
                "eq_constraint_size": eq_cstr_size[idx],
                "ineq_constraint_size": ineq_cstr_size[idx],
            }
            if variables is not None and variables[0] is not None:
                for varname, value in variables[idx].items():
                    self.__update_var_scaling(var_scaling, value, [varname])
                    scaling[varname] = value
            else:
                variables = [None] * n_scaling
            self.var_scalings.append(var_scaling)
            self.scalings.append(scaling)
        msg = MultiLineString()
        msg.add("Add scaling strategies")
        msg.indent()
        msg.add("Number of strategies: {}", n_scaling)
        for idx in range(n_scaling):
            if variables[idx] is not None:
                var_str = ", ".join(
                    [f"{name}({size})" for name, size in variables[idx].items()]
                )
            else:
                var_str = None
            msg.add("Strategy # {}", idx + 1)
            msg.indent()
            msg.add("Design variables: {}", design_size[idx])
            msg.add("Coupling variables: {}", coupling_size[idx])
            msg.add("Equality constraints: {}", eq_cstr_size[idx])
            msg.add("Inequality constraints: {}", ineq_cstr_size[idx])
            msg.add("Variables: {}", var_str)
            msg.dedent()
        LOGGER.info("%s", msg)

    @staticmethod
    def __format_scaling(size, n_scaling):
        """Convert a scaling size in a list of integers whose length is equal to the
        number of scalings.

        :param size: size(s) of a given variable
        :type size: int or list(int)
        :param int n_scaling: number of scalings
        :return: formatted sizes
        """
        formatted_sizes = size
        if isinstance(formatted_sizes, int) or formatted_sizes is None:
            formatted_sizes = [formatted_sizes]
        if len(formatted_sizes) == 1:
            formatted_sizes *= n_scaling
        return formatted_sizes

    @staticmethod
    def __update_var_scaling(scaling, size, varnames):
        """Update a scaling dictionary for a given list of variables and a given size.

        :param dict scaling: scaling dictionary whose keys are variable names
            and values are dictionary with scaling properties,
            e.g. {'size': val}
        :param int size: size of the variable
        :param list(str) varnames: list of variable names
        """
        if size is not None:
            scaling.update({varname: size for varname in varnames})

    @staticmethod
    def __check_scaling_consistency(n_var_scaling, n_scaling):
        """Check that for the different types of variables, the number of scalings is the
        same or equal to 1.

        :param int n_var_scaling: number of scalings
        :param int n_scaling: expected number of scalings
        """
        assert n_var_scaling in (n_scaling, 1)

    @staticmethod
    def __check_varsizes_type(varsizes):
        """Check the type of scaling sizes. Integer, list of integers or None is
        expected. Return the number of scalings.

        :return: length of scalings
        """
        length = 1
        if varsizes is not None:
            if isinstance(varsizes, list):
                for size in varsizes:
                    if isinstance(size, dict):
                        for _, value in size.items():
                            assert isinstance(value, int)
                    else:
                        assert isinstance(size, int)
                length = len(varsizes)
            else:
                assert isinstance(varsizes, int)
                length = 1
        return length

    def execute(self, n_replicates=1):
        """Execute the scalability study, one or several times to take into account the
        random features of the scalable problems.

        :param int n_replicates: number of times the scalability study
            is repeated. Default: 1.
        """
        plural = "s" if n_replicates > 1 else ""
        LOGGER.info("Execute scalability study %s time%s", n_replicates, plural)
        if not self.formulations and not self.algorithms:
            raise ValueError(
                "A scalable study needs at least 1 optimization strategy, "
                "defined by a mandatory optimization algorithm "
                "and optional optimization algorithm and options"
            )
        counter = "Formulation: {} - Algo: {} - Scaling: {}/{} - Replicate: {}/{}"
        n_scal_strategies = len(self.var_scalings)
        n_opt_strategies = len(self.algorithms)
        for opt_index in range(n_opt_strategies):
            algo = self.algorithms[opt_index]
            formulation = self.formulations[opt_index]
            for scal_index in range(n_scal_strategies):
                scaling = self.var_scalings[scal_index]
                for replicate in range(1, n_replicates + 1):
                    msg = MultiLineString()
                    msg.indent()
                    data = (
                        formulation,
                        algo,
                        scal_index + 1,
                        n_scal_strategies,
                        replicate,
                        n_replicates,
                    )
                    msg.add(counter, *data)
                    LOGGER.info("%s", msg)
                    msg = MultiLineString()
                    msg.indent()
                    msg.indent()
                    msg.add("Create scalable problem")
                    problem = self.__create_scalable_problem(scaling, replicate)
                    path = self.__dep_mat_path(algo, formulation, scal_index, replicate)
                    directory = path.stem
                    path.mkdir(exist_ok=True)
                    msg.add("Save dependency matrices in {}", path)
                    problem.plot_dependencies(True, False, str(path))
                    msg.add("Create MDO Scenario")
                    with LoggingContext():
                        self.__create_scenario(problem, formulation, opt_index)
                        msg.add("Execute MDO Scenario")
                        formulation_options = self.formulations_options[opt_index]
                        algo_options = self.__execute_scenario(problem, algo, opt_index)

                    path = self.__optview_path(algo, formulation, scal_index, replicate)
                    msg.add("Save optim history view in {}", path)
                    fpath = str(path) + "/"
                    problem.scenario.post_process(
                        "OptHistoryView", save=True, show=False, file_path=fpath
                    )
                    result = ScalabilityResult(directory, scal_index + 1, replicate)
                    self.results.append(result)
                    statistics = self.__get_statistics(problem, scaling)
                    result.get(
                        algo=algo,
                        algo_options=algo_options,
                        formulation=formulation,
                        formulation_options=formulation_options,
                        scaling=scaling,
                        disc_names=problem.disciplines,
                        output_names=problem.outputs,
                        **statistics,
                    )
                    fpath = result.get_file_path(self.directory)
                    msg.add("Save statistics in {}", fpath)
                    result.save(str(self.directory))
                    LOGGER.debug("%s", msg)
        return self.results

    def __get_statistics(self, problem, scaling):
        """Get statistics from an executed scalable problem.

        :param ScalableProblem problem: scalable problem.
        :param dict scaling: variables scaling.
        """
        statistics = {}
        stopidx, n_iter = self.__get_stop_index(problem)
        ratio = float(stopidx) / n_iter
        n_calls = {disc: n_calls * ratio for disc, n_calls in problem.n_calls.items()}
        statistics["n_calls"] = n_calls
        tmp = problem.n_calls_linearize
        n_calls_linearize = {disc: ncl * ratio for disc, ncl in tmp.items()}
        statistics["n_calls_linearize"] = n_calls_linearize
        tmp = problem.n_calls_top_level
        n_calls_tl = {disc: n_calls * ratio for disc, n_calls in tmp.items()}
        statistics["n_calls_top_level"] = n_calls_tl
        tmp = problem.n_calls_linearize_top_level
        n_calls_linearize_tl = {disc: ncltl * ratio for disc, ncltl in tmp.items()}
        statistics["n_calls_linearize_top_level"] = n_calls_linearize_tl
        statistics["exec_time"] = problem.exec_time() * ratio
        statistics["status"] = problem.status
        statistics["is_feasible"] = problem.is_feasible

        inputs = problem.inputs
        outputs = problem.outputs
        disc_varnames = {
            disc: inputs[disc] + outputs[disc] for disc in problem.disciplines
        }
        sizes = problem.varsizes
        statistics["new_varsizes"] = {
            disc: {name: scaling.get(name, sizes[name]) for name in disc_varnames[disc]}
            for disc in problem.disciplines
        }
        statistics["old_varsizes"] = problem.varsizes
        return statistics

    def __dep_mat_path(self, algo, formulation, id_scaling, replicate):
        """Path to the directory containing the dependency matrices files.

        :param str algo: algo name.
        :param str formulation: formulation name.
        :param int id_scaling: scaling index.
        :param int replicate: replicate number.
        """
        varnames = [algo, formulation, id_scaling + 1, replicate]
        name = "_".join([self.prefix] + [str(var) for var in varnames])
        if name[0] == "_":
            name = name[1:]
        path = self.directory / POSTSCAL_DIRECTORY / name
        return path

    def __optview_path(self, algo, formulation, id_scaling, replicate):
        """Path to the directory containing the dependency matrices files.

        :param str algo: algo name.
        :param str formulation: formulation name.
        :param int id_scaling: scaling index.
        :param int replicate: replicate number.
        """
        path = (
            self.directory
            / POSTOPTIM_DIRECTORY
            / Path(f"{self.prefix}_{algo}_{formulation}")
            / Path(f"scaling_{id_scaling + 1}")
            / Path(f"replicate_{replicate}")
        )
        path.mkdir(exist_ok=True, parents=True)
        return path

    def __create_scalable_problem(self, scaling, seed):
        """Create a scalable problem.

        :param dict scaling: scaling.
        :param int seed: seed for random features.
        """
        problem = ScalableProblem(
            self.datasets,
            self.design_variables,
            self.objective,
            self.eq_constraints,
            self.ineq_constraints,
            self.maximize_objective,
            scaling,
            fill_factor=self._fill_factor,
            seed=seed,
            group_dep=self._group_dep,
            force_input_dependency=True,
            allow_unused_inputs=False,
        )
        return problem

    def __create_scenario(self, problem, formulation, opt_index):
        """Create scenario for a given formulation.

        :param ScalableProblem problem: scalable problem.
        :param str formulation: MDO formulation name.
        :param int opt_index: optimization strategy index.
        """
        form_opt = self.formulations_options[opt_index]
        if not isinstance(form_opt, dict):
            formulation_options = {}
        else:
            formulation_options = form_opt
        problem.create_scenario(
            formulation,
            "MDO",
            self.start_at_equilibrium,
            self.active_probability,
            self.feasibility_level,
            **formulation_options,
        )

    def __execute_scenario(self, problem, algo, opt_index):
        """Execute scenario.

        :param ScalableProblem problem: scalable problem.
        :param str algo: optimization algorithm name.
        :param int opt_index: optimization strategy index.
        """
        top_level_disciplines = problem.scenario.formulation.get_top_level_disc()
        for disc in top_level_disciplines:
            disc.linearization_mode = self.top_level_diff[opt_index]
        algo_options = deepcopy(self.algorithms_options[opt_index])
        max_iter = algo_options["max_iter"]
        del algo_options["max_iter"]
        problem.scenario.execute(
            {"algo": algo, "max_iter": max_iter, "algo_options": algo_options}
        )
        return algo_options

    def __get_stop_index(self, problem):
        """Get stop index from a database.

        :param ScalableProblem problem: scalable problem
        :return: stop index, database length
        :rtype: int, int
        """
        database = problem.scenario.formulation.opt_problem.database
        n_iter = len(database)
        if self.early_stopping:
            y_prev = inf
            stopidx = 0
            for _, value in database.items():
                pbm = problem.scenario.formulation.opt_problem
                if y_prev == inf:
                    diff = inf
                else:
                    diff = abs(y_prev - value[pbm.get_objective_name()])
                    diff /= abs(y_prev)
                if diff < 1e-6:
                    break
                y_prev = value[pbm.get_objective_name()]
                stopidx += 1
        else:
            stopidx = n_iter
        return stopidx, n_iter
