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
#         documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Benchmark MDO formulations based on scalable disciplines
========================================================

The :mod:`~gemseo.problems.scalable.api` implements several classes
to benchmark MDO formulations based on scalable disciplines.

The :class:`.ScalabilityStudy` class implements
the concept of scalability study:

1. By instantiating a :class:`.ScalabilityStudy`, the user defines
   the MDO problem in terms of design parameters, objective function and
   constraints.
2. For each discipline, the user adds a dataset stored
   in a :class:`.AbstractFullCache` and select a type of
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
   in a hierarchical way, where names depends on both MDO formulation,
   scaling strategy and replications when it is necessary. Different kinds
   of files are stored: optimization graphs, dependency matrix plots and
   of course, scalability results by means of a dedicated class:
   :class:`.ScalabilityResult`.

The :class:`.PostScalabilityStudy` class implements the way as the set of
:class:`.ScalabilityResult`-based result files
contained in the study directory are graphically post-processed. This class
provides several methods to easily change graphical properties, notably
the plot labels. It also makes it possible to define a cost function per
MDO formulation, converting the numbers of executions and linearizations
of the different disciplines required by a MDO process in an estimation
of the computational cost associated with what would be a scaled version
of the true problem.

.. warning::

   Comparing MDO formulations in terms of estimated true computational time
   rather than CPU time of the :class:`.ScalabilityStudy` is highly
   recommended.
   Indeed, time is often an obviousness criterion to distinguish between
   MDO formulations having the same performance in terms of distance to the
   optimum: look at our calculation budget and choose the best formulation
   that satisfies this budget, or even saves us time. Thus, it is important
   to carefully define these cost functions.

"""
from __future__ import absolute_import, division, unicode_literals

import numbers
import os
import pickle
from builtins import int, open
from copy import deepcopy

import matplotlib.pyplot as plt
from future import standard_library
from matplotlib.lines import Line2D
from numpy import array, inf, median
from past.builtins import basestring

from gemseo.problems.scalable.problem import ScalableProblem

standard_library.install_aliases()

from gemseo import LOGGER

CURRENT_DIRECTORY = os.getcwd()
RESULTS_DIRECTORY = "results"
POST_DIRECTORY = "visualization"
POSTOPTIM_DIRECTORY = os.path.join(POST_DIRECTORY, "optimization_history")
POSTSTUDY_DIRECTORY = os.path.join(POST_DIRECTORY, "scalability_study")
POSTSCAL_DIRECTORY = os.path.join(POST_DIRECTORY, "dependency_matrix")


class ScalabilityStudy(object):
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
        LOGGER.info("*** Start a scalability study ***")
        self.prefix = prefix
        self.directory = directory
        self.__create_directories()
        self.datasets = []
        self.objective = objective
        self.design_variables = design_variables
        if eq_constraints is None:
            eq_constraints = []
        self.eq_constraints = eq_constraints
        if ineq_constraints is None:
            ineq_constraints = []
        self.ineq_constraints = ineq_constraints
        if coupling_variables is None:
            coupling_variables = []
        self.coupling_variables = coupling_variables
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
        if not hasattr(feasibility_level, "__len__"):
            self.__check_proportion(feasibility_level)
        else:
            for _, value in feasibility_level.items():
                self.__check_proportion(value)
        self.feasibility_level = feasibility_level
        self.start_at_equilibrium = start_at_equilibrium
        self.results = []
        self.early_stopping = early_stopping
        self._group_dep = {}
        self._fill_factor = {}
        self.top_level_diff = []
        self._all_data = None
        LOGGER.info("Initialize the scalability study:")
        optimize = "maximize" if self.maximize_objective else "minimize"
        LOGGER.info("|    Objective: %s %s", optimize, self.objective)
        tmp = ", ".join(self.design_variables)
        LOGGER.info("|    Design variables: %s", tmp)
        tmp = ", ".join(self.eq_constraints)
        LOGGER.info("|    Equality constraints: %s", tmp)
        tmp = ", ".join(self.ineq_constraints)
        LOGGER.info("|    Inequality constraints: %s", tmp)
        LOGGER.info("|    Default fill factor: %s", self._default_fill_factor)
        LOGGER.info("|    Active probability: %s", self.active_probability)
        LOGGER.info("|    Feasibility level: %s", self.feasibility_level)
        LOGGER.info("|    Start at equilibrium: %s", self.start_at_equilibrium)
        LOGGER.info("|    Early stopping: %s", self.early_stopping)

    def __create_directories(self):
        """Create the different directories
        to store results, post-processings, ..."""
        LOGGER.info("Create directories:")
        self.__create_directory(self.directory)
        LOGGER.info("|    Working directory: %s", self.directory)
        directory = os.path.join(self.directory, POST_DIRECTORY)
        self.__create_directory(directory)
        LOGGER.info("|    Post-processing: %s", directory)
        directory = os.path.join(self.directory, POSTOPTIM_DIRECTORY)
        self.__create_directory(directory)
        LOGGER.info("|    Optimization history view: %s", directory)
        directory = os.path.join(self.directory, POSTSTUDY_DIRECTORY)
        self.__create_directory(directory)
        LOGGER.info("|    Scalability views: %s", directory)
        directory = os.path.join(self.directory, POSTSCAL_DIRECTORY)
        self.__create_directory(directory)
        LOGGER.info("|    Dependency matrices: %s", directory)
        directory = os.path.join(self.directory, RESULTS_DIRECTORY)
        self.__create_directory(directory)
        LOGGER.info("|    Results: %s", directory)

    @staticmethod
    def __create_directory(path):
        """Create a directory from its relative or absolute path.

        :param str path: relative or absolute directory path.
        """
        if not os.path.exists(path):
            os.mkdir(path)

    def add_discipline(self, name, data):
        """This method adds a disciplinary dataset from:

        - its :code:`name`,
        - its :code:`data`,

        :param str name: name of the discipline.
        :param AbstractFullCache data: dataset provided as a cache.
        """
        self._group_dep[name] = {}
        self._all_data = data.get_all_data()
        datum = self._all_data[1]
        outputs_names = [str(fct_name) for fct_name in list(datum["outputs"].keys())]
        outputs_sizes = [
            str(fct_name) + "(" + str(len(fct_val)) + ")"
            for fct_name, fct_val in datum["outputs"].items()
        ]
        inputs_sizes = [
            str(fct_name) + "(" + str(len(fct_val)) + ")"
            for fct_name, fct_val in datum["inputs"].items()
        ]
        self.datasets.append(data)
        for output_name in outputs_names:
            self.set_fill_factor(name, output_name, self._default_fill_factor)
        LOGGER.info("Add a discipline:")
        LOGGER.info("|    Name: %s", name)
        LOGGER.info("|    Inputs: %s", ", ".join(inputs_sizes))
        LOGGER.info("|    Outputs: %s", ", ".join(outputs_sizes))
        LOGGER.info("|    Sample size: %s", data.get_length())
        LOGGER.info("|    Cache: %s", data.name)

    @property
    def disciplines_names(self):
        """Get discipline names.

        :return: list of discipline names
        :rtype: list(str)
        """
        disc_names = [discipline.name for discipline in self.datasets]
        return disc_names

    def set_input_output_dependency(self, discipline, output, inputs):
        """Set the dependency between an output and a set of inputs
        for a given discipline.

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
        """Check if discipline is a string
        comprised in the list of disciplines names.
        """
        if not isinstance(discipline, basestring):
            raise TypeError("The argument discipline should be a string")
        disciplines_names = self.disciplines_names
        if discipline not in disciplines_names:
            raise ValueError(
                "The argument discipline should be a string " "comprised in the list ",
                disciplines_names,
            )

    def __check_output(self, discipline, output):
        """ Check if output is an output function of discipline. """
        self.__check_discipline(discipline)
        if not isinstance(output, basestring):
            raise TypeError("The argument output should be a string")
        outputs_names = next(
            cache.outputs_names for cache in self.datasets if cache.name == discipline
        )
        if output not in outputs_names:
            raise ValueError(
                "The argument output should be a string " "comprised in the list ",
                outputs_names,
            )

    def __check_inputs(self, discipline, inputs):
        """ Check if inputs is a list of inputs of discipline. """
        self.__check_discipline(discipline)
        inputs_names = next(
            cache.inputs_names for cache in self.datasets if cache.name == discipline
        )
        if not isinstance(inputs, list):
            raise TypeError("The argument inputs must be a list of string.")
        for inpt in inputs:
            self.__check_input(inpt, inputs_names)

    @staticmethod
    def __check_input(inpt, inputs_names):
        """ Check if input is a string comprised in inputs_names. """
        if not isinstance(inpt, basestring):
            raise TypeError("The argument inpt must be a string.")
        if inpt not in inputs_names:
            raise ValueError(
                "The argument inpt must be a string " "comprised in the list ",
                inputs_names,
            )

    def __check_fill_factor(self, fill_factor):
        """Check if fill factor is a proportion or a number equal to -1

        :param float fill_factor: a proportion or -1
        """
        try:
            self.__check_proportion(fill_factor)
        except ValueError:
            if fill_factor != -1:
                raise TypeError(
                    "Fill factor should be "
                    "a float number comprised in 0 and 1 "
                    "or a number equal to -1."
                )

    @staticmethod
    def __check_proportion(proportion):
        """Check if a proportion is a float number comprised in 0 and 1.

        :param float proportion: proportion comprised in 0 and 1.
        """
        if not isinstance(proportion, numbers.Number):
            raise TypeError(
                "A proportion should be " "a float number comprised in 0 and 1."
            )
        if proportion < 0 or proportion > 1:
            raise ValueError(
                "A proportion should be " "a float number comprised in 0 and 1."
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
        """Add both optimization algorithm and MDO formulation,
        as well as their options.

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

    def print_optimization_strategies(self):
        """ Print the list of the optimization strategies. """
        index = 0
        print("Optimization strategies:")
        for algo, formulation in zip(self.algorithms, self.formulations):
            template = "# {} - Algorithm: {} {} - Formulation: {} {}"
            form_opt = self.formulations_options[index]
            form_opt = "" if form_opt is None else form_opt
            algo_opt = self.algorithms_options[index]
            algo_opt = "" if algo_opt is None else algo_opt
            index += 1
            print(template.format(index, algo, algo_opt, formulation, form_opt))

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
            if variables is not None:
                for varname, value in variables[idx].items():
                    self.__update_var_scaling(var_scaling, value, [varname])
                    scaling[varname] = value
            self.var_scalings.append(var_scaling)
            self.scalings.append(scaling)

    @staticmethod
    def __format_scaling(size, n_scaling):
        """Convert a scaling size in a list of integers
        whose length is equal to the number of scalings.

        :param size: size(s) of a given variable
        :type size: int or list(int)
        :param int n_scaling: number of scalings
        :return: formatted sizes
        """
        formatted_sizes = size
        if isinstance(formatted_sizes, int) or formatted_sizes is None:
            formatted_sizes = [formatted_sizes]
        if len(formatted_sizes) == 1:
            formatted_sizes = formatted_sizes * n_scaling
        return formatted_sizes

    @staticmethod
    def __update_var_scaling(scaling, size, varnames):
        """Update a scaling dictionary for a given list of variables
        and a given size.

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
        """Check that for the different types of variables,
        the number of scalings is the same or equal to 1.

        :param int n_var_scaling: number of scalings
        :param int n_scaling: expected number of scalings
        """
        assert n_var_scaling in (n_scaling, 1)

    @staticmethod
    def __check_varsizes_type(varsizes):
        """Check the type of scaling sizes. Integer, list of integers
        or None is expected. Return the number of scalings.

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

    def set_early_stopping(self):
        """Enable early stopping."""
        self.early_stopping = True

    def unset_early_stopping(self):
        """Disable early stopping."""
        self.early_stopping = False

    def print_scaling_strategies(self):
        """ Print the list of the scaling strategies. """
        index = 0
        print("Scaling strategies:")
        for scaling in self.scalings:
            index += 1
            msg = "# " + str(index)
            for var_type, var_size in scaling.items():
                msg += " - "
                msg += var_type + ": " + str(var_size)
            print(msg)

    def execute(self, n_replicates=1):
        """Execute the scalability study, one or several times to take into
        account the random features of the scalable problems.

        :param int n_replicates: number of times the scalability study
            is repeated. Default: 1.
        """
        LOGGER.info("Execute scalability study %s times", n_replicates)
        if not self.formulations and not self.algorithms:
            raise ValueError(
                "A scalable study needs at least "
                "1 optimization strategy, "
                "defined by a mandatory optimization algorithm, "
                "and optional optimization algorithm, "
                "optimization options"
            )
        index = 0
        for algo, formulation in zip(self.algorithms, self.formulations):
            for id_scaling, scaling in enumerate(self.var_scalings):
                for replicate in range(1, n_replicates + 1):
                    problem = ScalableProblem(
                        self.datasets,
                        self.design_variables,
                        self.objective,
                        self.eq_constraints,
                        self.ineq_constraints,
                        self.maximize_objective,
                        scaling,
                        fill_factor=self._fill_factor,
                        seed=replicate,
                        group_dep=self._group_dep,
                        force_input_dependency=True,
                        allow_unused_inputs=False,
                    )
                    varnames = [algo, formulation, id_scaling + 1, replicate]
                    name = "_".join([self.prefix] + [str(var) for var in varnames])
                    if name[0] == "_":
                        name = name[1:]
                    directory = os.path.join(self.directory, POSTSCAL_DIRECTORY, name)
                    self.__create_directory(directory)
                    problem.plot_dependencies(True, False, directory)
                    form_opt = self.formulations_options[index]
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
                        **formulation_options
                    )
                    top_level_disciplines = (
                        problem.scenario.formulation.get_top_level_disc()
                    )
                    for disc in top_level_disciplines:
                        disc.linearization_mode = self.top_level_diff[index]
                    algo_options = deepcopy(self.algorithms_options[index])
                    max_iter = algo_options["max_iter"]
                    del algo_options["max_iter"]
                    problem.scenario.execute(
                        {
                            "algo": algo,
                            "max_iter": max_iter,
                            "algo_options": algo_options,
                        }
                    )
                    stopidx, n_iter = self.__get_stop_index(problem)
                    ratio = float(stopidx) / n_iter
                    self.__save_opt_history_view(
                        problem,
                        algo + "_" + formulation,
                        str(id_scaling + 1),
                        str(replicate),
                    )
                    n_calls = {
                        disc: n_calls * ratio
                        for disc, n_calls in problem.n_calls.items()
                    }
                    tmp = problem.n_calls_linearize
                    n_calls_linearize = {disc: ncl * ratio for disc, ncl in tmp.items()}
                    tmp = problem.n_calls_top_level
                    n_calls_tl = {
                        disc: n_calls * ratio for disc, n_calls in tmp.items()
                    }
                    tmp = problem.n_calls_linearize_top_level
                    n_calls_linearize_tl = {
                        disc: ncltl * ratio for disc, ncltl in tmp.items()
                    }
                    exec_time = problem.exec_time()
                    exec_time *= ratio
                    status = problem.status
                    is_feasible = problem.is_feasible
                    result = ScalabilityResult(name, id_scaling + 1, replicate)
                    self.results.append(result)
                    inputs = problem.inputs
                    outputs = problem.outputs
                    disc_varnames = {
                        disc: inputs[disc] + outputs[disc]
                        for disc in problem.disciplines
                    }
                    sizes = problem.varsizes
                    new_varsizes = {
                        disc: {
                            name: scaling.get(name, sizes[name])
                            for name in disc_varnames[disc]
                        }
                        for disc in problem.disciplines
                    }
                    old_varsizes = problem.varsizes
                    result.get(
                        algo,
                        algo_options,
                        formulation,
                        formulation_options,
                        scaling,
                        n_calls,
                        n_calls_linearize,
                        n_calls_tl,
                        n_calls_linearize_tl,
                        exec_time,
                        status,
                        is_feasible,
                        problem.disciplines,
                        problem.outputs,
                        old_varsizes,
                        new_varsizes,
                    )
                    result.save(self.directory)
            index += 1
        return self.results

    def __save_opt_history_view(self, problem, optim, scaling, replicate):
        """Save optimization history view
        according to a directory tree structure.

        :param ScalableProblem problem: scalable problem
        :param str optim: optimization algorithm and MDO formulation
        :param str scaling: scaling index
        :param str replicate: replicate index
        """
        LOGGER.info("Save optimization history view")
        fpath = os.path.join(
            self.directory, POSTOPTIM_DIRECTORY, "_".join([self.prefix] + [optim])
        )
        self.__create_directory(fpath)
        fpath = os.path.join(fpath, "scaling_" + scaling)
        self.__create_directory(fpath)
        fpath = os.path.join(fpath, "replicate_" + replicate)
        self.__create_directory(fpath)
        fpath = fpath + "/"
        problem.scenario.post_process(
            "OptHistoryView", save=True, show=False, file_path=fpath
        )

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
                diff = abs(y_prev - value[pbm.get_objective_name()])
                diff /= abs(y_prev)
                if diff < 1e-6:
                    break
                y_prev = value[pbm.get_objective_name()]
                stopidx += 1
        else:
            stopidx = n_iter
        return stopidx, n_iter


class ScalabilityResult(object):
    """Scalability Result."""

    def __init__(self, name, id_scaling, id_sample):
        """Constructor.

        :param str name: name of the scalability result.
        :param int id_scaling: scaling identifiant
        :param int id_sample: sample identifiant
        """
        self.name = name
        self.id_scaling = id_scaling
        self.id_sample = id_sample
        self.algo = None
        self.algo_options = None
        self.formulation_options = None
        self.formulation = None
        self.scaling = None
        self.n_calls = None
        self.n_calls_linearize = None
        self.n_calls_top_level = None
        self.n_calls_linearize_top_level = None
        self.exec_time = None
        self.original_exec_time = None
        self.status = None
        self.is_feasible = None
        self.disc_names = None
        self.old_varsizes = None
        self.new_varsizes = None
        self.output_names = None

    def get(
        self,
        algo,
        algo_options,
        formulation,
        formulation_options,
        scaling,
        n_calls,
        n_calls_linearize,
        n_calls_top_level,
        n_calls_linearize_top_level,
        exec_time,
        status,
        is_feasible,
        disc_names,
        output_names,
        old_varsizes,
        new_varsizes,
    ):
        """Get a scalability result for a given optimization strategy
        and a given scaling strategy.

        :param str algo: name of the optimization algorithm
        :param dict algo_options: options of the optimization algorithm
        :param str formulation: name of the MDO formulation
        :param dict formulation_options: options of the MDO formulation
        :param scaling: scaling strategy
        :param list(int) n_calls: number of calls for each discipline
        :param list(int) n_calls_linearize: number of linearization
            for each discipline
        :param list(int) n_calls_top_level: number of calls for each discipline
        :param list(int) n_calls_linearize_top_level: number of linearization
            for each discipline
        :param float exec_time: execution time
        :param int status: status of the optimization scenario
        :param bool is_feasible: feasibility of the optimization solution
        :param list(str) disc_names: list of discipline names
        :param dict output_names: list of output names
        :param dict old_varsizes: old variable sizes
        :param dict new_varsizes: new variable sizes
        """
        self.algo = algo
        self.algo_options = algo_options
        self.formulation = formulation
        self.formulation_options = formulation_options
        self.scaling = scaling
        self.n_calls = n_calls
        self.n_calls_linearize = n_calls_linearize
        self.n_calls_top_level = n_calls_top_level
        self.n_calls_linearize_top_level = n_calls_linearize_top_level
        self.exec_time = exec_time
        self.status = status
        self.is_feasible = is_feasible
        self.disc_names = disc_names
        self.output_names = output_names
        self.old_varsizes = old_varsizes
        self.new_varsizes = new_varsizes

    def save(self, study_directory):
        """Save a scalability result into a pickle file
        whose name is the name of the ScalabilityResult instance."""
        fpath = os.path.join(study_directory, RESULTS_DIRECTORY, self.name + ".pkl")
        result = {
            "algo": self.algo,
            "algo_options": self.algo_options,
            "formulation": self.formulation,
            "formulation_options": self.formulation_options,
            "scaling": self.scaling,
            "n_calls": self.n_calls,
            "n_calls_linearize": self.n_calls_linearize,
            "n_calls_top_level": self.n_calls_top_level,
            "n_calls_linearize_top_level": self.n_calls_linearize_top_level,
            "exec_time": self.exec_time,
            "status": self.status,
            "is_feasible": self.is_feasible,
            "disc_names": self.disc_names,
            "output_names": self.output_names,
            "old_varsizes": self.old_varsizes,
            "new_varsizes": self.new_varsizes,
        }
        with open(fpath, "wb") as fout:
            pickle.dump(result, fout)

    def load(self, study_directory):
        """Load a scalability result from a pickle file
        whose name is the name of the ScalabilityResult instance."""
        fpath = os.path.join(study_directory, RESULTS_DIRECTORY, self.name + ".pkl")
        with open(fpath, "rb") as fin:
            result = pickle.load(fin)
        self.algo = result["algo"]
        self.formulation = result["formulation"]
        self.scaling = result["scaling"]
        self.n_calls = result["n_calls"]
        self.n_calls_linearize = result["n_calls_linearize"]
        self.n_calls_top_level = result["n_calls_top_level"]
        self.n_calls_linearize_top_level = result["n_calls_linearize_top_level"]
        self.exec_time = result["exec_time"]
        self.status = result["status"]
        self.is_feasible = result["is_feasible"]
        self.disc_names = result["disc_names"]
        self.output_names = result["output_names"]
        self.old_varsizes = result["old_varsizes"]
        self.new_varsizes = result["new_varsizes"]


class PostScalabilityStudy(object):

    """The PostScalabilityStudy class aims to post-process a list of
    scalability results stored in a directory.
    """

    NOMENCLATURE = {
        "exec_time": "Execution time (s)",
        "original_exec_time": "Pseudo-original execution time",
        "n_calls": "Number of discipline evaluations",
        "n_calls_linearize": "Number of gradient evaluations",
        "status": "Optimizatin status",
        "is_feasible": "Feasibility of the solution (0 or 1)",
        "scaling_strategy": "Scaling strategy index",
        "total_calls": "Total number of evaluations",
    }

    def __init__(self, study_directory):
        """Constructor.

        :param str study: directory of the scalability study."
        """
        LOGGER.info("*** Start post-processing for scalability study ***")
        LOGGER.info("Working directory: %s", study_directory)
        self.study_directory = study_directory
        self.scalability_results = self.__load_results()
        self.n_results = len(self.scalability_results)
        self.descriptions = self.NOMENCLATURE
        self.cost_function = {}
        self.unit_cost = None
        for result in self.scalability_results:
            result.total_calls = sum(result.n_calls.values())
            result.total_calls += sum(result.n_calls_linearize.values())

    def set_cost_function(self, formulation, cost):
        """Set cost function for each formulation.

        :param str formulation: name of the formulation.
        :param function cost: cost function
        """
        self.cost_function[formulation] = cost

    def set_cost_unit(self, cost_unit):
        """Set the measurement unit for cost evaluation.

        :param str cost_unit: cost unit, e.g. 'h', 'min', ...
        """
        self.unit_cost = cost_unit
        desc = self.descriptions["original_exec_time"]
        desc = desc.split(" (")[0]
        desc += " (" + cost_unit + ")"
        self.descriptions["original_exec_time"] = desc

    def labelize_exec_time(self, description):
        """Change the description of execution time,
        used in plotting methods notably.

        :param str description: description.
        """
        self._update_descriptions("exec_time", description)

    def labelize_original_exec_time(self, description):
        """Change the description of original execution time,
        used in plotting methods notably.

        :param str description: description.
        """
        self._update_descriptions("original_exec_time", description)
        if self.unit_cost is not None:
            self.set_cost_unit(self.unit_cost)

    def labelize_n_calls(self, description):
        """Change the description of number of calls,
        used in plotting methods notably.

        :param str description: description.
        """
        self._update_descriptions("n_calls", description)

    def labelize_n_calls_linearize(self, description):
        """Change the description of number of calls for linearization,
        used in plotting methods notably.

        :param str description: description.
        """
        self._update_descriptions("n_calls_linearize", description)

    def labelize_status(self, description):
        """Change the description of status,
        used in plotting methods notably.

        :param str description: description.
        """
        self._update_descriptions("status", description)

    def labelize_is_feasible(self, description):
        """Change the description of feasibility,
        used in plotting methods notably.

        :param str description: description.
        """
        self._update_descriptions("is_feasible", description)

    def labelize_scaling_strategy(self, description):
        """Change the description of scaling strategy,
        used in plotting methods notably.

        :param str description: description.
        """
        self._update_descriptions("scaling_strategy", description)

    def _update_descriptions(self, keyword, description):
        """Update the description initialized
        with the NOMENCLATURE class attribute.

        :param str keyword: keyword of the considered object.
        :param str description: new description
        """
        if not self.descriptions.get(keyword):
            raise ValueError(
                "The keyword " + keyword + " is not "
                "in the list: " + ", ".join(list(self.descriptions.keys()))
            )
        if not isinstance(description, basestring):
            raise TypeError(
                'The argument "description" must be '
                "of type string, "
                "not of type " + description.__class__.__name__
            )
        self.descriptions[keyword] = description

    def __load_results(self):
        """ Load results from the results directory of the study path."""
        if not os.path.isdir(self.study_directory):
            raise ValueError('Directory "' + self.study_directory + '" does not exist.')
        directory = os.path.join(self.study_directory, RESULTS_DIRECTORY)
        if not os.path.isdir(directory):
            raise ValueError('Directory "' + directory + '" does not exist.')
        filenames = [
            filename
            for filename in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, filename))
        ]
        results = []
        for filename in filenames:
            name = filename.split(".")[0]
            id_scaling = int(name.split("_")[-2])
            id_sample = int(name.split("_")[-1])
            result = ScalabilityResult(name, id_scaling, id_sample)
            result.load(self.study_directory)
            results.append(result)
        if not results:
            raise ValueError("Directory " + directory + " is empty.")
        return results

    def plot(
        self,
        legend_loc="upper left",
        xticks=None,
        xticks_labels=None,
        xmargin=0.0,
        **options
    ):
        """Plot the results using different methods
        according to the presence or absence of replicate values.

        :param str legend_loc: legend localization
        :param list(float) xticks: list of xticks (default: None)
        :param list(str) xticks_labels: list of xticks labels (default: None)
        :param float xmargin: margin on left and right sides of the x-axis
        :param options: options for the specialized plot methods
        """
        LOGGER.info("Execute post-processing")
        self.__has_scaling_dimension(xticks)
        self.__has_scaling_dimension(xticks_labels)
        there_are_replicates = len(self.get_replicates(True)) > 1
        if self.cost_function:
            self._estimate_original_time()
        if there_are_replicates:
            LOGGER.info("|    Type: replicate")
            self._replicate_plot(legend_loc, xticks, xticks_labels, xmargin, **options)
        else:
            LOGGER.info("|    Type: standard")
            self._standard_plot(legend_loc, xticks, xticks_labels, xmargin)

    def __has_scaling_dimension(self, value):
        """Assert if a value has the scaling dimension.

        :param ndarray value: value.
        """
        if value is not None and hasattr(value, "__len__"):
            assert len(value) == len(self.get_scaling_strategies(True))

    def _standard_plot(
        self, legend_loc="upper left", xticks=None, xticks_labels=None, xmargin=0.0
    ):
        """Deterministic plot.

        :param str legend_loc: legend localization
        :param list(float) xticks: list of xticks (default: None)
        :param list(str) xticks_labels: list of xticks labels (default: None)
        :param float xmargin: margin on left and right sides of the x-axis
        """
        colors = ["blue", "red", "green"]
        idos = -1
        for optim_strategy in self.get_optimization_strategies(True):
            idos += 1
            color = colors[idos]
            xcoord, values = self.__get_values(optim_strategy)
            for ylabel, ycoord in values.items():
                if hasattr(ycoord, "shape"):
                    plt.figure(ylabel)
                    if len(ycoord.shape) == 3:
                        n_criteria = ycoord.shape[0]
                    else:
                        n_criteria = 1
                    linestyles = [
                        (0, (id_criterion + 1, n_criteria))
                        for id_criterion in range(n_criteria)
                    ]
                    for criterion in range(n_criteria):
                        if len(ycoord.shape) == 3:
                            yycoord = ycoord[criterion, :, 0].flatten()
                        else:
                            yycoord = ycoord[:, 0].flatten()
                        yval = yycoord
                        xval = list(xcoord)
                        if xticks is not None:
                            assert hasattr(xticks, "__len__")
                            assert len(xticks) == len(xval)
                            xval = xticks
                        else:
                            xticks = xval
                        plt.plot(
                            xval,
                            yval,
                            "o",
                            linestyle=linestyles[criterion],
                            color=color,
                        )
                        if xticks_labels is None:
                            xticks_labels = xticks
                        else:
                            xticks_labels = xval
                        plt.xticks(xticks, xticks_labels)
                        plt.xlabel(self.descriptions["scaling_strategy"])
                        plt.ylabel(self.descriptions[ylabel])
                        plt.xlim(xval[0] - xmargin, xval[-1] + xmargin)
        problem_legends = self.get_optimization_strategies(True)
        legend_elements = [
            Line2D([0], [0], color=colors[idx], lw=4, label=problem_legends[idx])
            for idx in range(len(problem_legends))
        ]
        for ylabel, ycoord in values.items():
            if hasattr(ycoord, "shape"):
                plt.figure(ylabel)
                plt.legend(
                    handles=legend_elements,
                    loc=legend_loc,
                    frameon=False,
                    framealpha=0.5,
                )
                fpath = os.path.join(
                    self.study_directory, POSTSTUDY_DIRECTORY, ylabel + ".pdf"
                )
                plt.savefig(fpath)

    @property
    def n_samples(self):
        """ Number of samples """
        return len(self.get_replicates(True))

    def _replicate_plot(
        self,
        legend_loc="upper left",
        xticks=None,
        xticks_labels=None,
        xmargin=0.0,
        minbox=2,
        notch=False,
        widths=0.25,
        whis=1.5,
    ):
        """Probabilistic plot.

        :param str legend_loc: legend localization
        :param list(float) xticks: list of xticks (default: None)
        :param list(str) xticks_labels: list of xticks labels (default: None)
        :param float xmargin: margin on left and right sides of the x-axis
        :param int minbox: minimal number of values for boxplot (default: 2).
        :param bool notch: if True, will produced a notched boxplot.
        :param float whis:  the reach of the whiskers to the beyond
            the first and third quartiles (default: 1.5).
        """
        if not hasattr(widths, "__len__"):
            widths = [widths] * len(self.get_scaling_strategies(True))
        else:
            self.__has_scaling_dimension(widths)

        n_samples = len(self.get_replicates(True))
        colors = ["blue", "red", "green"]
        if n_samples >= minbox:
            idos = -1
            n_strategies = len(self.get_optimization_strategies(True))
            for optim_strategy in self.get_optimization_strategies(True):
                idos += 1
                xcoord, values = self.__get_values(optim_strategy)
                for ylabel, ycoord in values.items():
                    if hasattr(ycoord, "shape") and ylabel != "status":
                        plt.figure(ylabel)
                        xval = list(xcoord)
                        if xticks is not None:
                            assert hasattr(xticks, "__len__")
                            assert len(xticks) == len(xval)
                            xval = xticks
                        tmp = []
                        for idx, xtick in enumerate(xval):
                            tmp.append(float(idos) / n_strategies)
                            tmp[-1] *= widths[idx] * 3
                            tmp[-1] += xtick
                        self.__draw_boxplot(
                            ycoord, tmp, colors[idos], colors[idos], notch, widths, whis
                        )
                        if len(ycoord.shape) == 3:
                            zval = ycoord[0, :, :]
                        else:
                            zval = ycoord
                        xval_offset = [
                            xtick + float(idos) / n_strategies * widths[idx] * 3
                            for idx, xtick in enumerate(xval)
                        ]
                        plt.plot(xval_offset, median(zval, 1), "--", color=colors[idos])
                        if xticks_labels is None:
                            xticks_labels = xval
                        plt.xticks(xval, xticks_labels)
                        plt.xlim(xval[0] - xmargin, xval[-1] + xmargin)
                        plt.xlabel(self.descriptions["scaling_strategy"])
                        plt.ylabel(self.descriptions[ylabel])
                        plt.yscale("log")
                        plt.grid(True, "both")
        problem_legends = self.get_optimization_strategies(True)
        legend_elements = [
            Line2D([0], [0], color=colors[idx], lw=4, label=problem_legends[idx])
            for idx in range(len(problem_legends))
        ]
        for ylabel, ycoord in values.items():
            if hasattr(ycoord, "shape") and ylabel != "status":
                plt.figure(ylabel)
                plt.legend(
                    handles=legend_elements,
                    loc=legend_loc,
                    frameon=False,
                    framealpha=0.5,
                )
                fpath = os.path.join(
                    self.study_directory, POSTSTUDY_DIRECTORY, ylabel + ".pdf"
                )
                plt.savefig(fpath)

    @staticmethod
    def __draw_boxplot(data, xticks, edge_color, fill_color, notch, widths, whis):
        """Draw boxplot from a dataset.

        :param array data: dataset array of dimension 2 or 3
        :param list(float) xticks: values of xticks
        :param str edge_color: edge color
        :param str fill_color: fill color
        :param bool notch: if True, will produced a notched boxplot.
        :param list(float) widths: widths of boxplots
        :param float whis:  the reach of the whiskers to the beyond
            the first and third quartiles
        """
        if len(data.shape) == 3:
            data = data[0, :, :]
        boxplot = plt.boxplot(
            data.T,
            patch_artist=True,
            positions=xticks,
            showfliers=False,
            whiskerprops={"linestyle": "-"},
            notch=notch,
            widths=widths,
            whis=whis,
        )

        for element in ["boxes", "whiskers", "fliers", "means", "caps"]:
            plt.setp(boxplot[element], color=edge_color)
        plt.setp(boxplot["medians"], color="white")

        for patch in boxplot["boxes"]:
            patch.set(facecolor=fill_color)

        return boxplot

    def __get_values(self, optim_strategy):
        """Get values of criteria and corresponding scaling levels
        for a given optimization strategy and all replicates.

        :param str optim_strategy: name of the optimization strategy.
        :return: scaling levels, criteria values
        :rtype: list(int), dict(array)
        """
        exec_time = []
        original_exec_time = []
        n_calls = []
        n_calls_linearize = []
        status = []
        is_feasible = []
        scaling = []
        total_calls = []
        for replicate in self.get_replicates(True):
            xcoord, ycoord = self.__get_replicate_values(optim_strategy, replicate)
            exec_time.append(ycoord["exec_time"])
            total_calls.append(ycoord["total_calls"])
            if self.cost_function:
                original_exec_time.append(ycoord["original_exec_time"])
            tmp = [
                [val for _, val in ycoord["n_calls"][idx].items()]
                for idx in range(len(ycoord["n_calls"]))
            ]
            n_calls.append(tmp)
            tmp = [
                [val for _, val in ycoord["n_calls_linearize"][idx].items()]
                for idx in range(len(ycoord["n_calls_linearize"]))
            ]
            n_calls_linearize.append(tmp)
            status.append(ycoord["status"])
            is_feasible.append(ycoord["is_feasible"])
            scaling.append(ycoord["scaling"])
        scaling_levels = xcoord
        values = {}
        values["exec_time"] = array(exec_time).T
        if self.cost_function:
            values["original_exec_time"] = array(original_exec_time).T
        values["n_calls"] = array(n_calls).T
        values["n_calls_linearize"] = array(n_calls_linearize).T
        values["total_calls"] = array(total_calls).T
        values["status"] = array(status).T
        values["is_feasible"] = array(is_feasible).T
        values["scaling"] = scaling
        return scaling_levels, values

    def __get_replicate_values(self, optim_strategy, replicate):
        """Get values of criteria and corresponding scaling levels
        for a given optimization strategy and a given replicate.

        :param str optim_strategy: optimization strategy.
        :param int replicate: replicate index.
        :return: scaling levels, criteria values
        :rtype: list(int), dict(array-like)
        """
        are_replicate = [value == replicate for value in self.get_replicates()]
        are_optim_strategy = [
            value == optim_strategy for value in self.get_optimization_strategies()
        ]
        are_ok = [
            is_rep and is_oo for is_rep, is_oo in zip(are_replicate, are_optim_strategy)
        ]
        indices = [index for index, is_ok in enumerate(are_ok) if is_ok]
        scaling_levels = self.get_scaling_strategies()
        scaling_levels = [scaling_levels[index] for index in indices]
        tmp = sorted(list(range(len(scaling_levels))), key=lambda k: scaling_levels[k])
        indices = [indices[index] for index in tmp]
        scaling_levels = [scaling_levels[index] for index in tmp]
        results = [self.scalability_results[index] for index in indices]
        exec_time = [result.exec_time for result in results]
        total_calls = [result.total_calls for result in results]
        original_exec_time = [result.original_exec_time for result in results]
        n_calls = [result.n_calls for result in results]
        n_calls_linearize = [result.n_calls_linearize for result in results]
        status = [result.status for result in results]
        is_feasible = [result.is_feasible for result in results]
        scaling = [result.scaling for result in results]
        values = {}
        values["exec_time"] = exec_time
        values["total_calls"] = total_calls
        if self.cost_function:
            values["original_exec_time"] = original_exec_time
        values["n_calls"] = n_calls
        values["n_calls_linearize"] = n_calls_linearize
        values["status"] = status
        values["is_feasible"] = is_feasible
        values["scaling"] = scaling
        return scaling_levels, values

    @property
    def names(self):
        """ Get the names of the scalability results. """
        return [value.name for value in self.scalability_results]

    def get_optimization_strategies(self, unique=False):
        """Get the names of the optimization strategies.

        :param bool unique: return either unique values if True
            or one value per scalability result if False (default: False).
        :return: names of the optimization stategies.
        :rtype: list(str)
        """
        os_list = ["_".join(name.split("_")[0:-2]) for name in self.names]
        if unique:
            os_list = sorted(set(os_list))
        return os_list

    def get_scaling_strategies(self, unique=False):
        """Get the identificants of the scaling strategies.

        :param bool unique: return either unique values if True
            or one value per scalability result if False (default: False).
        :return: identifiants of scaling strategies
        :rtype: list(int)
        """
        ss_list = [int(name.split("_")[-2]) for name in self.names]
        if unique:
            ss_list = sorted(set(ss_list))
        return ss_list

    def get_replicates(self, unique=False):
        """Get the replicate identifiants.

        :param bool unique: return either unique values if True
            or one value per scalability result if False (default: False).
        :return: identifiants of replicates.
        :rtype: list(int)
        """
        rep = [int(name.split("_")[-1]) for name in self.names]
        if unique:
            rep = sorted(set(rep))
        return rep

    def _estimate_original_time(self):
        """Estimate the original execution time from the number of calls
        and linearizations of the different disciplines and
        top-level disciplines and from the cost functions provided by the
        user.

        :return: original time
        :rtype: float
        """
        for scalability_result in self.scalability_results:
            n_c = scalability_result.n_calls
            n_cl = scalability_result.n_calls_linearize
            n_tl_c = scalability_result.n_calls_top_level
            n_tl_cl = scalability_result.n_calls_linearize_top_level
            varsizes = scalability_result.new_varsizes
            formulation = scalability_result.formulation
            if formulation not in self.cost_function:
                raise ValueError(
                    "The cost function of "
                    + formulation
                    + " must be defined in order to compute "
                    " the estimated original time."
                )
            result = self.cost_function[formulation](
                varsizes, n_c, n_cl, n_tl_c, n_tl_cl
            )
            scalability_result.original_exec_time = result
