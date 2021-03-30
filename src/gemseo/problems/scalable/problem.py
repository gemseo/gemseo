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
Scalable MDO problem
====================

This module implements the concept of scalable problem by means of the
:class:`.ScalableProblem` class.

Given

- a MDO scenario based on a set of sampled disciplines
  with a particular problem dimension,
- a new problem dimension (= number of inputs and outputs),

a scalable problem:

1. makes each discipline scalable based on the new problem dimension,
2. creates the corresponding MDO scenario.

Then, this MDO scenario can be executed and post-processed.

We can repeat this tasks for different sizes of variables
and compare the scalability, which is the dependence of the scenario results
on the problem dimension.

.. seealso:: :class:`.MDODiscipline`, :class:`.ScalableDiscipline`
   and :class:`.Scenario`
"""
from __future__ import absolute_import, division, unicode_literals

import os
from copy import deepcopy

from future import standard_library
from numpy import array, ones, random, where, zeros

from gemseo import LOGGER
from gemseo.algos.design_space import DesignSpace
from gemseo.api import (
    create_design_space,
    create_scenario,
    generate_coupling_graph,
    generate_n2_plot,
    get_all_inputs,
)
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.mda.mda_factory import MDAFactory
from gemseo.problems.scalable.discipline import ScalableDiscipline

standard_library.install_aliases()


class ScalableProblem(object):
    """ Scalable problem. """

    def __init__(
        self,
        datasets,
        design_variables,
        objective_function,
        eq_constraints=None,
        ineq_constraints=None,
        maximize_objective=False,
        sizes=None,
        **parameters
    ):
        """Constructor

        :param list(AbstractFullCache) datasets: disciplinary datasets.
        :param list(str) design_variables: list of design variable names
        :param str objective_function: objective function
        :param list(str) eq_constraints: equality constraints. Default: None.
        :param list(str) eq_constraints: inequality constraints. Default: None.
        :param bool maximize_objective: maximize objective. Default: False.
        :param dict sizes: sizes of input and output variables.
            If None, use the original sizes. Default: None.
        :param parameters: optional parameters for the scalable model.
        """
        self.disciplines = [cache.name for cache in datasets]
        self.data = {cache.name: cache for cache in datasets}
        self.inputs = {cache.name: cache.inputs_names for cache in datasets}
        self.outputs = {cache.name: cache.outputs_names for cache in datasets}
        self.varsizes = {}
        for cache in datasets:
            self.varsizes.update(cache.varsizes)
        self.design_variables = design_variables
        self.objective_function = objective_function
        self.ineq_constraints = ineq_constraints
        self.eq_constraints = eq_constraints
        self.maximize_objective = maximize_objective
        self.scaled_disciplines = []
        self.scaled_sizes = {}
        self._build_scalable_disciplines(sizes, **parameters)
        self.scenario = None

    def __str__(self):
        """String representation of information about the scalable problem

        :return: scalable problem description
        :rtype: str
        """
        tmp = "MDO problem\n"
        tmp += "| Disciplines: " + ", ".join(self.disciplines) + "\n"
        if self.design_variables is not None:
            tmp += "| Design variables: " + ", ".join(self.design_variables) + "\n"

        tmp += "| Objective function: " + self.objective_function
        if self.maximize_objective:
            tmp += " (to maximize) \n"
        else:
            tmp += " (to minimize) \n"
        if self.ineq_constraints is not None:
            tmp += (
                "| Inequality constraints: " + ", ".join(self.ineq_constraints) + "\n"
            )
        if self.eq_constraints is not None:
            tmp += "| Equality constraints: " + ", ".join(self.eq_constraints) + "\n"
        sizes = [
            name + " (" + str(size) + ")" for name, size in self.scaled_sizes.items()
        ]
        tmp += "| Sizes: " + ", ".join(sizes) + "\n"
        return tmp

    def plot_n2_chart(self, save=True, show=False):
        """Plot a N2 chart

        :param bool save: save plot. Default: True.
        :param bool show: show plot. Default: False.
        """
        generate_n2_plot(self.scaled_disciplines, save=save, show=show)

    def plot_coupling_graph(self):
        """ Plot a coupling graph """
        generate_coupling_graph(self.scaled_disciplines)
        os.remove("coupling_graph.gv")

    def plot_1d_interpolations(
        self, save=True, show=False, step=0.01, varnames=None, directory=".", png=False
    ):
        """Plot 1d interpolations

        :param bool save: save plot. Default: True.
        :param bool show: show plot. Default: False.
        :param bool step: Step to evaluate the 1d interpolation function
            Default: 0.01.
        :param list(str) varnames: names of the variable to plot;
            if None, all variables are plotted. Default: None.
        :param str directory: directory path. Default: '.'.
        :param bool png: if True, the file format is PNG. Otherwise, use PDF.
            Default: False.
        """
        if not os.path.exists(directory):
            os.mkdir(directory)
        allfnames = []
        for scalable_discipline in self.scaled_disciplines:
            func = scalable_discipline.scalable_model.plot_1d_interpolations
            fnames = func(save, show, step, varnames, directory, png)
            allfnames = allfnames + [os.path.join(directory, fname) for fname in fnames]
        return allfnames

    def plot_dependencies(self, save=True, show=False, directory="."):
        """Plot dependency matrices

        :param bool save: save plot (default: True)
        :param bool show: show plot (default: False)
        :param str directory: directory path (default: '.')
        """
        fnames = []
        for scalable_discipline in self.scaled_disciplines:
            scalable_model = scalable_discipline.scalable_model
            plot_dependency = scalable_model.plot_dependency
            fname = plot_dependency(
                add_levels=True, save=save, show=show, directory=directory
            )
            fnames.append(fname)
        return fnames

    def _build_scalable_disciplines(self, sizes=None, **parameters):
        """Build scalable disciplines.

        :param dict sizes: dictionary whose keys are variable names
            and variables sizes.
        :param parameters: options.
        """
        copied_parameters = deepcopy(parameters)
        for disc in self.disciplines:
            varnames = self.inputs[disc] + self.outputs[disc]
            sizes = sizes or {}
            new_varsizes = {
                varname: sizes.get(varname, self.varsizes[varname])
                for varname in varnames
            }
            if "group_dep" in parameters:
                copied_parameters["group_dep"] = parameters["group_dep"][disc]
            self.scaled_disciplines.append(
                ScalableDiscipline(
                    "ScalableDiagonalModel",
                    self.data[disc],
                    new_varsizes,
                    **copied_parameters
                )
            )
            self.scaled_sizes.update(deepcopy(new_varsizes))

    def create_scenario(
        self,
        formulation="DisciplinaryOpt",
        scenario_type="MDO",
        start_at_equilibrium=False,
        active_probability=0.1,
        feasibility_level=0.5,
        **options
    ):
        """Create MDO scenario from the scalable disciplines

        :param str formulation: MDO formulation. Default: 'DisciplinaryOpt'.
        :param str scenario_type: type of scenario ('MDO' or 'DOE').
            Default: 'MDO'.
        :param bool start_at_equilibrium: start at equilibrium
            using a preliminary MDA. Default: True.
        :param float active_probability: probability to set the inequality
            constraints as active at initial step of the optimization.
            Default: 0.1.
        :param float feasibility_level: offset of satisfaction for inequality
            constraints. Default: 0.5.
        :param options: formulation options.
        """
        equilibrium = None
        if start_at_equilibrium:
            equilibrium = self.__get_equilibrium()
        disciplines = self.scaled_disciplines
        design_space = self._create_design_space(disciplines, formulation)
        max_obj = self.maximize_objective
        if formulation == "BiLevel":
            self.scenario = self._create_bilevel_scenario(disciplines, **options)
        else:
            self.scenario = create_scenario(
                disciplines,
                formulation,
                self.objective_function,
                deepcopy(design_space),
                scenario_type=scenario_type,
                maximize_objective=max_obj,
                **options
            )
        equilibrium = {} if not isinstance(equilibrium, dict) else equilibrium
        self.__add_ineq_constraints(active_probability, feasibility_level, equilibrium)
        self.__add_eq_constraints(equilibrium)
        return self.scenario

    def _create_bilevel_scenario(self, disciplines, **sub_scenario_options):
        """Create a bilevel scenario from disciplines

        :param list(MDODiscipline) disciplines: list of MDODiscipline
        """
        cpl_structure = MDOCouplingStructure(disciplines)
        st_cpl_disciplines = cpl_structure.strongly_coupled_disciplines()
        wk_cpl_disciplines = cpl_structure.weakly_coupled_disciplines()
        obj = self.objective_function
        max_obj = self.maximize_objective

        # Construction of the subsystem scenarios
        sub_scenarios = []
        sub_inputs = []
        for discipline in st_cpl_disciplines:
            cplt_disciplines = list(set(disciplines) - set([discipline]))
            sub_disciplines = [discipline] + wk_cpl_disciplines
            design_space = DesignSpace()
            inputs = get_all_inputs([discipline])
            all_inputs = get_all_inputs(cplt_disciplines)
            inputs = list(set(inputs) - set(all_inputs))
            sub_inputs += inputs
            for name in inputs:
                design_space.add_variable(
                    name, self.scaled_sizes[name], "float", 0.0, 1.0, 0.5
                )
            sub_scenarios.append(
                create_scenario(
                    disciplines=sub_disciplines,
                    formulation="DisciplinaryOpt",
                    objective_name=obj,
                    design_space=design_space,
                    maximize_objective=max_obj,
                )
            )
            sub_scenarios[-1].default_inputs = sub_scenario_options

        # Construction of the system scenario
        all_inputs = get_all_inputs(disciplines)
        inputs = list(set(all_inputs) - set(sub_inputs))
        design_space = DesignSpace()
        for name in inputs:
            design_space.add_variable(
                name, self.scaled_sizes[name], "float", 0.0, 1.0, 0.5
            )
        sub_disciplines = sub_scenarios + wk_cpl_disciplines
        system_scenario = create_scenario(
            disciplines=sub_disciplines,
            formulation="BiLevel",
            objective_name=obj,
            design_space=design_space,
            maximize_objective=max_obj,
            mda_name="MDAJacobi",
            tolerance=1e-8,
        )
        return system_scenario

    def _create_design_space(self, disciplines=None, formulation="DisciplinaryOpt"):
        """Create a design space into the unit hypercube.

        :param list(MDODiscipline) disciplines: list of MDODiscipline
        :param str formulation: MDO formulation (default: 'DisciplinaryOpt')
        """
        design_space = create_design_space()
        for varname in self.design_variables:
            size = self.scaled_sizes[varname]
            l_b = zeros(size)
            u_b = ones(size)
            value = 0.5 + zeros(size)
            design_space.add_variable(varname, size, "float", l_b, u_b, value)
        if formulation == "IDF":
            coupling_structure = MDOCouplingStructure(disciplines)
            all_couplings = set(coupling_structure.get_all_couplings())
            for varname in all_couplings:
                size = self.scaled_sizes[varname]
                design_space.add_variable(
                    varname, size, "float", zeros(size), ones(size), 0.5 + zeros(size)
                )
        return design_space

    def __get_equilibrium(self, mda_name="MDAJacobi", **options):
        """Get the equilibrium point from a MDA method.

        :param str mda_name: MDA name (default: 'MDAJacobi')
        :return: equilibrium point
        :rtype: dict
        """
        LOGGER.info("Build a preliminary MDA to start at equilibrium")
        factory = MDAFactory()
        mda = factory.create(mda_name, self.scaled_disciplines, **options)
        return mda.execute()

    def __add_ineq_constraints(
        self, active_probability, feasibility_level, equilibrium
    ):
        """Add inequality constraints.

        :param float active_probability: probability to set the inequality
            constraints as active at initial step of the optimization
        :param float feasibility_level: offset of satisfaction for inequality
            constraints
        :param dict equilibrium: starting point at equilibrium
        """
        if not hasattr(feasibility_level, "__len__"):
            feasibility_level = {
                constraint: feasibility_level for constraint in self.ineq_constraints
            }
        for constraint, alphai in feasibility_level.items():
            if constraint in list(equilibrium.keys()):
                sample = random.rand(len(equilibrium[constraint]))
                val = equilibrium[constraint]
                taui = where(
                    sample < active_probability, val, alphai + (1 - alphai) * val
                )
            else:
                taui = 0.0
            self.scenario.add_constraint(constraint, "ineq", value=taui)

    def __add_eq_constraints(self, equilibrium):
        """Add equality constraints.

        :param dict equilibrium: starting point at equilibrium
        """
        for constraint in self.eq_constraints:
            cstr_value = equilibrium.get(constraint, array([0.0]))[0]
            self.scenario.add_constraint(constraint, "eq", value=cstr_value)

    def exec_time(self, do_sum=True):
        """Get total execution time per discipline

        :param bool do_sum: sum over disciplines (default: True)
        :return: execution time
        :rtype: list(float) or float
        """
        exec_time = [discipline.exec_time for discipline in self.scenario.disciplines]
        if do_sum:
            exec_time = sum(exec_time)
        return exec_time

    @property
    def n_calls_top_level(self):
        """Get number of top level disciplinary calls per discipline.

        :return: number of top level disciplinary calls per discipline
        :rtype: list(int) or int
        """
        disciplines = self.scenario.formulation.get_top_level_disc()
        n_calls = {discipline.name: discipline.n_calls for discipline in disciplines}
        return n_calls

    @property
    def n_calls_linearize_top_level(self):
        """Get number of top level disciplinary calls per discipline

        :return: number of top level disciplinary calls per discipline
        :rtype: list(int) or int
        """
        disciplines = self.scenario.formulation.get_top_level_disc()
        n_calls = {
            discipline.name: discipline.n_calls_linearize for discipline in disciplines
        }
        return n_calls

    @property
    def n_calls(self):
        """Get number of disciplinary calls per discipline

        :return: number of disciplinary calls per discipline
        :rtype: list(int) or int
        """
        n_calls = {
            discipline.name: discipline.n_calls
            for discipline in self.scenario.disciplines
        }
        return n_calls

    @property
    def n_calls_linearize(self):
        """Get number of disciplinary calls per discipline

        :return: number of disciplinary calls per discipline
        :rtype: list(int) or int
        """
        tmp = {
            discipline.name: discipline.n_calls_linearize
            for discipline in self.scenario.disciplines
        }
        return tmp

    @property
    def status(self):
        """ Get the status of the scenario """
        return self.scenario.optimization_result.status

    @property
    def is_feasible(self):
        """ Get the feasibility property of the scenario """
        return self.scenario.optimization_result.is_feasible
