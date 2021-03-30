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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
A Bi level formulation
**********************
"""
from __future__ import absolute_import, division, unicode_literals

from builtins import str, super

from future import standard_library

from gemseo.core.chain import MDOChain, MDOParallelChain
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.formulation import MDOFormulation
from gemseo.core.function import MDOFunction
from gemseo.core.mdo_scenario import MDOScenarioAdapter
from gemseo.mda.mda_factory import MDAFactory

standard_library.install_aliases()


from gemseo import LOGGER


class BiLevel(MDOFormulation):
    """
    A bi-level formulation draws an optimization architecture
    that involves multiple optimization problems to be solved to
    obtain the solution of the MDO problem.

    Here, at each iteration on the global design variables,
    the bi-level MDO formulation implementation performs a first
    MDA to compute the coupling variables, then disciplinary
    optimizations on the local design variables in parallel and then,
    a second MDA to update the coupling variables.
    """

    def __init__(
        self,
        disciplines,
        objective_name,
        design_space,
        maximize_objective=False,
        mda_name="MDAChain",
        parallel_scenarios=False,
        multithread_scenarios=True,
        apply_cstr_tosub_scenarios=True,
        apply_cstr_to_system=True,
        reset_x0_before_opt=False,
        **mda_options
    ):
        """
        Constructor, initializes the objective functions and constraints

        :param disciplines: the disciplines list.
        :type disciplines: list(MDODiscipline)
        :param objective_name: the objective function data name.
        :type objective_name: str
        :param design_space: the design space.
        :type design_space: DesignSpace
        :param maximize_objective: if True, the objective function
            is maximized, by default, a minimization is performed.
        :type maximize_objective: bool
        :param mda_name: class name of the MDA to be used.
        :type mda_name: str
        :param parallel_scenarios: if True, the sub scenarios are run
            in parallel.
        :type parallel_scenarios: bool
        :param multithread_scenarios: if True and parallel_scenarios=True,
             the sub scenarios are run in parallel using multi-threading,
             if False and parallel_scenarios=True, multi-processing is used.
        :type multithread_scenarios: bool
        :param apply_cstr_tosub_scenarios: if True, the add_constraint
            method adds the constraint to the optimization problem
            of the sub-scenario capable of computing the constraint.
        :type apply_cstr_tosub_scenarios: bool
        :param apply_cstr_to_system: if True, the add_constraint
            method adds the constraint to the optimization problem
            of the system scenario.
        :type apply_cstr_to_system: bool
        :param reset_x0_before_opt: if True, restart the sub optimizations
            from the initial guesses, otherwise warm start them
        :type reset_x0_before_opt: bool
        :param mda_options: options passed to the MDA at construction
        """
        super(BiLevel, self).__init__(
            disciplines,
            objective_name,
            design_space,
            maximize_objective=maximize_objective,
        )
        self._shared_dv = list(design_space.variables_names)
        self.mda1 = None
        self.mda2 = None
        self.reset_x0_before_opt = reset_x0_before_opt
        self.scenario_adapters = []
        self.chain = None
        self._mda_factory = MDAFactory()
        self._apply_cstr_to_system = apply_cstr_to_system
        self._apply_cstr_tosub_scenarios = apply_cstr_tosub_scenarios
        self.__parallel_scenarios = parallel_scenarios
        self._multithread_scenarios = multithread_scenarios
        self.couplstr = MDOCouplingStructure(self.get_sub_disciplines())

        # Create MDA
        self._build_mdas(mda_name, **mda_options)

        # Create MDOChain : MDA1 -> sub scenarios -> MDA2
        self._build_chain()

        # Cleanup design space
        self._update_design_space()

        # Builds the objective function on top of the chain
        self._build_objective_from_disc(self._objective_name)

    def _build_scenario_adapters(
        self,
        output_functions=False,
        pass_nonshared_var=False,
        adapter_class=MDOScenarioAdapter,
        **adapter_options
    ):
        """Builds the MDOScenarioAdapter required for each sub scenario
        This is used to build the self.chain.

        :param output_functions: if True then the optimization functions are
            outputs of the adapter
        :type output_functions: bool
        :param pass_nonshared_var: If True, the non-shared design variables
            are inputs of the scenarios adapters
        :type pass_nonshared_var: bool
        :param adapter_class: class of the adapters
        :type adapter_class: MDOScenarioAdapter
        :param adapter_options: options for the adapters initialization
        :type adapter_options: dict
        """
        adapters = []
        # coupled sub-disciplines

        couplings = self.couplstr.strong_couplings()
        mda2_inpts = self._get_mda2_inputs()

        shared_dv = set(self._shared_dv)
        for scenario in self.get_sub_scenarios():

            # Get the I/O names of the sub-scenario top-level disciplines
            top_disc = scenario.formulation.get_top_level_disc()
            top_inputs = [
                inpt for disc in top_disc for inpt in disc.get_input_data_names()
            ]
            top_outputs = [
                outpt for disc in top_disc for outpt in disc.get_output_data_names()
            ]

            # All couplings of the scenarios are taken from the MDA
            sc_allins = list(
                set(top_inputs) & set(couplings)
                |
                # Add shared variables from system scenario driver
                set(top_inputs) & shared_dv
            )
            if pass_nonshared_var:
                nonshared_var = scenario.design_space.variables_names
                sc_allins = list(set(sc_allins) | set(top_inputs) & set(nonshared_var))
            # Output couplings of scenario are given to MDA for speedup
            if output_functions:
                opt_problem = scenario.formulation.opt_problem
                sc_outvars = opt_problem.objective.outvars
                sc_constraints = opt_problem.get_constraints_names()
                sc_out_coupl = sc_outvars + sc_constraints
            else:
                sc_out_coupl = list(set(top_outputs) & set(couplings + mda2_inpts))

            # Add private variables from disciplinary scenario design space
            sc_allouts = sc_out_coupl + scenario.design_space.variables_names

            adapter = adapter_class(scenario, sc_allins, sc_allouts, **adapter_options)
            adapters.append(adapter)
        return adapters

    def _get_mda2_inputs(self):
        """Return the list of MDA2 inputs."""
        return []

    @classmethod
    def get_sub_options_grammar(cls, **options):
        """
        When some options of the formulation depend on higher level
        options, a sub option schema may be specified here, mainly for
        use in the API

        :param options: options dict required to deduce the sub options grammar
        :returns: None, or the sub options grammar
        """
        main_mda = options.get("mda_name")
        if main_mda is None:
            raise ValueError(
                "'mda_name' option is required \n"
                + "to deduce the sub options of BiLevel !"
            )
        factory = MDAFactory().factory
        return factory.get_options_grammar(main_mda)

    @classmethod
    def get_default_sub_options_values(cls, **options):
        """
        When some options of the formulation depend on higher level
        options, a sub option defaults may be specified here, mainly for
        use in the API

        :param options: options dict required to deduce the sub options grammar
        :returns: None, or the sub options defaults
        """
        main_mda = options.get("mda_name")
        if main_mda is None:
            raise ValueError(
                "'mda_name' option is required \n"
                + "to deduce the sub options of BiLevel !"
            )
        factory = MDAFactory().factory
        return factory.get_default_options_values(main_mda)

    def _build_mdas(self, mda_name, **mda_options):
        """Builds the chain : MDA -> MDOScenarios -> MDA
        on top of which all functions are built.
        """
        disc_mda1 = self.couplstr.strongly_coupled_disciplines()
        if len(disc_mda1) > 0:
            self.mda1 = self._mda_factory.create(mda_name, disc_mda1, **mda_options)
            self.mda1.warm_start = True
        else:
            LOGGER.warning(
                "No strongly coupled disciplines detected, "
                + " MDA1 is deactivated in the BiLevel formulation."
            )

        disc_mda2 = self.get_sub_disciplines()
        self.mda2 = self._mda_factory.create(mda_name, disc_mda2, **mda_options)

        self.mda2.warm_start = False

    def _build_chain_dis_sub_opts(self):
        """
        Inits the chain of disciplines and the list of sub scenarios
        """
        chain_dis = []
        if self.mda1 is not None:
            chain_dis = [self.mda1]
        sub_opts = self.scenario_adapters
        return chain_dis, sub_opts

    def _build_chain(self):
        """Builds the chain : MDA -> MDOScenarios -> MDA
        on top of which all functions are built.
        """
        # Build the scenario adapters to be chained with MDAs
        adapter_opt = {"reset_x0_before_opt": self.reset_x0_before_opt}
        self.scenario_adapters = self._build_scenario_adapters(**adapter_opt)
        chain_dis, sub_opts = self._build_chain_dis_sub_opts()

        if self.__parallel_scenarios:
            use_threading = self._multithread_scenarios

            par_chain = MDOParallelChain(sub_opts, use_threading=use_threading)

            chain_dis += [par_chain, self.mda2]
        else:
            # Chain MDA -> scenarios exec -> MDA
            chain_dis += sub_opts + [self.mda2]

        self.chain = MDOChain(chain_dis, name="bilevel_chain")

        if not self.reset_x0_before_opt and self.mda1 is not None:
            run_mda1_orig = self.mda1._run

            def _run_mda():
                """Redefine mda1 execution to warm start the chain
                with previous x_local opt

                :param input_data: Default value = None)
                """
                # TODO : Define a pre run method to be overloaded in MDA maybe
                # Or use observers at the system driver level to pass the local
                # vars
                for scenario in self.get_sub_scenarios():
                    x_loc_d = scenario.design_space.get_current_x_dict()
                    for indata, x_loc in x_loc_d.items():
                        if self.mda1.is_input_existing(indata):
                            if x_loc is not None:
                                self.mda1.local_data[indata] = x_loc
                return run_mda1_orig()

            self.mda1._run = _run_mda

    def _update_design_space(self):
        """Update the design space by removing the coupling variables"""
        self._set_defaultinputs_from_ds()
        self._remove_sub_scenario_dv_from_ds()
        self._remove_couplings_from_ds()
        self._remove_unused_variables()

    def _remove_couplings_from_ds(self):
        """Removes the coupling variables from the design space"""
        if hasattr(self.mda2, "strong_couplings"):
            # Otherwise, the MDA2 may be a user provided MDA
            # Which manages the couplings internally
            couplings = self.mda2.strong_couplings
            design_space = self.opt_problem.design_space
            for coupling in couplings:
                if coupling in design_space.variables_names:
                    design_space.remove_variable(coupling)

    def get_top_level_disc(self):
        """ Overriden method from MDOFormulation base class """
        return [self.chain]

    def get_expected_workflow(self):
        """Overriden method from MDOFormulation base class
        delegated to chain object"""
        return self.chain.get_expected_workflow()

    def get_expected_dataflow(self):
        """Overriden method from MDOFormulation base class
        delegated to chain object"""
        return self.chain.get_expected_dataflow()

    def add_constraint(
        self,
        output_name,
        constraint_type=MDOFunction.TYPE_EQ,
        constraint_name=None,
        value=None,
        positive=False,
    ):
        """
        Add a contraint to the formulation

        :param output_name: param constraint_type:
            (Default value = MDOFunction.TYPE_EQ)
        :param constraint_name: Default value = None)
        :param value: Default value = None)
        :param positive: Default value = False)
        :param constraint_type:  (Default value = MDOFunction.TYPE_EQ)

        """
        if self._apply_cstr_to_system:
            super(BiLevel, self).add_constraint(
                output_name, constraint_type, constraint_name, value, positive
            )
        if self._apply_cstr_tosub_scenarios:
            added = False
            outputs_list = self._check_add_cstr_input(output_name, constraint_type)
            for scen in self.get_sub_scenarios():
                if self._scenario_computes_outputs(scen, outputs_list):
                    scen.add_constraint(
                        outputs_list, constraint_type, constraint_name, value, positive
                    )
                    added = True
            if not added:
                raise ValueError(
                    "No sub scenario has an output named "
                    + str(output_name)
                    + " cannot create such a constraint."
                )

    @staticmethod
    def _scenario_computes_outputs(scenario, output_names):
        """Returns True if the top level disciplines compute
        the outputs named output_names

        :param output_names: name of the variable names to check
        :param scenario: the scenario to be tested
        """
        for disc in scenario.formulation.get_top_level_disc():
            if disc.is_all_outputs_existing(output_names):
                return True
        return False
