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
A Scenario which driver is an optimization algorithm
****************************************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from copy import copy, deepcopy
from datetime import timedelta
from timeit import default_timer as timer

from future import standard_library
from numpy import atleast_1d, zeros
from numpy.linalg import norm

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.post_optimal_analysis import PostOptimalAnalysis
from gemseo.core.discipline import MDODiscipline
from gemseo.core.json_grammar import JSONGrammar
from gemseo.core.parallel_execution import DiscParallelLinearization
from gemseo.core.scenario import Scenario

# The detection of formulations requires to import them,
# before calling get_formulation_from_name
standard_library.install_aliases()


from gemseo import LOGGER


class MDOScenario(Scenario):
    """Multidisciplinary Design Optimization Scenario, main user interface
    Creates an optimization problem and solves it with an optimizer

    The main differences between Scenario and MDOScenario are the allowed
    inputs in the MDOScenario.json, which differs from DOEScenario.json,
    at least on the driver names

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
    use the add_user_defined_constraint method

    To view the results, use the "post_process" method after execution.
    You can view:

    - the design variables history, the objective value, the constraints,
      by using:
      scenario.post_process("OptHistoryView", show=False, save=True)
    - Quadratic approximations of the functions close to the
      optimum, when using gradient based algorithms, by using:
      scenario.post_process("QuadApprox", method="SR1", show=False,
      save=True, function="my_objective_name",
      file_path="appl_dir")
    - Self Organizing Maps of the design space, by using:
      scenario.post_process("SOM", save=True, file_path="appl_dir")

    To list post-processing on your setup,
    use the method scenario.posts
    For more detains on their options, go to the "gemseo.post" package

    """

    # Constants for input variables in json schema
    MAX_ITER = "max_iter"
    X_OPT = "x_opt"

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
        # This loads the right json grammars from class name
        super(MDOScenario, self).__init__(
            disciplines,
            formulation,
            objective_name,
            design_space,
            name,
            **formulation_options
        )
        self.clear_history_before_run = False

    def _run_algorithm(self):
        """Runs the optimization algo"""
        problem = self.formulation.opt_problem
        # Clears the database when multiple runs are performed (bi level)
        if self.clear_history_before_run:
            problem.database.clear()
        algo_name = self.local_data[self.ALGO]
        max_iter = self.local_data[self.MAX_ITER]
        options = self.local_data.get(self.ALGO_OPTIONS)
        if options is None:
            options = {}
        if self.MAX_ITER in options:
            LOGGER.warning(
                "Double definition of algorithm option " "max_iter, keeping value: %s",
                max_iter,
            )
            options.pop(self.MAX_ITER)

        lib = self._algo_factory.create(algo_name)
        self.optimization_result = lib.execute(
            problem, algo_name=algo_name, max_iter=max_iter, **options
        )
        return self.optimization_result

    def _run(self):
        """Execute the scenario and run the optimization problems"""
        t_0 = timer()
        LOGGER.info(" ")
        LOGGER.info("*** Start MDO Scenario execution ***")
        self.log_me()
        self._run_algorithm()

        # MDODiscipline.execute is not finished therefore self.exec_time is not
        # computed yet, need to recompute it, besides exec_time is the total
        # execution time, while this is for a single execution
        delta_t = timer() - t_0
        LOGGER.info(
            "*** MDO Scenario run terminated in %s ***", str(timedelta(seconds=delta_t))
        )

    def _init_algo_factory(self):
        """
        Initalizes the algorithms factory
        """
        self._algo_factory = OptimizersFactory()


class MDOScenarioAdapter(MDODiscipline):
    """An adapter class for MDO Scenario:
    input variables are specified
    they update default_data in the top level discipline
    they output data from the top level discipline outputs."""

    LOWER_BND_SUFFIX = "_lower_bnd"
    UPPER_BND_SUFFIX = "_upper_bnd"

    def __init__(
        self,
        scenario,
        inputs_list,
        outputs_list,
        reset_x0_before_opt=False,
        set_x0_before_opt=False,
        set_bounds_before_opt=False,
    ):
        """
        Constructor

        :param scenario: the scenario to adapt
        :type scenario: MDOScenario
        :param inputs_list: list of inputs to overload at
            sub scenario execution
        :type inputs_list: list(str)
        :param outputs_list: list of outputs to get from
            scenario execution
        :type outputs_list: list(str)
        :param reset_x0_before_opt: before running the sub optimization, reset
            the initial guess
        :type reset_x0_before_opt: bool
        :param set_x0_before_opt: if True, sets the initial point of the sub
            scenario, useful for multi-start
        :type set_x0_before_opt: bool
        :param set_bounds_before_opt: if True, sets the bounds of the design
            space, useful for trust regions
        :type set_bounds_before_opt: bool
        """
        if reset_x0_before_opt and set_x0_before_opt:
            raise ValueError("Inconsistent options for ScenarioAdapter !")
        self.scenario = scenario
        self._set_x0_before_opt = set_x0_before_opt
        self._set_bounds_before_opt = set_bounds_before_opt
        self._inputs_list = inputs_list
        self._outputs_list = outputs_list
        self._reset_x0_before_opt = reset_x0_before_opt
        name = scenario.name
        super(MDOScenarioAdapter, self).__init__(name)

        self._update_grammars()
        self._dv_in_names = None
        if set_x0_before_opt:
            dv_names = set(self.scenario.formulation.design_space.variables_names)
            self._dv_in_names = list(dv_names & set(self._inputs_list))

        # Set the initial bounds as default bounds
        self._bounds_names = []
        if set_bounds_before_opt:
            dspace = scenario.design_space
            lower_bounds = dspace.array_to_dict(dspace.get_lower_bounds())
            lower_suffix = MDOScenarioAdapter.LOWER_BND_SUFFIX
            upper_bounds = dspace.array_to_dict(dspace.get_upper_bounds())
            upper_suffix = MDOScenarioAdapter.UPPER_BND_SUFFIX
            for bounds, suffix in [
                (lower_bounds, lower_suffix),
                (upper_bounds, upper_suffix),
            ]:
                bounds = {name + suffix: val for name, val in bounds.items()}
                self.default_inputs.update(bounds)
                self._bounds_names.extend(bounds.keys())

        # Optimization functions are redefined at each run
        # since default inputs of top
        # level discipline change
        # History must be erased otherwise the wrong values are retrieved
        # between two runs
        scenario.clear_history_before_run = True
        self._x_dict_0 = deepcopy(scenario.design_space.get_current_x_dict())

        self.post_optimal_analysis = None

    def _update_grammars(self):
        """
        Updates the inputs and outputs grammars
        """

        formulation = self.scenario.formulation
        opt_problem = formulation.opt_problem
        top_leveld = formulation.get_top_level_disc()
        for disc in top_leveld:
            self.input_grammar.update_from(disc.input_grammar)
            self.output_grammar.update_from(disc.output_grammar)
            # The output may also be the optimum value of the design
            # variables, so the output grammar may contain inputs
            # of the disciplines. All grammars are filtered just after
            # this loop
            self.output_grammar.update_from(disc.input_grammar)
            self.default_inputs.update(disc.default_inputs)

        self.input_grammar.restrict_to(self._inputs_list)
        self.output_grammar.restrict_to(self._outputs_list)
        # If a DV is not an input of the top level disciplines:
        output_names = self.output_grammar.get_data_names()
        missing_out = set(self._outputs_list) - set(output_names)
        if missing_out:
            dv_names = opt_problem.design_space.variables_names
            miss_dvs = set(dv_names) & set(missing_out)
            if miss_dvs:
                dv_gram = JSONGrammar("dvs")
                dv_gram.initialize_from_data_names(miss_dvs)
                self.output_grammar.update_from(dv_gram)

        output_names = self.output_grammar.get_data_names()
        missing_out = set(self._outputs_list) - set(output_names)

        # Add the design variables bounds to the input grammar
        if self._set_bounds_before_opt:
            current_x = self.scenario.design_space.get_current_x_dict()
            typical_data_dict = dict()
            for suffix in [
                MDOScenarioAdapter.LOWER_BND_SUFFIX,
                MDOScenarioAdapter.UPPER_BND_SUFFIX,
            ]:
                bnds = {name + suffix: val for name, val in current_x.items()}
                typical_data_dict.update(bnds)
            bnds_gram = JSONGrammar("bnds")
            bnds_gram.initialize_from_base_dict(typical_data_dict)
            self.input_grammar.update_from(bnds_gram)

        if missing_out:
            raise ValueError(
                "Can't compute outputs from scenarios: " + str(missing_out)
            )
        missing_inpt = set(self._inputs_list) - set(self.input_grammar.get_data_names())
        if missing_inpt:
            raise ValueError(
                "Can't compute inputs from scenarios: " + str(missing_inpt)
            )

    def _run(self):
        """
        Runs the scenario
        """
        self._pre_run()
        self.scenario.execute()
        self._post_run()

    def _pre_run(self):
        """
        Pre-processes the scenario.
        """
        formulation = self.scenario.formulation
        opt_problem = formulation.opt_problem
        design_space = opt_problem.design_space
        top_leveld = formulation.get_top_level_disc()

        # Update the top level discipline default inputs with adapter inputs
        # This is the key role of the adapter
        for indata in self._inputs_list:
            for disc in top_leveld:
                if disc.is_input_existing(indata):
                    disc.default_inputs[indata] = self.local_data[indata]

        # Default inputs have changed, therefore caches shall be cleared
        self.scenario.cache.clear()
        self.scenario.reset_statuses_for_run()
        for func in opt_problem.get_all_functions():
            # Avoids max_iter reached
            func.n_calls = 0

        if self._reset_x0_before_opt:
            design_space.set_current_x(self._x_dict_0)

        # Set the starting point of the sub scenario with current dv names
        if self._set_x0_before_opt:
            dv_values = {dv_n: self.local_data[dv_n] for dv_n in self._dv_in_names}
            self.scenario.formulation.design_space.set_current_x(dv_values)

        # Set the bounds of the sub-scenario
        if self._set_bounds_before_opt:
            for name in design_space.variables_names:
                # Set the lower bound
                lower_suffix = MDOScenarioAdapter.LOWER_BND_SUFFIX
                lower_bound = self.local_data[name + lower_suffix]
                design_space.set_lower_bound(name, lower_bound)
                # Set the upper bound
                upper_suffix = MDOScenarioAdapter.UPPER_BND_SUFFIX
                upper_bound = self.local_data[name + upper_suffix]
                design_space.set_upper_bound(name, upper_bound)

    def _post_run(self):
        """
        Post-processes the scenario.
        """
        formulation = self.scenario.formulation
        opt_problem = formulation.opt_problem
        design_space = opt_problem.design_space

        # Test if the last evaluation is the optimum
        x_opt = design_space.get_current_x()
        last_x = opt_problem.database.get_x_by_iter(-1)
        last_eval_not_opt = norm(x_opt - last_x) / (1.0 + norm(last_x)) > 1e-14
        if last_eval_not_opt:
            # Revaluate all functions at optimum
            # To re execute all disciplines and get the right data
            opt_problem.evaluate_functions(
                x_opt,
                eval_jac=False,
                eval_obj=True,
                normalize=False,
                # Force call without database
                no_db_no_norm=True,
            )

        # Retrieves top-level discipline outputs
        self._retrieve_top_level_outputs()

    def _retrieve_top_level_outputs(self):
        """
        Overwrites the adapter outputs with the top-level discipline outputs
        and optimal design parameters
        """
        formulation = self.scenario.formulation
        opt_problem = formulation.opt_problem
        top_leveld = formulation.get_top_level_disc()
        x_dict = opt_problem.design_space.get_current_x_dict()
        for outdata in self._outputs_list:
            for disc in top_leveld:
                if disc.is_output_existing(outdata) and outdata not in x_dict:
                    self.local_data[outdata] = disc.local_data[outdata]
            out_ds = x_dict.get(outdata)
            if out_ds is not None:
                self.local_data[outdata] = out_ds

    def get_expected_workflow(self):
        return self.scenario.get_expected_workflow()

    def get_expected_dataflow(self):
        return self.scenario.get_expected_dataflow()

    def _compute_jacobian(self, inputs=None, outputs=None):
        """
        Computes the Jacobian of the adapted scenario outputs with respect to
        its inputs.
        The Jacobian is stored as a dict of ndarray dict:
        jac = {name: { input_name: (output_dim, input_dim) ndarray } }

        The bound-constraints on the scenario optimization variables are
        assumed independent of the other scenario inputs.

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """
        opt_problem = self.scenario.formulation.opt_problem
        ineq_tol = opt_problem.ineq_tolerance
        outvars = opt_problem.objective.outvars
        if len(outvars) != 1:
            raise ValueError("The objective must be single-valued.")

        # Check the required inputs
        if inputs is None and not self._set_bounds_before_opt:
            inputs = self._inputs_list
        elif inputs is None and self._set_bounds_before_opt:
            # Bounds are inputs of the adapter
            inputs = [
                name for name in self._bounds_names if name not in self._inputs_list
            ]
            inputs = self._inputs_list + inputs
        elif set(inputs) - set(self._inputs_list) - set(self._bounds_names):
            not_inputs = set(inputs) - set(self._inputs_list) - set(self._bounds_names)
            raise ValueError(
                "The following are not inputs of the adapter: "
                + ", ".join(not_inputs)
                + "."
            )
        # N.B the adapter is assumed constant w.r.t. bounds
        bound_inputs = set(inputs) & set(self._bounds_names)

        # Check the required outputs
        if outputs is None:
            outputs = outvars
        elif set(outputs) - set(self._outputs_list):
            raise ValueError(
                "The following are not outputs of the adapter: "
                + str(set(outputs) - set(self._outputs_list))
                + "."
            )
        nondifferentiable_outputs = set(outputs) - set(outvars)
        if nondifferentiable_outputs:
            raise ValueError(
                "Post-optimal Jacobians of "
                + ", ".join(nondifferentiable_outputs)
                + " cannot be computed."
            )

        # Initialize the Jacobian
        diff_inputs = [name for name in inputs if name not in bound_inputs]
        # N.B. there may be only bound inputs
        self._init_jacobian(diff_inputs, outputs)

        # Compute the Jacobians of the optimization functions
        jacobians = self._compute_auxiliary_jacobians(diff_inputs, use_threading=True)

        # Perform the post-optimal analysis
        self.post_optimal_analysis = PostOptimalAnalysis(opt_problem, ineq_tol)
        post_opt_jac = self.post_optimal_analysis.execute(
            outputs, diff_inputs, jacobians
        )
        self.jac.update(post_opt_jac)

        # Fill the Jacobian blocks w.r.t. bounds with zeros
        for out_jac in self.jac.values():
            for in_name in bound_inputs:
                in_dim = self.default_inputs[in_name].size
                out_jac[in_name] = zeros((1, in_dim))

    def _compute_auxiliary_jacobians(self, inputs, func_names=None, use_threading=True):
        """
        Computes the Jacobians of the optimization functions.

        :param inputs: names list of the inputs w.r.t. which differentiate
        :type inputs: list(str)
        :param func_names: names list of the functions to differentiate
            If None then all the optimizations functions are differentiated
        :type func_names: list(str)
        :param use_threading : if True, use Threads instead of processes
            to parallelize the execution
        :type use_threading: bool
        """
        # Gather the names of the functions to differentiate
        opt_problem = self.scenario.formulation.opt_problem
        if func_names is None:
            func_names = (
                opt_problem.objective.outvars + opt_problem.get_constraints_names()
            )

        # Identify the disciplines that compute the functions
        disc_dict = dict()
        for name in func_names:
            for disc in self.scenario.formulation.get_top_level_disc():
                if disc.is_all_outputs_existing([name]):
                    disc_dict[name] = disc
                    break

        # Linearize the required disciplines
        for disc in set(disc_dict.values()):
            inputs_set = set(disc.get_input_data_names()) & set(inputs)
            outputs_set = set(disc.get_output_data_names()) & set(func_names)
            if inputs_set and outputs_set:
                disc.add_differentiated_inputs(list(inputs_set))
                disc.add_differentiated_outputs(list(outputs_set))
        disc_list = list(set(disc_dict.values()))
        paralell_lin = DiscParallelLinearization(disc_list, use_threading=use_threading)
        # Update the local data with the optimal design parameters
        # [The adapted scenario is assumed to have been run beforehand.]
        x_opt_dict = opt_problem.design_space.get_current_x_dict()
        post_opt_data = copy(self.local_data)
        post_opt_data.update(x_opt_dict)
        paralell_lin.execute([post_opt_data] * len(disc_list))

        # Store the Jacobians
        jacobians = dict()
        for name in func_names:
            jacobians[name] = dict()
            for input_name in inputs:
                jac_block = disc_dict[name].jac[name].get(input_name)
                if jac_block is None:
                    output_value = self.get_outputs_by_name(name)
                    input_value = self.get_inputs_by_name(input_name)
                    jac_block = zeros((len(output_value, len(input_value))))
                jacobians[name][input_name] = jac_block

        return jacobians


class MDOObjScenarioAdapter(MDOScenarioAdapter):
    """
    A scenario adapter that overwrites the local data with the optimal
    objective function value.
    """

    def _retrieve_top_level_outputs(self):
        """
        Overwrites the adapter outputs with the top-level discipline outputs
        and optimal design parameters
        """
        formulation = self.scenario.formulation
        opt_problem = formulation.opt_problem
        top_leveld = formulation.get_top_level_disc()

        # Get the optimal outputs
        optim_data = opt_problem.design_space.get_current_x_dict()
        f_opt = opt_problem.get_optimum()[0]
        if not opt_problem.minimize_objective:
            f_opt = -f_opt
        outvars = opt_problem.objective.outvars
        if not len(outvars) == 1:
            raise ValueError("The objective function must be single-valued.")
        optim_data[outvars[0]] = atleast_1d(f_opt)  # FIXME

        # Overwrite the adapter local data
        for outdata in self._outputs_list:
            for disc in top_leveld:
                if disc.is_output_existing(outdata) and outdata not in optim_data:
                    self.local_data[outdata] = disc.local_data[outdata]
            out_ds = optim_data.get(outdata)
            if out_ds is not None:
                self.local_data[outdata] = out_ds

    def _compute_jacobian(self, inputs=None, outputs=None):
        """
        Computes the Jacobian of the adapted scenario outputs with respect to
        its inputs.
        The Jacobian is stored as a dict of ndarray dict:
        jac = {name: { input_name: (output_dim, input_dim) ndarray } }

        The bound-constraints on the scenario optimization variables are
        assumed independent of the other scenario inputs.

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """
        MDOScenarioAdapter._compute_jacobian(self, inputs, outputs)
        # The gradient of the objective function cannot be computed by the
        # disciplines, but the gradients of the constraints can.
        # The objective function is assumed independent of non-optimization
        # variables.
        obj_name = self.scenario.formulation.opt_problem.objective.outvars[0]
        mult_cstr_jac_key = PostOptimalAnalysis.MULT_DOT_CONSTR_JAC
        self.jac[obj_name] = dict(self.jac[mult_cstr_jac_key])
