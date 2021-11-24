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
#        :author: Pierre-Jean Barjhoux, Benoit Pauwels - MDOScenarioAdapter
#                                                        Jacobian computation
"""A scenario whose driver is an optimization algorithm."""
from __future__ import division, unicode_literals

import logging
from copy import copy, deepcopy
from datetime import timedelta
from timeit import default_timer as timer
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from numpy import atleast_1d, zeros
from numpy.core.multiarray import ndarray
from numpy.linalg import norm

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.lagrange_multipliers import LagrangeMultipliers
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_result import OptimizationResult
from gemseo.algos.post_optimal_analysis import PostOptimalAnalysis
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import LoopExecSequence
from gemseo.core.json_grammar import JSONGrammar
from gemseo.core.parallel_execution import DiscParallelLinearization
from gemseo.core.scenario import Scenario

# The detection of formulations requires to import them,
# before calling get_formulation_from_name


LOGGER = logging.getLogger(__name__)


class MDOScenario(Scenario):
    """A multidisciplinary scenario to be executed by an optimizer.

    A :class:`MDOScenario` is a particular :class:`.Scenario`
    whose driver is an optimization algorithm.
    This algorithm must be implemented in an :class:`.OptimizationLibrary`.

    Attributes:
        clear_history_before_run (bool): If True, clear history before run.
    """

    # Constants for input variables in json schema
    MAX_ITER = "max_iter"
    X_OPT = "x_opt"

    _ATTR_TO_SERIALIZE = Scenario._ATTR_TO_SERIALIZE + (
        "formulation",
        "disciplines",
        "clear_history_before_run",
        "_algo_factory",
    )

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
                , e.g. JSON_GRAMMAR_TYPE or SIMPLE_GRAMMAR_TYPE.
            **formulation_options: The options
                to be passed to the :class:`.MDOFormulation`.
        """
        # This loads the right json grammars from class name
        super(MDOScenario, self).__init__(
            disciplines,
            formulation,
            objective_name,
            design_space,
            name=name,
            grammar_type=grammar_type,
            **formulation_options
        )
        self.clear_history_before_run = False

    def _run_algorithm(self):  # type: (...) -> OptimizationResult
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
                "Double definition of algorithm option max_iter, keeping value: %s",
                max_iter,
            )
            options.pop(self.MAX_ITER)

        lib = self._algo_factory.create(algo_name)
        self.optimization_result = lib.execute(
            problem, algo_name=algo_name, max_iter=max_iter, **options
        )
        return self.optimization_result

    def _run(self):
        t_0 = timer()
        LOGGER.info(" ")
        LOGGER.info("*** Start MDO Scenario execution ***")
        LOGGER.info("%s", repr(self))
        self._run_algorithm()
        # MDODiscipline.execute is not finished therefore self.exec_time is not
        # computed yet, need to recompute it, besides exec_time is the total
        # execution time, while this is for a single execution
        delta_t = timer() - t_0
        LOGGER.info(
            "*** MDO Scenario run terminated in %s ***", timedelta(seconds=delta_t)
        )

    def _init_algo_factory(self):
        self._algo_factory = OptimizersFactory()

    def _update_grammar_input(self):  # type: (...) -> None
        self.input_grammar.update_elements(
            algo=str, max_iter=int, algo_options=dict, python_typing=True
        )
        self.input_grammar.update_required_elements(
            algo=True, max_iter=True, algo_options=False
        )


class MDOScenarioAdapter(MDODiscipline):
    """An adapter class for MDO Scenario.

    The specified input variables update the default input data
    of the top level discipline
    while the output ones filter the output data from the top level discipline outputs.

    Attributes:
        scenario (Scenario): The scenario to be adapted.
        post_optimal_analysis (PostOptimalAnalysis): The post-optimal analysis.
    """

    LOWER_BND_SUFFIX = "_lower_bnd"
    UPPER_BND_SUFFIX = "_upper_bnd"
    MULTIPLIER_SUFFIX = "_multiplier"

    _ATTR_TO_SERIALIZE = MDODiscipline._ATTR_TO_SERIALIZE + (
        "scenario",
        "_inputs_list",
        "_reset_x0_before_opt",
        "_x_dict_0",
        "_set_x0_before_opt",
        "_set_bounds_before_opt",
        "_outputs_list",
        "_output_multipliers",
    )

    def __init__(
        self,
        scenario,  # type: Scenario
        inputs_list,  # type: Sequence[str]
        outputs_list,  # type: Sequence[str]
        reset_x0_before_opt=False,  # type: bool
        set_x0_before_opt=False,  # type: bool
        set_bounds_before_opt=False,  # type: bool
        cache_type=MDODiscipline.SIMPLE_CACHE,  # type: str
        output_multipliers=False,  # type: bool
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,  # type: str
        name=None,  # type: Optional[str]
    ):  # type: (...) -> None
        """
        Args:
            scenario: The scenario to adapt.
            inputs_list: The inputs to overload at sub-scenario execution.
            outputs_list: The outputs to get from the sub-scenario execution.
            reset_x0_before_opt: If True, reset the initial guess
                before running the sub optimization.
            set_x0_before_opt: If True, set the initial guess of the sub-scenario.
                This is useful for multi-start optimization.
            set_bounds_before_opt: If True, set the bounds of the design space.
                This is useful for trust regions.
            cache_type: The type of cache policy.
            output_multipliers: If True,
                the Lagrange multipliers of the scenario optimal solution are computed
                and added to the outputs.
            name: The name of the scenario adapter.
                If None, use ``"{}_adapter"``.

        Raises:
            ValueError: If both `reset_x0_before_opt` and `set_x0_before_opt` are True.
        """
        if reset_x0_before_opt and set_x0_before_opt:
            raise ValueError("Inconsistent options for MDOScenarioAdapter.")
        self.scenario = scenario
        self._set_x0_before_opt = set_x0_before_opt
        self._set_bounds_before_opt = set_bounds_before_opt
        self._inputs_list = inputs_list
        self._outputs_list = outputs_list
        self._reset_x0_before_opt = reset_x0_before_opt
        self._output_multipliers = output_multipliers
        name = name or "{}_adapter".format(scenario.name)
        super(MDOScenarioAdapter, self).__init__(
            name, cache_type=cache_type, grammar_type=grammar_type
        )

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

    def _update_grammars(self):  # type: (...) -> None
        """Update the input and output grammars.

        Raises:
            ValueError: Either if a specified input is missing from the input grammar
                or if a specified output is missing from the output grammar.
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
                "Can't compute outputs from scenarios: {}.".format(
                    ", ".join(sorted(missing_out))
                )
            )
        missing_inpt = set(self._inputs_list) - set(self.input_grammar.get_data_names())
        if missing_inpt:
            raise ValueError(
                "Can't compute inputs from scenarios: {}.".format(
                    ", ".join(sorted(missing_inpt))
                )
            )

        # Add the Lagrange multipliers to the output grammar
        if self._output_multipliers:
            self._add_output_multipliers()

    def _add_output_multipliers(self):  # type: (...) -> None
        """Add the Lagrange multipliers of the scenario optimal solution as outputs."""
        # Fill a dictionary with data of typical shapes
        base_dict = dict()
        problem = self.scenario.formulation.opt_problem
        # bound-constraints multipliers
        current_x = problem.design_space.get_current_x_dict()
        base_dict.update(
            {
                self.get_bnd_mult_name(var_name, False): val
                for var_name, val in current_x.items()
            }
        )
        base_dict.update(
            {
                self.get_bnd_mult_name(var_name, True): val
                for var_name, val in current_x.items()
            }
        )
        # equality- and inequality-constraints multipliers
        base_dict.update(
            {
                self.get_cstr_mult_name(cstr_name): zeros(1)
                for cstr_name in problem.get_constraints_names()
            }
        )

        # Update the output grammar
        multipliers_grammar = JSONGrammar("multipliers")
        multipliers_grammar.initialize_from_base_dict(base_dict)
        self.output_grammar.update_from(multipliers_grammar)

    @staticmethod
    def get_bnd_mult_name(
        variable_name,  # type: str
        is_upper,  # type:bool
    ):  # type: (...) -> str
        """Return the name of the lower bound-constraint multiplier of a variable.

        Args:
            variable_name: The name of the variable.
            is_upper: If True, return name of the upper bound-constraint multiplier.
                Otherwise, return the name of the lower bound-constraint multiplier.

        Returns:
            The name of a bound-constraint multiplier.
        """
        mult_name = variable_name
        mult_name += "_upp-bnd" if is_upper else "_low-bnd"
        mult_name += MDOScenarioAdapter.MULTIPLIER_SUFFIX
        return mult_name

    @staticmethod
    def get_cstr_mult_name(
        constraint_name,  # type: str
    ):  # type: (...) ->str
        """Return the name of the multiplier of a constraint.

        Args:
            constraint_name: The name of the constraint.

        Returns:
            The name of the multiplier.
        """
        return constraint_name + MDOScenarioAdapter.MULTIPLIER_SUFFIX

    def _run(self):  # type: (...) -> None
        self._pre_run()
        self.scenario.execute()
        self._post_run()

    def _pre_run(self):  # type: (...) -> None
        """Pre-run the scenario."""
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

        # Reset the iter counter for the opt problem.
        opt_problem.current_iter = 0
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

    def _post_run(self):  # type: (...) -> None
        """Post-process the scenario."""
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

        # Compute the Lagrange multipliers and store them in the local data
        if self._output_multipliers:
            self._compute_lagrange_multipliers()

    def _retrieve_top_level_outputs(self):  # type: (...) -> None
        """Retrieve the top-level outputs.

        This methods overwrites the adapter outputs with the top-level discipline
        outputs and the optimal design parameters.
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

    def _compute_lagrange_multipliers(self):  # type: (...) -> None
        """Compute the Lagrange multipliers for the optimal solution of the scenario.

        This methods stores the multipliers in the local data.
        """
        # Compute the Lagrange multipliers
        problem = self.scenario.formulation.opt_problem
        x_opt = problem.solution.x_opt
        lagrange = LagrangeMultipliers(problem)
        lagrange.compute(x_opt, problem.ineq_tolerance)

        # Store the Lagrange multipliers in the local data
        multipliers = lagrange.get_multipliers_arrays()
        self.local_data.update(
            {
                self.get_bnd_mult_name(name, False): mult
                for name, mult in multipliers[lagrange.LOWER_BOUNDS].items()
            }
        )
        self.local_data.update(
            {
                self.get_bnd_mult_name(name, True): mult
                for name, mult in multipliers[lagrange.UPPER_BOUNDS].items()
            }
        )
        self.local_data.update(
            {
                self.get_cstr_mult_name(name): mult
                for name, mult in multipliers[lagrange.EQUALITY].items()
            }
        )
        self.local_data.update(
            {
                self.get_cstr_mult_name(name): mult
                for name, mult in multipliers[lagrange.INEQUALITY].items()
            }
        )

    def get_expected_workflow(self):  # type: (...) -> LoopExecSequence
        return self.scenario.get_expected_workflow()

    def get_expected_dataflow(
        self,
    ):  # type: (...) -> List[Tuple[MDODiscipline,MDODiscipline,List[str]]]
        return self.scenario.get_expected_dataflow()

    def _compute_jacobian(
        self,
        inputs=None,  # type: Optional[Sequence[str]]
        outputs=None,  # type: Optional[Sequence[str]]
    ):  # type: (...) -> None
        """Compute the Jacobian of the adapted scenario outputs.

        The Jacobian is stored as a dictionary of numpy arrays:
        jac = {name: { input_name: ndarray(output_dim, input_dim) } }

        The bound-constraints on the scenario optimization variables
        are assumed independent of the other scenario inputs.

        Args:
            inputs: The linearization should be performed with respect to these inputs.
                If None, the linearization should be performed w.r.t. all inputs.
            outputs: The linearization should be performed on these outputs.
                If None, the linearization should be performed on all outputs.

        Raises:
            ValueError: Either
                if the dimension of the objective function is greater than 1,
                if a specified input is not an input of the adapter,
                if a specified output is not an output of the adapter,
                or if there is non differentiable outputs.
        """
        opt_problem = self.scenario.formulation.opt_problem
        objective_names = self.scenario.formulation.opt_problem.objective.outvars
        if len(objective_names) != 1:
            raise ValueError("The objective must be single-valued.")

        # Check the required inputs
        if inputs is None:
            inputs = set(self._inputs_list + self._bounds_names)
        else:
            not_inputs = set(inputs) - set(self._inputs_list) - set(self._bounds_names)
            if not_inputs:
                raise ValueError(
                    "The following are not inputs of the adapter: {}.".format(
                        ", ".join(sorted(not_inputs))
                    )
                )
        # N.B the adapter is assumed constant w.r.t. bounds
        bound_inputs = set(inputs) & set(self._bounds_names)

        # Check the required outputs
        if outputs is None:
            outputs = objective_names
        else:
            not_outputs = sorted(set(outputs) - set(self._outputs_list))
            if not_outputs:
                raise ValueError(
                    "The following are not outputs of the adapter: {}.".format(
                        ", ".join(not_outputs)
                    )
                )

        non_differentiable_outputs = sorted(set(outputs) - set(objective_names))
        if non_differentiable_outputs:
            raise ValueError(
                "Post-optimal Jacobians of {} cannot be computed.".format(
                    ", ".join(non_differentiable_outputs)
                )
            )

        # Initialize the Jacobian
        diff_inputs = [name for name in inputs if name not in bound_inputs]
        # N.B. there may be only bound inputs
        self._init_jacobian(diff_inputs, outputs)

        # Compute the Jacobians of the optimization functions
        jacobians = self._compute_auxiliary_jacobians(diff_inputs, use_threading=True)

        # Perform the post-optimal analysis
        ineq_tolerance = opt_problem.ineq_tolerance
        self.post_optimal_analysis = PostOptimalAnalysis(opt_problem, ineq_tolerance)
        post_opt_jac = self.post_optimal_analysis.execute(
            outputs, diff_inputs, jacobians
        )
        self.jac.update(post_opt_jac)

        # Fill the Jacobian blocks w.r.t. bounds with zeros
        for output_derivatives in self.jac.values():
            for bound_input_name in bound_inputs:
                bound_input_size = self.default_inputs[bound_input_name].size
                output_derivatives[bound_input_name] = zeros((1, bound_input_size))

    def _compute_auxiliary_jacobians(
        self,
        inputs,  # type: Iterable[str]
        func_names=None,  # type: Optional[Iterable[str]]
        use_threading=True,  # type: bool
    ):  # type: (...) -> Dict[str,Dict[str,ndarray]]
        """Compute the Jacobians of the optimization functions.

        Args:
            inputs: The names of the inputs w.r.t. which differentiate.
            func_names: The names of the functions to differentiate
                If None, then all the optimizations functions are differentiated.
            use_threading: If True, use threads instead of processes
                to parallelize the execution.

        Returns:
            The Jacobians of the optimization functions.
        """
        # Gather the names of the functions to differentiate
        opt_problem = self.scenario.formulation.opt_problem
        if func_names is None:
            func_names = (
                opt_problem.objective.outvars + opt_problem.get_constraints_names()
            )

        # Identify the disciplines that compute the functions
        disciplines = dict()
        for func_name in func_names:
            for discipline in self.scenario.formulation.get_top_level_disc():
                if discipline.is_all_outputs_existing([func_name]):
                    disciplines[func_name] = discipline
                    break

        # Linearize the required disciplines
        unique_disciplines = list(set(disciplines.values()))
        for discipline in unique_disciplines:
            diff_inputs = set(discipline.get_input_data_names()) & set(inputs)
            diff_outputs = set(discipline.get_output_data_names()) & set(func_names)
            if diff_inputs and diff_outputs:
                discipline.add_differentiated_inputs(list(diff_inputs))
                discipline.add_differentiated_outputs(list(diff_outputs))

        parallel_linearization = DiscParallelLinearization(
            unique_disciplines, use_threading=use_threading
        )
        # Update the local data with the optimal design parameters
        # [The adapted scenario is assumed to have been run beforehand.]
        post_opt_data = copy(self.local_data)
        post_opt_data.update(opt_problem.design_space.get_current_x_dict())
        parallel_linearization.execute([post_opt_data] * len(unique_disciplines))

        # Store the Jacobians
        jacobians = dict()
        for func_name in func_names:
            jacobians[func_name] = dict()
            func_jacobian = disciplines[func_name].jac[func_name]
            for input_name in inputs:
                jacobians[func_name][input_name] = func_jacobian[input_name]

        return jacobians

    def add_outputs(
        self,
        outputs_names,  # type: Iterable[str]
    ):  # type: (...) -> None
        """Add outputs to the scenario adapter.

        Args:
            outputs_names: The names of the outputs to be added.
        """
        names_to_add = [
            name for name in outputs_names if name not in self._outputs_list
        ]
        self._outputs_list.extend(names_to_add)
        self._update_grammars()


class MDOObjScenarioAdapter(MDOScenarioAdapter):
    """A scenario adapter overwriting the local data with the optimal objective."""

    def _retrieve_top_level_outputs(self):  # type: (...) -> None
        formulation = self.scenario.formulation
        opt_problem = formulation.opt_problem
        top_leveld = formulation.get_top_level_disc()

        # Get the optimal outputs
        optim_data = opt_problem.design_space.get_current_x_dict()
        f_opt = opt_problem.get_optimum()[0]
        if not opt_problem.minimize_objective:
            f_opt = -f_opt
        if not opt_problem.is_mono_objective:
            raise ValueError("The objective function must be single-valued.")
        optim_data[opt_problem.objective.outvars[0]] = atleast_1d(f_opt)

        # Overwrite the adapter local data
        for outdata in self._outputs_list:
            for disc in top_leveld:
                if disc.is_output_existing(outdata) and outdata not in optim_data:
                    self.local_data[outdata] = disc.local_data[outdata]
            out_ds = optim_data.get(outdata)
            if out_ds is not None:
                self.local_data[outdata] = out_ds

    def _compute_jacobian(
        self,
        inputs=None,  # type: Optional[Sequence[str]]
        outputs=None,  # type: Optional[Sequence[str]]
    ):  # type: (...) -> None
        MDOScenarioAdapter._compute_jacobian(self, inputs, outputs)
        # The gradient of the objective function cannot be computed by the
        # disciplines, but the gradients of the constraints can.
        # The objective function is assumed independent of non-optimization
        # variables.
        obj_name = self.scenario.formulation.opt_problem.objective.outvars[0]
        mult_cstr_jac_key = PostOptimalAnalysis.MULT_DOT_CONSTR_JAC
        self.jac[obj_name] = dict(self.jac[mult_cstr_jac_key])
