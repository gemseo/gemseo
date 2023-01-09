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
"""A discipline running a scenario."""
from __future__ import annotations

import logging
from copy import copy
from copy import deepcopy
from typing import Iterable
from typing import Sequence

from numpy import atleast_1d
from numpy import zeros
from numpy.core.multiarray import ndarray
from numpy.linalg import norm

from gemseo.algos.database import Database
from gemseo.algos.lagrange_multipliers import LagrangeMultipliers
from gemseo.algos.post_optimal_analysis import PostOptimalAnalysis
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import LoopExecSequence
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.core.parallel_execution import DiscParallelLinearization
from gemseo.core.scenario import Scenario

LOGGER = logging.getLogger(__name__)


class MDOScenarioAdapter(MDODiscipline):
    """An adapter class for MDO Scenario.

    The specified input variables update the default input data of the top level
    discipline while the output ones filter the output data from the top level discipline
    outputs.
    """

    scenario: Scenario
    """The scenario to be adapted."""

    post_optimal_analysis: PostOptimalAnalysis
    """The post-optimal analysis."""

    keep_opt_history: bool
    """Whether to keep databases copies after each execution."""

    databases: list[Database]
    """The copies of the scenario databases after execution."""

    LOWER_BND_SUFFIX = "_lower_bnd"
    UPPER_BND_SUFFIX = "_upper_bnd"
    MULTIPLIER_SUFFIX = "_multiplier"

    _ATTR_TO_SERIALIZE = MDODiscipline._ATTR_TO_SERIALIZE + (
        "scenario",
        "_input_names",
        "_reset_x0_before_opt",
        "_initial_x",
        "_set_x0_before_opt",
        "_set_bounds_before_opt",
        "_output_names",
        "_output_multipliers",
        "keep_opt_history",
        "_dv_in_names",
        "_bounds_names",
    )

    def __init__(
        self,
        scenario: Scenario,
        input_names: Sequence[str],
        output_names: Sequence[str],
        reset_x0_before_opt: bool = False,
        set_x0_before_opt: bool = False,
        set_bounds_before_opt: bool = False,
        cache_type: str = MDODiscipline.SIMPLE_CACHE,
        output_multipliers: bool = False,
        grammar_type: str = MDODiscipline.JSON_GRAMMAR_TYPE,
        name: str | None = None,
        keep_opt_history: bool = False,
        opt_history_file_prefix: str = "",
    ) -> None:
        """..
        Args:
            scenario: The scenario to adapt.
            input_names: The inputs to overload at sub-scenario execution.
            output_names: The outputs to get from the sub-scenario execution.
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
                If ``None``, use the name of the scenario adapter
                suffixed by ``"_adapter"``.
            keep_opt_history: Whether to keep databases copies after each execution.
            opt_history_file_prefix: The base name for the databases to be exported.
                The full names of the databases are built from
                the provided base name suffixed by ``"_i.h5"``
                where ``i`` is replaced by the execution number,
                i.e the number of stored databases.
                If empty, the databases are not exported.
                The databases can be exported only is ``keep_opt_history=True``.

        Raises:
            ValueError: If both `reset_x0_before_opt` and `set_x0_before_opt` are True.
        """  # noqa: D205, D212, D415
        if reset_x0_before_opt and set_x0_before_opt:
            raise ValueError("Inconsistent options for MDOScenarioAdapter.")
        self.scenario = scenario
        self._set_x0_before_opt = set_x0_before_opt
        self._set_bounds_before_opt = set_bounds_before_opt
        self._input_names = input_names
        self._output_names = output_names
        self._reset_x0_before_opt = reset_x0_before_opt
        self._output_multipliers = output_multipliers
        self.keep_opt_history = keep_opt_history
        self.databases = []
        self.__opt_history_file_prefix = opt_history_file_prefix

        name = name or f"{scenario.name}_adapter"
        super().__init__(name, cache_type=cache_type, grammar_type=grammar_type)

        self._update_grammars()
        self._dv_in_names = None
        if set_x0_before_opt:
            dv_names = set(self.scenario.formulation.design_space.variables_names)
            self._dv_in_names = list(dv_names & set(self._input_names))

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
        self._initial_x = deepcopy(
            scenario.design_space.get_current_value(as_dict=True)
        )
        self.post_optimal_analysis = None

    def _update_grammars(self) -> None:
        """Update the input and output grammars.

        Raises:
            ValueError: Either if a specified input is missing from the input grammar
                or if a specified output is missing from the output grammar.
        """
        formulation = self.scenario.formulation
        opt_problem = formulation.opt_problem
        top_leveld = formulation.get_top_level_disc()
        for disc in top_leveld:
            self.input_grammar.update(disc.input_grammar)
            self.output_grammar.update(disc.output_grammar)
            # The output may also be the optimum value of the design
            # variables, so the output grammar may contain inputs
            # of the disciplines. All grammars are filtered just after
            # this loop
            self.output_grammar.update(disc.input_grammar)
            self.default_inputs.update(disc.default_inputs)

        try:
            self.input_grammar.restrict_to(self._input_names)
        except KeyError:
            missing_inputs = set(self._input_names) - set(self.input_grammar.keys())

            if missing_inputs:
                raise ValueError(
                    "Can't compute inputs from scenarios: {}.".format(
                        ", ".join(sorted(missing_inputs))
                    )
                )

        # Add the design variables bounds to the input grammar
        if self._set_bounds_before_opt:
            current_x = self.scenario.design_space.get_current_value(as_dict=True)
            names_to_values = dict()
            for suffix in [
                MDOScenarioAdapter.LOWER_BND_SUFFIX,
                MDOScenarioAdapter.UPPER_BND_SUFFIX,
            ]:
                names_to_values.update({k + suffix: v for k, v in current_x.items()})
            bounds_grammar = JSONGrammar("bounds")
            bounds_grammar.update_from_data(names_to_values)
            self.input_grammar.update(bounds_grammar)

        # If a DV is not an input of the top level disciplines:
        missing_outputs = set(self._output_names) - set(self.output_grammar.keys())

        if missing_outputs:
            dv_names = opt_problem.design_space.variables_names
            miss_dvs = set(dv_names) & set(missing_outputs)
            if miss_dvs:
                dv_gram = JSONGrammar("dvs")
                dv_gram.update(miss_dvs)
                self.output_grammar.update(dv_gram)

        try:
            self.output_grammar.restrict_to(self._output_names)
        except KeyError:
            missing_outputs = set(self._output_names) - set(self.output_grammar.keys())

            if missing_outputs:
                raise ValueError(
                    "Can't compute outputs from scenarios: {}.".format(
                        ", ".join(sorted(missing_outputs))
                    )
                )

        # Add the Lagrange multipliers to the output grammar
        if self._output_multipliers:
            self._add_output_multipliers()

    def _add_output_multipliers(self) -> None:
        """Add the Lagrange multipliers of the scenario optimal solution as outputs."""
        # Fill a dictionary with data of typical shapes
        base_dict = dict()
        problem = self.scenario.formulation.opt_problem
        # bound-constraints multipliers
        current_x = problem.design_space.get_current_value(as_dict=True)
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
        multipliers_grammar.update_from_data(base_dict)
        self.output_grammar.update(multipliers_grammar)

    @staticmethod
    def get_bnd_mult_name(
        variable_name: str,
        is_upper: bool,
    ) -> str:
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
        constraint_name: str,
    ) -> str:
        """Return the name of the multiplier of a constraint.

        Args:
            constraint_name: The name of the constraint.

        Returns:
            The name of the multiplier.
        """
        return constraint_name + MDOScenarioAdapter.MULTIPLIER_SUFFIX

    def _run(self) -> None:
        self._pre_run()
        self.scenario.execute()
        self._post_run()

    def _pre_run(self) -> None:
        """Pre-run the scenario."""
        formulation = self.scenario.formulation
        design_space = formulation.opt_problem.design_space
        top_leveld = formulation.get_top_level_disc()

        # Update the top level discipline default inputs with adapter inputs
        # This is the key role of the adapter
        for indata in self._input_names:
            for disc in top_leveld:
                if disc.is_input_existing(indata):
                    disc.default_inputs[indata] = self.local_data[indata]

        if self.scenario.cache is not None:
            # Default inputs have changed, therefore caches shall be cleared
            self.scenario.cache.clear()

        self.scenario.reset_statuses_for_run()

        self._reset_optimization_problem()

        # Set the starting point of the sub scenario with current dv names
        if self._set_x0_before_opt:
            dv_values = {dv_n: self.local_data[dv_n] for dv_n in self._dv_in_names}
            self.scenario.formulation.design_space.set_current_value(dv_values)

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

    def _reset_optimization_problem(self) -> None:
        """Reset the optimization problem."""
        self.scenario.formulation.opt_problem.reset(
            design_space=self._reset_x0_before_opt, database=False, preprocessing=False
        )

    def _post_run(self) -> None:
        """Post-process the scenario."""
        formulation = self.scenario.formulation
        opt_problem = formulation.opt_problem
        design_space = opt_problem.design_space

        if self.keep_opt_history and opt_problem.solution is not None:
            self.databases.append(deepcopy(opt_problem.database))
            if self.__opt_history_file_prefix:
                self.databases[-1].export_hdf(
                    f"{self.__opt_history_file_prefix}_{len(self.databases)}.h5"
                )

        # Test if the last evaluation is the optimum
        x_opt = design_space.get_current_value()
        last_x = opt_problem.database.get_x_by_iter(-1)
        last_eval_not_opt = norm(x_opt - last_x) / (1.0 + norm(last_x)) > 1e-14
        if last_eval_not_opt:
            # Revaluate all functions at optimum
            # To re execute all disciplines and get the right data
            opt_problem.evaluate_functions(x_opt, normalize=False, no_db_no_norm=True)

        # Retrieves top-level discipline outputs
        self._retrieve_top_level_outputs()

        # Compute the Lagrange multipliers and store them in the local data
        if self._output_multipliers:
            self._compute_lagrange_multipliers()

    def _retrieve_top_level_outputs(self) -> None:
        """Retrieve the top-level outputs.

        This method overwrites the adapter outputs with the top-level discipline outputs
        and the optimal design parameters.
        """
        formulation = self.scenario.formulation
        top_level_disciplines = formulation.get_top_level_disc()
        current_x = formulation.opt_problem.design_space.get_current_value(as_dict=True)
        for name in self._output_names:
            for discipline in top_level_disciplines:
                if discipline.is_output_existing(name) and name not in current_x:
                    self.local_data[name] = discipline.local_data[name]

            output_value_in_current_x = current_x.get(name)
            if output_value_in_current_x is not None:
                self.local_data[name] = output_value_in_current_x

    def _compute_lagrange_multipliers(self) -> None:
        """Compute the Lagrange multipliers for the optimal solution of the scenario.

        This method stores the multipliers in the local data.
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

    def get_expected_workflow(self) -> LoopExecSequence:  # noqa: D102
        return self.scenario.get_expected_workflow()

    def get_expected_dataflow(  # noqa: D102
        self,
    ) -> list[tuple[MDODiscipline, MDODiscipline, list[str]]]:
        return self.scenario.get_expected_dataflow()

    def _compute_jacobian(
        self,
        inputs: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
    ) -> None:
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
                or if there is non-differentiable outputs.
        """
        opt_problem = self.scenario.formulation.opt_problem
        objective_names = self.scenario.formulation.opt_problem.objective.outvars
        if len(objective_names) != 1:
            raise ValueError("The objective must be single-valued.")

        # Check the required inputs
        if inputs is None:
            inputs = set(self._input_names + self._bounds_names)
        else:
            not_inputs = set(inputs) - set(self._input_names) - set(self._bounds_names)
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
            not_outputs = sorted(set(outputs) - set(self._output_names))
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
        jacobians = self._compute_auxiliary_jacobians(diff_inputs)

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
        inputs: Iterable[str],
        func_names: Iterable[str] | None = None,
        use_threading: bool = True,
    ) -> dict[str, dict[str, ndarray]]:
        """Compute the Jacobians of the optimization functions.

        Args:
            inputs: The names of the inputs w.r.t. which differentiate.
            func_names: The names of the functions to differentiate
                If None, then all the optimizations functions are differentiated.
            use_threading: Whether to use threads instead of processes
                to parallelize the execution;
                multiprocessing will copy (serialize) all the disciplines,
                while threading will share all the memory.
                This is important to note
                if you want to execute the same discipline multiple times,
                you shall use multiprocessing.

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
        post_opt_data.update(opt_problem.design_space.get_current_value(as_dict=True))
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
        outputs_names: Iterable[str],
    ) -> None:
        """Add outputs to the scenario adapter.

        Args:
            outputs_names: The names of the outputs to be added.
        """
        names_to_add = [
            name for name in outputs_names if name not in self._output_names
        ]
        self._output_names.extend(names_to_add)
        self._update_grammars()


class MDOObjScenarioAdapter(MDOScenarioAdapter):
    """A scenario adapter overwriting the local data with the optimal objective."""

    def _retrieve_top_level_outputs(self) -> None:
        formulation = self.scenario.formulation
        opt_problem = formulation.opt_problem
        top_level_disciplines = formulation.get_top_level_disc()

        # Get the optimal outputs
        optimum = opt_problem.design_space.get_current_value(as_dict=True)
        f_opt = opt_problem.get_optimum()[0]
        if not opt_problem.minimize_objective:
            f_opt = -f_opt
        if not opt_problem.is_mono_objective:
            raise ValueError("The objective function must be single-valued.")

        # Overwrite the adapter local data
        objective = opt_problem.objective.outvars[0]
        if objective in self._output_names:
            self.local_data[objective] = atleast_1d(f_opt)

        for output in self._output_names:
            if output != objective:
                for discipline in top_level_disciplines:
                    if discipline.is_output_existing(output) and output not in optimum:
                        self.local_data[output] = discipline.local_data[output]

                value = optimum.get(output)
                if value is not None:
                    self.local_data[output] = value

    def _compute_jacobian(
        self,
        inputs: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
    ) -> None:
        MDOScenarioAdapter._compute_jacobian(self, inputs, outputs)
        # The gradient of the objective function cannot be computed by the
        # disciplines, but the gradients of the constraints can.
        # The objective function is assumed independent of non-optimization
        # variables.
        obj_name = self.scenario.formulation.opt_problem.objective.outvars[0]
        mult_cstr_jac_key = PostOptimalAnalysis.MULT_DOT_CONSTR_JAC
        self.jac[obj_name] = dict(self.jac[mult_cstr_jac_key])
