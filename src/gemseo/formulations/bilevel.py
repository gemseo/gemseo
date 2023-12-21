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
"""A Bi-level formulation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.core.chain import MDOChain
from gemseo.core.chain import MDOParallelChain
from gemseo.core.chain import MDOWarmStartedChain
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from gemseo.core.formulation import MDOFormulation
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter
from gemseo.mda.mda_factory import MDAFactory
from gemseo.scenarios.scenario_results.bilevel_scenario_result import (
    BiLevelScenarioResult,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.execution_sequence import ExecutionSequence
    from gemseo.core.grammars.json_grammar import JSONGrammar
    from gemseo.core.scenario import Scenario
    from gemseo.mda.mda import MDA

LOGGER = logging.getLogger(__name__)


class BiLevel(MDOFormulation):
    """A bi-level formulation.

    This formulation draws an optimization architecture
    that involves multiple optimization problems to be solved
    to obtain the solution of the MDO problem.

    Here,
    at each iteration on the global design variables,
    the bi-level MDO formulation implementation performs:

    1. a first MDA to compute the coupling variables,
    2. several disciplinary optimizations on the local design variables in parallel,
    3. a second MDA to update the coupling variables.
    """

    DEFAULT_SCENARIO_RESULT_CLASS_NAME: ClassVar[str] = BiLevelScenarioResult.__name__

    SYSTEM_LEVEL = "system"
    SUBSCENARIOS_LEVEL = "sub-scenarios"
    LEVELS = (SYSTEM_LEVEL, SUBSCENARIOS_LEVEL)

    __sub_scenarios_log_level: int | None
    """The level of the root logger during the sub-scenarios executions.

    If ``None``, do not change the level of the root logger.
    """

    def __init__(
        self,
        disciplines: list[MDODiscipline],
        objective_name: str,
        design_space: DesignSpace,
        maximize_objective: bool = False,
        main_mda_name: str = "MDAChain",
        inner_mda_name: str = "MDAJacobi",
        parallel_scenarios: bool = False,
        multithread_scenarios: bool = True,
        apply_cstr_tosub_scenarios: bool = True,
        apply_cstr_to_system: bool = True,
        reset_x0_before_opt: bool = False,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        sub_scenarios_log_level: int | None = None,
        **main_mda_options: Any,
    ) -> None:
        """
        Args:
            main_mda_name: The name of the class used for the main MDA,
                typically the :class:`.MDAChain`,
                but one can force to use :class:`.MDAGaussSeidel` for instance.
            inner_mda_name: The name of the class used for the inner-MDA of the main
                MDA, if any; typically when the main MDA is an :class:`.MDAChain`.
            parallel_scenarios: Whether to run the sub-scenarios in parallel.
            multithread_scenarios: If ``True`` and parallel_scenarios=True,
                the sub-scenarios are run in parallel using multi-threading;
                if False and parallel_scenarios=True, multiprocessing is used.
            apply_cstr_tosub_scenarios: Whether the :meth:`.add_constraint` method
                adds the constraint to the optimization problem of the sub-scenario
                capable of computing the constraint.
            apply_cstr_to_system: Whether the :meth:`.add_constraint` method adds
                the constraint to the optimization problem of the system scenario.
            reset_x0_before_opt: Whether to restart the sub optimizations
                from the initial guesses, otherwise warm start them.
            sub_scenarios_log_level: The level of the root logger
                during the sub-scenarios executions.
                If ``None``, do not change the level of the root logger.
            **main_mda_options: The options of the main MDA, which may include those
                of the inner-MDA.
        """  # noqa: D205, D212, D415
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            maximize_objective=maximize_objective,
            grammar_type=grammar_type,
        )
        self._shared_dv = list(design_space.variable_names)
        self._mda1 = None
        self._mda2 = None
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
        self.__sub_scenarios_log_level = sub_scenarios_log_level
        self._build_mdas(main_mda_name, inner_mda_name, **main_mda_options)

        # Create MDOChain : MDA1 -> sub scenarios -> MDA2
        self._build_chain()

        # Cleanup design space
        self._update_design_space()

        # Builds the objective function on top of the chain
        self._build_objective_from_disc(self._objective_name)

    @property
    def mda1(self) -> MDODiscipline:
        """The MDA1 instance."""
        return self._mda1

    @property
    def mda2(self) -> MDODiscipline:
        """The MDA2 instance."""
        return self._mda2

    def _build_scenario_adapters(
        self,
        output_functions: bool = False,
        use_non_shared_vars: bool = False,
        adapter_class: type[MDOScenarioAdapter] = MDOScenarioAdapter,
        **adapter_options,
    ) -> list[MDOScenarioAdapter]:
        """Build the MDOScenarioAdapter required for each sub scenario.

        This is used to build the self.chain.

        Args:
            output_functions: Whether to add the optimization functions in the adapter
                outputs.
            use_non_shared_vars: Whether the non-shared design variables are inputs
                of the scenarios adapters.
            adapter_class: The class of the adapters.
            **adapter_options: The options for the adapters' initialization.

        Returns:
            The adapters for the sub-scenarios.
        """
        adapters = []
        scenario_log_level = adapter_options.pop(
            "scenario_log_level", self.__sub_scenarios_log_level
        )
        for scenario in self.get_sub_scenarios():
            adapter_inputs = self._compute_adapter_inputs(scenario, use_non_shared_vars)
            adapter_outputs = self._compute_adapter_outputs(scenario, output_functions)
            adapter = adapter_class(
                scenario,
                adapter_inputs,
                adapter_outputs,
                grammar_type=self._grammar_type,
                scenario_log_level=scenario_log_level,
                **adapter_options,
            )
            adapters.append(adapter)
        return adapters

    def _compute_adapter_outputs(
        self,
        scenario: Scenario,
        output_functions: bool,
    ) -> list[str]:
        """Compute the scenario adapter outputs.

        Args:
             scenario: A sub-scenario.
             output_functions: Whether to add the objective and constraints in the
                 outputs.

        Returns:
            The output variables of the adapter.
        """
        couplings = self.couplstr.all_couplings
        mda2_inputs = self._get_mda2_inputs()
        top_disc = scenario.formulation.get_top_level_disc()
        top_outputs = [
            outpt for disc in top_disc for outpt in disc.get_output_data_names()
        ]

        # Output couplings of scenario are given to MDA for speedup
        if output_functions:
            opt_problem = scenario.formulation.opt_problem
            sc_output_names = opt_problem.objective.output_names
            sc_constraints = opt_problem.get_constraint_names()
            sc_out_coupl = sc_output_names + sc_constraints
        else:
            sc_out_coupl = list(set(top_outputs) & set(couplings + mda2_inputs))

        # Add private variables from disciplinary scenario design space
        return sc_out_coupl + scenario.design_space.variable_names

    def _compute_adapter_inputs(
        self,
        scenario: Scenario,
        use_non_shared_vars: bool,
    ) -> list[str]:
        """Compute the scenario adapter inputs.

        Args:
            scenario: A sub-scenario.
            use_non_shared_vars: Whether to add the non-shared variables
                as inputs of the adapter.

        Returns:
            The input variables of the adapter.
        """
        shared_dv = set(self._shared_dv)
        couplings = self.couplstr.all_couplings
        mda1_outputs = self._get_mda1_outputs()
        top_disc = scenario.formulation.get_top_level_disc()
        top_inputs = [inpt for disc in top_disc for inpt in disc.get_input_data_names()]

        # All couplings of the scenarios are taken from the MDA
        adapter_inputs = list(
            # Add shared variables from system scenario driver
            set(top_inputs) & (set(couplings) | shared_dv | set(mda1_outputs))
        )
        if use_non_shared_vars:
            nonshared_var = scenario.design_space.variable_names
            adapter_inputs = list(
                set(adapter_inputs) | set(top_inputs) & set(nonshared_var)
            )
        return adapter_inputs

    def _get_mda1_outputs(self) -> list[str]:
        """Return the MDA1 outputs.

        Returns:
             The MDA1 outputs.
        """
        return list(self._mda1.get_output_data_names()) if self._mda1 else []

    def _get_mda2_inputs(self) -> list[str]:
        """Return the MDA2 inputs.

        Returns:
             The MDA2 inputs.
        """
        return list(self._mda2.get_input_data_names()) if self._mda2 else []

    @classmethod
    def get_sub_options_grammar(cls, **options: str) -> JSONGrammar:
        """Return the grammar of the selected MDA.

        Args:
            **options: The options of the BiLevel formulation.

        Returns:
            The MDA grammar.

        Raises:
            ValueError: When the MDA name is not provided.
        """
        main_mda_name = options.get("main_mda_name")
        if main_mda_name is None:
            raise ValueError(
                "'main_mda_name' option is required to deduce the "
                "sub options of BiLevel."
            )
        return MDAFactory().get_options_grammar(main_mda_name)

    @classmethod
    def get_default_sub_option_values(
        cls, **options: str
    ) -> Mapping[str, str | int | float | bool | None] | None:
        """
        Raises:
            ValueError: When the MDA name is not provided.
        """  # noqa: D205, D212, D415
        main_mda_name = options.get("main_mda_name")
        if main_mda_name is None:
            raise ValueError(
                "'main_mda_name' option is required to deduce the "
                "sub options of BiLevel."
            )
        return MDAFactory().get_default_option_values(main_mda_name)

    def _build_mdas(
        self,
        main_mda_name: str,
        inner_mda_name: str,
        **main_mda_options: str | int | float | bool | None,
    ) -> None:
        """Build the chain on top of which all functions are built.

        This chain is as follows: MDA1 -> MDOScenarios -> MDA2.

        Args:
            main_mda_name: The class name of the main MDA.
            inner_mda_name: The name of the class used for the inner-MDA of the main
                MDA, if any; typically when the main MDA is an :class:`.MDAChain`.
            **main_mda_options: The options of the main MDA, which may include those
                of the inner-MDA.
        """
        if main_mda_name == "MDAChain":
            main_mda_options["inner_mda_name"] = inner_mda_name

        disc_mda1 = self.couplstr.strongly_coupled_disciplines
        if len(disc_mda1) > 0:
            self._mda1 = self._mda_factory.create(
                main_mda_name,
                disc_mda1,
                grammar_type=self._grammar_type,
                **main_mda_options,
            )
            self._mda1.warm_start = True
        else:
            LOGGER.warning(
                "No strongly coupled disciplines detected, "
                " MDA1 is deactivated in the BiLevel formulation"
            )

        disc_mda2 = self.get_sub_disciplines()
        self._mda2 = self._mda_factory.create(
            main_mda_name,
            disc_mda2,
            grammar_type=self._grammar_type,
            **main_mda_options,
        )

        self._mda2.warm_start = False

    def _build_chain_dis_sub_opts(
        self,
    ) -> tuple[list | MDA, list[MDOScenarioAdapter]]:
        """Initialize the chain of disciplines and the sub-scenarios.

        Returns:
            The first MDA (if exists) and the sub-scenarios.
        """
        chain_dis = []
        if self._mda1 is not None:
            chain_dis = [self._mda1]
        sub_opts = self.scenario_adapters
        return chain_dis, sub_opts

    def _build_chain(self) -> None:
        """Build the chain on top of which all functions are built.

        This chain is: MDA -> MDOScenarios -> MDA.
        """
        # Build the scenario adapters to be chained with MDAs
        self.scenario_adapters = self._build_scenario_adapters(
            reset_x0_before_opt=self.reset_x0_before_opt, keep_opt_history=True
        )
        chain_dis, sub_opts = self._build_chain_dis_sub_opts()

        if self.__parallel_scenarios:
            use_threading = self._multithread_scenarios
            par_chain = MDOParallelChain(
                sub_opts, use_threading=use_threading, grammar_type=self._grammar_type
            )
            chain_dis += [par_chain]
        else:
            # Chain MDA -> scenarios exec -> MDA
            chain_dis += sub_opts

        # Add MDA2 if needed
        if self._mda2:
            chain_dis += [self._mda2]

        if not self.reset_x0_before_opt and self._mda1 is not None:
            self.chain = MDOWarmStartedChain(
                chain_dis,
                name="bilevel_chain",
                grammar_type=self._grammar_type,
                variable_names_to_warm_start=self._get_variable_names_to_warm_start(),
            )
        else:
            self.chain = MDOChain(
                chain_dis, name="bilevel_chain", grammar_type=self._grammar_type
            )

    def _get_variable_names_to_warm_start(self) -> list[str]:
        """Retrieve the names of the variables to warm start.

        The outputs of all the sub scenarios that shall be warm started.

        Returns:
            The names of the variables to warm start.
        """
        return [
            name
            for adapter in self.scenario_adapters
            for name in adapter.get_output_data_names()
        ]

    def _update_design_space(self) -> None:
        """Update the design space by removing the coupling variables."""
        self._set_default_input_values_from_design_space()
        self._remove_sub_scenario_dv_from_ds()
        self._remove_couplings_from_ds()
        self._remove_unused_variables()

    def _remove_couplings_from_ds(self) -> None:
        """Removes the coupling variables from the design space."""
        if hasattr(self._mda2, "strong_couplings"):
            # Otherwise, the MDA2 may be a user provided MDA
            # Which manages the couplings internally
            couplings = self.mda2.strong_couplings
            design_space = self.opt_problem.design_space
            for coupling in couplings:
                if coupling in design_space.variable_names:
                    design_space.remove_variable(coupling)

    def get_top_level_disc(self) -> list[MDODiscipline]:  # noqa:D102
        return [self.chain]

    def get_expected_workflow(  # noqa:D102
        self,
    ) -> list[ExecutionSequence, tuple[ExecutionSequence]]:
        return self.chain.get_expected_workflow()

    def get_expected_dataflow(  # noqa:D102
        self,
    ) -> list[tuple[MDODiscipline, MDODiscipline, list[str]]]:
        return self.chain.get_expected_dataflow()

    def add_constraint(
        self,
        output_name: str,
        constraint_type: MDOFunction.ConstraintType = MDOFunction.ConstraintType.EQ,
        constraint_name: str | None = None,
        value: float | None = None,
        positive: bool = False,
        levels: list[str] | None = None,
    ) -> None:
        """Add a constraint to the formulation.

        Args:
            levels: The levels at which the constraint is to be added
                (sublist of Bilevel.LEVELS).
                By default, the policy set at the initialization
                of the formulation is enforced.

        Raises:
            ValueError: When the constraint levels are not a sublist of BiLevel.LEVELS.
        """
        # If the constraint levels are not specified the initial policy is enforced.
        if levels is None:
            if self._apply_cstr_to_system:
                self._add_system_level_constraint(
                    output_name, constraint_type, constraint_name, value, positive
                )
            if self._apply_cstr_tosub_scenarios:
                self._add_sub_level_constraint(
                    output_name, constraint_type, constraint_name, value, positive
                )
        # Otherwise the constraint is applied at the specified levels.
        elif not isinstance(levels, list) or not set(levels) <= set(BiLevel.LEVELS):
            raise ValueError(f"Constraint levels must be a sublist of {BiLevel.LEVELS}")
        elif not levels:
            LOGGER.warning("Empty list of constraint levels, constraint not added")
        else:
            if BiLevel.SYSTEM_LEVEL in levels:
                self._add_system_level_constraint(
                    output_name, constraint_type, constraint_name, value, positive
                )
            if BiLevel.SUBSCENARIOS_LEVEL in levels:
                self._add_sub_level_constraint(
                    output_name, constraint_type, constraint_name, value, positive
                )

    def _add_system_level_constraint(
        self,
        output_name: str,
        constraint_type: MDOFunction.ConstraintType = MDOFunction.ConstraintType.EQ,
        constraint_name: str | None = None,
        value: float | None = None,
        positive: bool = False,
    ) -> None:
        """Add a constraint at the system level.

        Args:
            output_name: The name of the output to be used as a constraint.
                For instance, if g_1 is given and constraint_type="eq",
                g_1=0 will be added as a constraint to the optimizer.
            constraint_type: The type of constraint,
                either "eq" for equality constraint or "ineq" for inequality constraint.
            constraint_name: The name of the constraint to be stored,
                If ``None``, the name is generated from the output name.
            value: The value of activation of the constraint.
                If ``None``, the value is equal to 0.
            positive: Whether the inequality constraint is positive.
        """
        super().add_constraint(
            output_name, constraint_type, constraint_name, value, positive
        )

    def _add_sub_level_constraint(
        self,
        output_name: str,
        constraint_type: MDOFunction.ConstraintType = MDOFunction.ConstraintType.EQ,
        constraint_name: str | None = None,
        value: float | None = None,
        positive: bool = False,
    ) -> None:
        """Add a constraint at the sub-scenarios level.

        Args:
            output_name: The name of the output to be used as a constraint.
                For instance, if g_1 is given and constraint_type="eq",
                g_1=0 will be added as a constraint to the optimizer.
            constraint_type: The type of constraint,
                either "eq" for equality constraint or "ineq" for inequality constraint.
            constraint_name: The name of the constraint to be stored,
                If ``None``, the name is generated from the output name.
            value: The value of activation of the constraint.
                If ``None``, the value is equal to 0.
            positive: Whether the inequality constraint is positive.

        Raises:
            ValueError: If a constraint is not found in the scenario
                top-level disciplines outputs.
        """
        added = False
        output_names = self._check_add_cstr_input(output_name, constraint_type)
        for sub_scenario in self.get_sub_scenarios():
            if self._scenario_computes_outputs(sub_scenario, output_names):
                sub_scenario.add_constraint(
                    output_names, constraint_type, constraint_name, value, positive
                )
                added = True
        if not added:
            raise ValueError(
                f"No sub scenario has an output named {output_name} "
                "cannot create such a constraint."
            )

    @staticmethod
    def _scenario_computes_outputs(
        scenario: Scenario,
        output_names: Iterable[str],
    ) -> bool:
        """Check if the top level disciplines compute the given outputs.

        Args:
            output_names: The names of the variable to check.
            scenario: The scenario to be tested.

        Returns:
            True if the top level disciplines compute the given outputs.
        """
        for disc in scenario.formulation.get_top_level_disc():
            if disc.is_all_outputs_existing(output_names):
                return True
        return False
