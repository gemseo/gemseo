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
"""A BiLevel formulation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.core.chains.chain import MDOChain
from gemseo.core.chains.parallel_chain import MDOParallelChain
from gemseo.core.chains.warm_started_chain import MDOWarmStartedChain
from gemseo.core.coupling_structure import CouplingStructure
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter
from gemseo.formulations.base_mdo_formulation import BaseMDOFormulation
from gemseo.formulations.bilevel_settings import BiLevel_Settings
from gemseo.mda.factory import MDAFactory
from gemseo.scenarios.scenario_results.bilevel_scenario_result import (
    BiLevelScenarioResult,
)
from gemseo.utils.discipline import get_sub_disciplines
from gemseo.utils.string_tools import convert_strings_to_iterable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.discipline import Discipline
    from gemseo.core.grammars.json_grammar import JSONGrammar
    from gemseo.mda.base_mda import BaseMDA
    from gemseo.scenarios.base_scenario import BaseScenario
    from gemseo.typing import StrKeyMapping

LOGGER = logging.getLogger(__name__)


class BiLevel(BaseMDOFormulation):
    """A BiLevel formulation.

    This formulation draws an optimization architecture
    that involves multiple optimization problems to be solved
    to obtain the solution of the MDO problem.

    Here,
    at each iteration on the global design variables,
    the BiLevel MDO formulation implementation performs:

    1. a first MDA to compute the coupling variables,
    2. several disciplinary optimizations on the local design variables in parallel,
    3. a second MDA to update the coupling variables.


    The residual norm of MDA1 and MDA2 can be captured into scenario
    observables thanks to different namespaces :attr:`.BiLevel.MDA1_RESIDUAL_NAMESPACE`
    and :attr:`.BiLevel.MDA2_RESIDUAL_NAMESPACE`.
    """

    DEFAULT_SCENARIO_RESULT_CLASS_NAME: ClassVar[str] = BiLevelScenarioResult.__name__
    """The default name of the scenario results."""

    SYSTEM_LEVEL: ClassVar[str] = "system"
    """The name of the system level."""

    SUBSCENARIOS_LEVEL: ClassVar[str] = "sub-scenarios"
    """The name of the sub-scenarios level."""

    LEVELS = (SYSTEM_LEVEL, SUBSCENARIOS_LEVEL)
    """The collection of levels."""

    CHAIN_NAME: ClassVar[str] = "bilevel_chain"
    """The name of the internal chain."""

    MDA1_RESIDUAL_NAMESPACE: ClassVar[str] = "MDA1"
    """The name of the namespace for the MDA1 residuals."""

    MDA2_RESIDUAL_NAMESPACE: ClassVar[str] = "MDA2"
    """The name of the namespace for the MDA2 residuals."""

    Settings: ClassVar[type[BiLevel_Settings]] = BiLevel_Settings

    chain: MDOChain
    """The chain of the inner problem of the BiLevel formulation
    (MDA1 -> sub-scenarios -> MDA2)"""

    coupling_structure: CouplingStructure
    """The coupling structure between the involved disciplines."""

    _settings: BiLevel_Settings

    _scenario_adapters: list[MDOScenarioAdapter]
    """The adapters of the optimization sub-scenarios."""

    _mda1: BaseMDA | None
    """The first MDA that solves the couplings before sub-scenarios.

    The MDA1 is not built (``None``) if disciplines are not strongly coupled.
    """

    _mda2: BaseMDA
    """The second MDA that solves the couplings after sub-scenarios."""

    __mda_factory: ClassVar[MDAFactory] = MDAFactory()
    """The MDA factory."""

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[Discipline],
        objective_name: str,
        design_space: DesignSpace,
        settings_model: BiLevel_Settings | None = None,
        **settings: Any,
    ) -> None:
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            settings_model=settings_model,
            **settings,
        )
        self._scenario_adapters = []
        self.coupling_structure = CouplingStructure(
            get_sub_disciplines(self.disciplines)
        )
        self._mda1, self._mda2 = self._create_mdas()

        self._create_scenario_adapters(
            reset_x0_before_opt=self._settings.reset_x0_before_opt,
            keep_opt_history=self._settings.keep_opt_history,
            save_opt_history=self._settings.save_opt_history,
            scenario_log_level=self._settings.sub_scenarios_log_level,
            naming=self._settings.naming,
        )

        # Create the inner chain: MDA1 -> sub scenarios -> MDA2
        self.chain = self._create_inner_chain()

        # Cleanup design space
        self._update_design_space()

        # Builds the objective function on top of the chain
        self._build_objective_from_disc(self._objective_name)

    @property
    def mda1(self) -> BaseMDA | None:
        """The MDA1 instance."""
        return self._mda1

    @property
    def mda2(self) -> BaseMDA:
        """The MDA2 instance."""
        return self._mda2

    @property
    def scenario_adapters(self) -> list[MDOScenarioAdapter]:
        """All the adapters that wrap sub-scenarios."""
        return self._scenario_adapters

    def _create_scenario_adapters(
        self,
        output_functions: bool = False,
        adapter_class: type[MDOScenarioAdapter] = MDOScenarioAdapter,
        **adapter_options,
    ) -> None:
        """Create the MDOScenarioAdapter required for each sub scenario.

        This is used to build the self.chain.

        Args:
            output_functions: Whether to add the optimization functions in the adapter
                outputs.
            adapter_class: The class of the adapters.
            **adapter_options: The options for the adapters' initialization.
        """
        for scenario in self.get_sub_scenarios():
            adapter_inputs = self._compute_adapter_inputs(scenario)
            adapter_outputs = self._compute_adapter_outputs(scenario, output_functions)
            adapter = adapter_class(
                scenario,
                adapter_inputs,
                adapter_outputs,
                opt_history_file_prefix=scenario.name,
                **adapter_options,
            )
            self._scenario_adapters.append(adapter)

    def _compute_adapter_outputs(
        self,
        scenario: BaseScenario,
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
        couplings = self.coupling_structure.all_couplings
        mda2_inputs = self._get_mda2_inputs()
        top_disc = scenario.formulation.get_top_level_disciplines()
        top_outputs = [outpt for disc in top_disc for outpt in disc.io.output_grammar]

        # Output couplings of scenario are given to MDA for speedup
        if output_functions:
            opt_problem = scenario.formulation.optimization_problem
            sc_output_names = opt_problem.objective.output_names
            sc_constraints = opt_problem.constraints.get_names()
            sc_out_coupl = sc_output_names + sc_constraints
        else:
            sc_out_coupl = list(set(top_outputs) & set(couplings + mda2_inputs))

        # Add private variables from disciplinary scenario design space
        return sc_out_coupl + scenario.formulation.design_space.variable_names

    def _compute_adapter_inputs(
        self,
        scenario: BaseScenario,
    ) -> list[str]:
        """Compute the scenario adapter inputs.

        Args:
            scenario: A sub-scenario.

        Returns:
            The input variables of the adapter.
        """
        local_dv = [
            var
            for scn in self.get_sub_scenarios()
            for var in scn.design_space.variable_names
        ]
        shared_dv = set(self.optimization_problem.design_space.variable_names)
        couplings = self.coupling_structure.all_couplings
        mda1_outputs = self._get_mda1_outputs()
        top_disc = scenario.formulation.get_top_level_disciplines()
        top_inputs = [inpt for disc in top_disc for inpt in disc.io.input_grammar]

        nonshared_var = set(scenario.design_space.variable_names)

        # All couplings of the scenarios are taken from the MDA
        return list(
            # Add shared variables from system scenario driver
            set(top_inputs)
            & (
                set(couplings)
                | shared_dv
                | set(mda1_outputs)
                | set(local_dv) - nonshared_var
            )
        )

    def _get_mda1_outputs(self) -> list[str]:
        """Return the MDA1 outputs.

        Returns:
             The MDA1 outputs.
        """
        return list(self._mda1.io.output_grammar) if self._mda1 else []

    def _get_mda2_inputs(self) -> list[str]:
        """Return the MDA2 inputs.

        Returns:
             The MDA2 inputs.
        """
        return list(self._mda2.io.input_grammar) if self._mda2 else []

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
            msg = (
                "'main_mda_name' option is required to deduce the "
                "sub options of BiLevel."
            )
            raise ValueError(msg)
        return cls.__mda_factory.get_options_grammar(main_mda_name)

    @classmethod
    def get_default_sub_option_values(cls, **options: str) -> StrKeyMapping:
        """
        Raises:
            ValueError: When the MDA name is not provided.
        """  # noqa: D205, D212, D415
        main_mda_name = options.get("main_mda_name")
        if main_mda_name is None:
            msg = (
                "'main_mda_name' option is required to deduce the "
                "sub options of BiLevel."
            )
            raise ValueError(msg)
        return cls.__mda_factory.get_default_option_values(main_mda_name)

    def _create_mdas(self) -> tuple[BaseMDA | None, BaseMDA]:
        """Build the chain on top of which all functions are built.

        This chain is as follows: (MDA1 ->) MDOScenarios -> MDA2.

        Returns:
            The first MDA, in presence of strongly coupled discipline,
            and the second MDA.
        """
        mda1 = None
        strongly_coupled_disciplines = (
            self.coupling_structure.strongly_coupled_disciplines
        )
        if len(strongly_coupled_disciplines) > 0:
            mda1 = self.__mda_factory.create(
                self._settings.main_mda_name,
                strongly_coupled_disciplines,
                settings_model=self._settings.main_mda_settings,
            )
            mda1.settings.warm_start = True

        else:
            LOGGER.warning(
                "No strongly coupled disciplines detected, "
                "MDA1 is disabled in the BiLevel formulation"
            )

        mda2 = self.__mda_factory.create(
            self._settings.main_mda_name,
            get_sub_disciplines(self.disciplines),
            settings_model=self._settings.main_mda_settings,
        )
        mda2.settings.warm_start = False

        return mda1, mda2

    def _create_inner_chain(self) -> MDOChain:
        """Create the inner chain.

        This chain is: MDA -> MDOScenarios -> MDA.

        Returns:
            The multidisciplinary chain.
        """
        chain_dis = [] if self._mda1 is None else [self._mda1]

        chain_dis += [self._create_sub_scenarios_chain()]

        if self._mda2:
            chain_dis += [self._mda2]

        if self._settings.reset_x0_before_opt:
            return MDOChain(chain_dis, name=self.CHAIN_NAME)

        return MDOWarmStartedChain(
            chain_dis,
            name=self.CHAIN_NAME,
            variable_names_to_warm_start=self._get_variable_names_to_warm_start(),
        )

    def _create_sub_scenarios_chain(self) -> MDOChain | MDOParallelChain:
        """Create the chain of sub-scenarios.

        Returns:
            The chain of sub-scenarios,
            either parallel or sequential.
        """
        if self._settings.parallel_scenarios:
            return MDOParallelChain(
                self.scenario_adapters,
                use_threading=self._settings.multithread_scenarios,
            )
        return MDOChain(self.scenario_adapters)

    def _get_variable_names_to_warm_start(self) -> list[str]:
        """Retrieve the names of the variables to warm start.

        The outputs of all the sub scenarios that shall be warm started.

        Returns:
            The names of the variables to warm start.
        """
        variable_names = [
            name
            for adapter in self.scenario_adapters
            for name in adapter.io.output_grammar
        ]
        if self._mda1:
            for variable_name in self._mda1.io.output_grammar:
                if variable_name not in variable_names:
                    variable_names.append(variable_name)
        if self._mda2:
            for variable_name in self._mda2.io.output_grammar:
                if variable_name not in variable_names:
                    variable_names.append(variable_name)

        return [
            variable_name
            for variable_name in variable_names
            if variable_name not in self.design_space.variable_names
        ]

    def _update_design_space(self) -> None:
        """Update the design space by removing the coupling variables."""
        self._set_default_input_values_from_design_space()
        self._remove_sub_scenario_dv_from_ds()
        self._remove_couplings_from_ds()
        self._remove_unused_variables()

    def _remove_couplings_from_ds(self) -> None:
        """Removes the coupling variables from the design space."""
        if hasattr(self._mda2.settings, "coupling_structure"):
            # Otherwise, the MDA2 may be a user provided MDA
            # Which manages the couplings internally
            couplings = self.mda2.coupling_structure.strong_couplings
            design_space = self.optimization_problem.design_space
            for coupling in couplings:
                if coupling in design_space:
                    LOGGER.warning(
                        "The coupling variable %s was removed from the design space.",
                        coupling,
                    )
                    design_space.remove_variable(coupling)

    def get_top_level_disciplines(self) -> tuple[Discipline]:  # noqa:D102
        return (self.chain,)

    def add_constraint(
        self,
        output_name: str,
        constraint_type: MDOFunction.ConstraintType = MDOFunction.ConstraintType.EQ,
        constraint_name: str = "",
        value: float = 0,
        positive: bool = False,
        levels: list[str] = (),
    ) -> None:
        """Add a constraint to the formulation.

        Args:
            levels: The levels at which the constraint is to be added
                (sublist of :attr:`.LEVELS`).
                By default, the policy set at the initialization
                of the formulation is enforced.

        Raises:
            ValueError: When the constraint levels are not a sublist of BiLevel.LEVELS.
        """
        # If the constraint levels are not specified the initial policy is enforced.
        if not levels:
            if self._settings.apply_cstr_to_system:
                self._add_system_level_constraint(
                    output_name,
                    constraint_type=constraint_type,
                    constraint_name=constraint_name,
                    value=value,
                    positive=positive,
                )
            if self._settings.apply_cstr_tosub_scenarios:
                self._add_sub_level_constraint(
                    output_name,
                    constraint_type=constraint_type,
                    constraint_name=constraint_name,
                    value=value,
                    positive=positive,
                )
        # Otherwise the constraint is applied at the specified levels.
        elif not isinstance(levels, list) or not set(levels) <= set(BiLevel.LEVELS):
            msg = f"Constraint levels must be a sublist of {BiLevel.LEVELS}"
            raise ValueError(msg)
        elif not levels:
            LOGGER.warning("Empty list of constraint levels, constraint not added")
        else:
            if BiLevel.SYSTEM_LEVEL in levels:
                self._add_system_level_constraint(
                    output_name,
                    constraint_type=constraint_type,
                    constraint_name=constraint_name,
                    value=value,
                    positive=positive,
                )
            if BiLevel.SUBSCENARIOS_LEVEL in levels:
                self._add_sub_level_constraint(
                    output_name,
                    constraint_type=constraint_type,
                    constraint_name=constraint_name,
                    value=value,
                    positive=positive,
                )

    def _add_system_level_constraint(
        self,
        output_name: str,
        constraint_type: MDOFunction.ConstraintType = MDOFunction.ConstraintType.EQ,
        constraint_name: str = "",
        value: float = 0,
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
                If empty, the name is generated from the output name.
            value: The value of activation of the constraint.
            positive: Whether the inequality constraint is positive.
        """
        super().add_constraint(
            output_name,
            constraint_type=constraint_type,
            constraint_name=constraint_name,
            value=value,
            positive=positive,
        )

    def _add_sub_level_constraint(
        self,
        output_name: str,
        constraint_type: MDOFunction.ConstraintType = MDOFunction.ConstraintType.EQ,
        constraint_name: str = "",
        value: float = 0,
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
                If empty, the name is generated from the output name.
            value: The value of activation of the constraint.
            positive: Whether the inequality constraint is positive.

        Raises:
            ValueError: If a constraint is not found in the scenario
                top-level disciplines outputs.
        """
        added = False
        output_names = convert_strings_to_iterable(output_name)
        for sub_scenario in self.get_sub_scenarios():
            if self._scenario_computes_outputs(sub_scenario, output_names):
                sub_scenario.add_constraint(
                    output_names,
                    constraint_type=constraint_type,
                    constraint_name=constraint_name,
                    value=value,
                    positive=positive,
                )
                added = True
        if not added:
            msg = (
                f"No sub scenario has an output named {output_name} "
                "cannot create such a constraint."
            )
            raise ValueError(msg)

    @staticmethod
    def _scenario_computes_outputs(
        scenario: BaseScenario,
        output_names: Iterable[str],
    ) -> bool:
        """Check if the top level disciplines compute the given outputs.

        Args:
            output_names: The names of the variable to check.
            scenario: The scenario to be tested.

        Returns:
            True if the top level disciplines compute the given outputs.
        """
        for disc in scenario.formulation.get_top_level_disciplines():
            if disc.io.output_grammar.has_names(output_names):
                return True
        return False
