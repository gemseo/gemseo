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
from typing import ClassVar

from gemseo.core.chains.chain import DisciplineChain
from gemseo.core.chains.parallel_chain import ParallelDisciplineChain
from gemseo.core.chains.warm_started_chain import WarmStartedDisciplineChain
from gemseo.core.coupling_structure import CouplingStructure
from gemseo.core.functions.array_function import ArrayFunction
from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter
from gemseo.formulations.base_mdo import BaseMDOFormulation
from gemseo.formulations.bilevel_settings import BiLevel_Settings
from gemseo.mda.base import BaseMDA
from gemseo.mda.factory import MDA_FACTORY
from gemseo.scenarios.scenario_results.bilevel_scenario_result import (
    BiLevelScenarioResult,
)
from gemseo.utils.discipline import get_sub_disciplines

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.algos.database import DatabaseKeyType
    from gemseo.core.discipline import Discipline
    from gemseo.core.grammars.json import JSONGrammar
    from gemseo.scenarios.mdo import MDOScenario
    from gemseo.typing import StrKeyMapping

LOGGER = logging.getLogger(__name__)


class BiLevel(BaseMDOFormulation[BiLevel_Settings]):
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
    observables thanks to different namespaces
    [MDA1_RESIDUAL_NAMESPACE][gemseo.formulations.bilevel.BiLevel.MDA1_RESIDUAL_NAMESPACE]
    and
    [MDA2_RESIDUAL_NAMESPACE][gemseo.formulations.bilevel.BiLevel.MDA2_RESIDUAL_NAMESPACE].

    Both MDAs are optional.
    """

    DEFAULT_SCENARIO_RESULT_CLASS_NAME: ClassVar[str] = BiLevelScenarioResult.__name__
    """The default name of the scenario results."""

    CHAIN_NAME: ClassVar[str] = "bilevel_chain"
    """The name of the internal chain."""

    MDA1_RESIDUAL_NAMESPACE: ClassVar[str] = "MDA1"
    """The name of the namespace for the MDA1 residuals."""

    MDA2_RESIDUAL_NAMESPACE: ClassVar[str] = "MDA2"
    """The name of the namespace for the MDA2 residuals."""

    settings_class: ClassVar[type[BiLevel_Settings]] = BiLevel_Settings

    chain: DisciplineChain
    """The chain of the inner problem of the BiLevel formulation
    (MDA1 -> sub-scenarios -> MDA2)"""

    coupling_structure: CouplingStructure
    """The coupling structure between the involved disciplines."""

    _disciplines_as_sub_scenario: Sequence[Discipline]
    """The disciplines to be treated as sub-scenarios and not as disciplines."""

    _scenario_adapters: list[MDOScenarioAdapter]
    """The adapters of the optimization sub-scenarios."""

    _mda1: BaseMDA | None
    """The first MDA that solves the couplings before sub-scenarios.

    The MDA1 is not built (`None`) if disciplines are not strongly coupled.
    """

    _mda2: BaseMDA
    """The second MDA that solves the couplings after sub-scenarios."""

    def _create_multidisciplinary_process(self) -> None:
        if not self._settings.use_mda1:
            LOGGER.warning(
                "The first MDA has been deactivated in the Bilevel formulation. "
                "This may lead to premature convergence or "
                "an inconsistent solution. This setting should be "
                "used only when the couplings are handled in "
                "the sub-scenarios."
            )
        self._disciplines_as_sub_scenario = self._settings.disciplines_as_sub_scenario
        self._scenario_adapters = []
        self.coupling_structure = CouplingStructure(
            get_sub_disciplines(self.disciplines)
        )
        self._mda1, self._mda2 = self._create_mdas()

        self._create_scenario_adapters(
            reset_x0_before_opt=self._settings.reset_x0_before_opt,
            set_x0_before_opt=self._settings.set_x0_before_opt,
            keep_opt_history=self._settings.keep_opt_history,
            save_opt_history=self._settings.save_opt_history,
            scenario_log_level=self._settings.sub_scenarios_log_level,
            naming=self._settings.naming,
        )

        # Create the inner chain: MDA1 -> sub scenarios -> MDA2
        self.chain = self._create_inner_chain()

        self.problem.database.add_new_iter_listener(
            self._store_optimal_local_design_values
        )

    @property
    def mda1(self) -> BaseMDA | None:
        """The MDA1 instance."""
        return self._mda1

    @property
    def mda2(self) -> BaseMDA:
        """The MDA2 instance."""
        return self._mda2

    @property
    def scenario_adapters(self) -> list[MDOScenarioAdapter | Discipline]:
        """All the adapters that wrap sub-scenarios."""
        return self._scenario_adapters + self._settings.disciplines_as_sub_scenario

    def _create_scenario_adapters(
        self,
        adapter_class: type[MDOScenarioAdapter] = MDOScenarioAdapter,
        **adapter_options,
    ) -> None:
        """Create the MDOScenarioAdapter required for each sub scenario.

        This is used to build the self.chain.

        Args:
            adapter_class: The class of the adapters.
            **adapter_options: The options for the adapters' initialization.
        """
        for scenario in self.get_sub_scenarios():
            input_names = self._compute_adapter_inputs(scenario)
            output_names = self._compute_adapter_outputs(scenario)
            adapter = adapter_class(
                scenario,
                input_names,
                output_names,
                opt_history_file_prefix=scenario.name,
                **adapter_options,
            )
            self._scenario_adapters.append(adapter)

    def _compute_adapter_outputs(self, scenario: MDOScenario) -> list[str]:
        """Compute the scenario adapter outputs.

        Args:
             scenario: A sub-scenario.

        Returns:
            The output variables of the adapter.
        """
        couplings = self.coupling_structure.all_couplings
        mda2_inputs = self._get_mda2_inputs()
        top_disc = scenario.formulation.get_top_level_disciplines()
        top_outputs = [output for disc in top_disc for output in disc.io.output_grammar]
        sc_out_coupl = list(set(top_outputs) & set(couplings + mda2_inputs))
        adapter_outputs = (
            sc_out_coupl + scenario.formulation.design_space.variable_names
        )
        if not self._mda2:
            top_disc = scenario.formulation.get_top_level_disciplines()
            top_outputs = [
                outpt for disc in top_disc for outpt in disc.io.output_grammar.names
            ]
            adapter_outputs += top_outputs
        return adapter_outputs

    def _compute_adapter_inputs(
        self,
        scenario: MDOScenario,
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
        shared_dv = set(self.problem.design_space.variable_names)
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
        return MDA_FACTORY.get_options_grammar(main_mda_name)

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
        return MDA_FACTORY.get_default_option_values(main_mda_name)

    def _create_mdas(self) -> tuple[BaseMDA | None, BaseMDA]:
        """Build the chain on top of which all functions are built.

        This chain is as follows: (MDA1 ->) MDOScenarios -> MDA2.

        Returns:
            The first MDA, in presence of strongly coupled discipline,
            and the second MDA.
        """
        mda1 = None
        mda2 = None
        strongly_coupled_disciplines = (
            self.coupling_structure.strongly_coupled_disciplines
        )

        if self._settings.use_mda1:
            if (mda1 := self._settings.mda1_instance) is None:
                if len(strongly_coupled_disciplines) > 0:
                    mda1 = MDA_FACTORY.create(
                        self._settings.main_mda_settings.target_class_name,
                        strongly_coupled_disciplines,
                        settings=self._settings.main_mda_settings,
                    )
                    mda1.settings.warm_start = True
                else:
                    LOGGER.warning(
                        "No strongly coupled disciplines detected, "
                        "MDA1 is disabled in the BiLevel formulation"
                    )
            else:
                LOGGER.info("Using the provided MDA1 instance.")

        if self._settings.use_mda2:
            if (mda2 := self._settings.mda2_instance) is None:
                mda2 = MDA_FACTORY.create(
                    self._settings.main_mda_settings.target_class_name,
                    get_sub_disciplines(self.disciplines),
                    settings=self._settings.main_mda_settings,
                )
                mda2.settings.warm_start = False
            else:
                LOGGER.info("Using the provided MDA2 instance.")

        return mda1, mda2

    def _create_inner_chain(self) -> DisciplineChain:
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
            return DisciplineChain(chain_dis, name=self.CHAIN_NAME)

        return WarmStartedDisciplineChain(
            chain_dis,
            name=self.CHAIN_NAME,
            variable_names_to_warm_start=self._get_variable_names_to_warm_start(),
        )

    def _create_sub_scenarios_chain(self) -> DisciplineChain | ParallelDisciplineChain:
        """Create the chain of sub-scenarios.

        Returns:
            The chain of sub-scenarios,
            either parallel or sequential.
        """
        if self._settings.parallel_scenarios:
            return ParallelDisciplineChain(
                self.scenario_adapters,
                use_threading=self._settings.multithread_scenarios,
            )
        return DisciplineChain(self.scenario_adapters)

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
        self._set_default_input_values_from_design_space()
        self._remove_sub_scenario_dv_from_ds()
        self._remove_couplings_from_ds()
        self._remove_unused_variables()

    def _remove_couplings_from_ds(self) -> None:
        """Removes the coupling variables from the design space."""
        if not isinstance(self._mda2, BaseMDA):
            return

        if hasattr(self._mda2.settings, "coupling_structure"):
            # Otherwise, the MDA2 may be a user provided MDA
            # Which manages the couplings internally
            couplings = self.mda2.coupling_structure.strong_couplings
            design_space = self.problem.design_space
            for coupling in couplings:
                if coupling in design_space:
                    LOGGER.warning(
                        "The coupling variable %s was removed from the design space.",
                        coupling,
                    )
                    design_space.remove_variable(coupling)

    def get_top_level_disciplines(  # noqa:D102
        self, include_sub_formulations: bool = False
    ) -> tuple[Discipline, ...]:
        if include_sub_formulations:
            return (
                self.chain,
                *(
                    discipline
                    for scenario_adapter in self._scenario_adapters
                    for discipline in scenario_adapter.scenario.formulation.get_top_level_disciplines(  # noqa: E501
                        include_sub_formulations=include_sub_formulations
                    )
                ),
            )

        return (self.chain,)

    def create_constraint(
        self,
        output_names: Iterable[str],
        constraint_type: ArrayFunction.ConstraintType = ArrayFunction.ConstraintType.EQ,  # noqa: E501
        constraint_name: str = "",
        value: float = 0,
        positive: bool = False,
        apply_to_system_level: bool | None = None,
        apply_to_sub_level: bool | None = None,
    ) -> ArrayFunction | None:
        """
        Args:
            apply_to_system_level: Whether to add the constraint
                to the optimization problem of the main level.
                If `None`, use the `apply_constraints_to_system` option.
            apply_to_sub_level: Whether to add the constraint
                to the optimization problems of the sublevel.
                If `None`, use the `apply_constraints_to_sub_scenarios` option.
        """  # noqa: D205, D212
        if apply_to_system_level is None:
            apply_to_system_level = self._settings.apply_constraints_to_system

        if apply_to_sub_level is None:
            apply_to_sub_level = self._settings.apply_constraints_to_sub_scenarios

        if apply_to_system_level:
            system_level_constraint = self._create_system_level_constraint(
                output_names,
                constraint_type=constraint_type,
                constraint_name=constraint_name,
                value=value,
                positive=positive,
            )
        else:
            system_level_constraint = None

        if apply_to_sub_level:
            self._add_sub_level_constraint(
                output_names,
                constraint_type=constraint_type,
                constraint_name=constraint_name,
                value=value,
                positive=positive,
            )

        return system_level_constraint

    def _create_system_level_constraint(
        self,
        output_names: Iterable[str],
        constraint_type: ArrayFunction.ConstraintType,
        constraint_name: str,
        value: float,
        positive: bool,
    ) -> ArrayFunction:
        """Create a constraint function at the system level.

        Args:
            output_names: The names of the constrained outputs.
            constraint_type: The type of constraint.
            constraint_name: The name of the constraint to be stored,
                If empty, the name is generated from the output name.
            value: The value of activation of the constraint.
            positive: Whether the inequality constraint is positive.

        Returns:
            A constraint function at the system level.
        """
        return super().create_constraint(
            output_names,
            constraint_type=constraint_type,
            constraint_name=constraint_name,
            value=value,
            positive=positive,
        )

    def _add_sub_level_constraint(
        self,
        output_names: Iterable[str],
        constraint_type: ArrayFunction.ConstraintType,
        constraint_name: str,
        value: float,
        positive: bool,
    ) -> None:
        """Add a constraint at the sub-scenarios level.

        Args:
            output_names: The names of the constrained outputs.
            constraint_type: The type of constraint.
            constraint_name: The name of the constraint to be stored,
                If empty, the name is generated from the output name.
            value: The value of activation of the constraint.
            positive: Whether the inequality constraint is positive.

        Raises:
            ValueError: If a constraint is not found in the scenario
                top-level disciplines outputs.
        """
        added = False
        for sub_scenario in self.get_sub_scenarios():
            if self._scenario_computes_outputs(sub_scenario, output_names):
                sub_scenario.add_constraint(
                    *output_names,
                    constraint_type=constraint_type,
                    constraint_name=constraint_name,
                    value=value,
                    positive=positive,
                )
                added = True
        if not added:
            msg = (
                f"No sub scenario has the following output names: {output_names} "
                "cannot create such a constraint."
            )
            raise ValueError(msg)

    @staticmethod
    def _scenario_computes_outputs(
        scenario: MDOScenario,
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

    def _store_optimal_local_design_values(self, x_vect: DatabaseKeyType) -> None:
        """Store the optimal values of the local design variables in the database.

        Args:
            x_vect: The input value.
        """
        self.problem.database.store(
            x_vect,
            {
                k: v
                for adapter in self._scenario_adapters
                for k, v in adapter.scenario.design_space.get_current_value(
                    as_dict=True
                ).items()
            },
        )

        if self._settings.disciplines_as_sub_scenario:
            self.problem.database.store(
                x_vect,
                {
                    k: v
                    for discipline in self._settings.disciplines_as_sub_scenario
                    for k, v in discipline.output_data.items()
                },
            )
