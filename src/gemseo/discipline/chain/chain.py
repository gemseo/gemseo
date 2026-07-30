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
"""Chains of disciplines."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from networkx import DiGraph
from strenum import StrEnum

from gemseo.core._process_flow.base_process_flow import BaseProcessFlow
from gemseo.core.dependency_graph import DependencyGraph
from gemseo.core.derivative.derivation_mode import DerivationMode
from gemseo.core.derivative.graph_traversal import set_differentiated_ios
from gemseo.core.discipline import Discipline
from gemseo.core.discipline.process_discipline import ProcessDiscipline

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.core.derivative.graph_traversal import DisciplineIOs
    from gemseo.core.discipline.base_discipline import BaseDiscipline


class _ProcessFlow(BaseProcessFlow):
    """The process flow."""

    def get_data_flow(  # noqa: D102
        self,
    ) -> list[tuple[Discipline, Discipline, list[str]]]:
        disciplines = self.get_disciplines_in_data_flow()
        disciplines_couplings = self._get_disciplines_couplings(disciplines)

        # Add discipline inner couplings (ex. MDA case)
        for discipline in disciplines:
            disciplines_couplings.extend(discipline.get_process_flow().get_data_flow())

        return disciplines_couplings

    def _get_disciplines_couplings(
        self, disciplines: Sequence[Discipline]
    ) -> list[tuple[Discipline, Discipline, list[str]]]:
        """Return the couplings between the disciplines.

        Args:
            disciplines: The disciplines.

        Returns:
            The disciplines couplings, a coupling is
            composed of a discipline, one of its successor and the sorted
            variables names.
        """
        return DependencyGraph(disciplines).get_disciplines_couplings()


class ChainDerivationMode(StrEnum):
    """The derivation modes of a chain of disciplines.

    Each member aliases the corresponding
    [DerivationMode][gemseo.core.derivative.derivation_mode.DerivationMode] member so
    values stay in sync.
    """

    FORWARD = DerivationMode.FORWARD
    """The forward chain rule, accumulating from inputs to outputs."""

    REVERSE = DerivationMode.REVERSE
    """The reverse chain rule, accumulating from outputs to inputs."""

    AUTO = DerivationMode.AUTO
    """Automatic switch between the forward and reverse modes based on the data
    sizes."""


class DisciplineChain(ProcessDiscipline):
    """Chain of disciplines that is based on a predefined order of execution."""

    ChainDerivationMode = ChainDerivationMode
    """The enumeration of chain derivation modes."""

    _process_flow_class: ClassVar[type[BaseProcessFlow]] = _ProcessFlow

    __execution_graph: DiGraph
    """Forward-only dependency graph built from the execution order."""

    __last_differentiated_io_key: tuple[frozenset[str], frozenset[str]] | None
    """Cache key for the last differentiated I/O pair."""

    __discipline_to_ios: dict[Discipline, DisciplineIOs]
    """Mapping from each active discipline to its coupling I/Os for the sweep."""

    def __init__(self, disciplines: Sequence[BaseDiscipline], name: str = "") -> None:
        """
        Args:
            disciplines: The disciplines.
            name: The name of the discipline.
                If empty, use the name of the class.
        """  # noqa: D205, D212, D415
        super().__init__(disciplines, name=name)

        self.__discipline_to_ios = {}
        self.__execution_graph = DiGraph()
        self.__last_differentiated_io_key = None

        self._initialize_grammars()

    def _initialize_grammars(self) -> None:
        """Define the input and output grammars from the disciplines' ones."""
        for discipline in self._disciplines:
            self.io.input_grammar.update(
                discipline.io.input_grammar,
                excluded_names=self.io.output_grammar,
                allow_namespace_nesting=True,
            )
            self.io.output_grammar.update(
                discipline.io.output_grammar, allow_namespace_nesting=True
            )

    def __construct_execution_graph(self) -> None:
        """Construct the coupling graph based on chain execution order.

        Each consumer input is bound to its closest upstream producer: a later
        producer overwrites an earlier one, as in execution. Backward edges are
        excluded by construction.
        """
        self.__execution_graph.add_nodes_from(self._disciplines)
        outputs = {
            discipline: set(discipline.io.output_grammar)
            for discipline in self._disciplines
        }
        for i, consumer in enumerate(self._disciplines):
            unclaimed = set(consumer.io.input_grammar)
            for producer in reversed(self._disciplines[:i]):
                if shared := unclaimed & outputs[producer]:
                    self.__execution_graph.add_edge(producer, consumer, io=shared)
                    unclaimed -= shared
                    if not unclaimed:
                        break

    def _execute(self) -> None:
        out_data = self.io.output_data
        merged = self.io.get_merged_data()
        for discipline in self._disciplines:
            output = discipline.execute(merged)
            out_data |= output
            merged.update(output)

    def __select_linearization_mode(
        self,
        input_names: frozenset[str],
        output_names: frozenset[str],
    ) -> ChainDerivationMode:
        """Select reverse or forward mode based on total I/O variable sizes.

        Args:
            input_names: The chain-level inputs to differentiate with respect to.
            output_names: The chain-level outputs to differentiate.

        Returns:
            `ChainDerivationMode.REVERSE` if total output size ≤ total input size,
            `ChainDerivationMode.FORWARD` otherwise.
        """
        # Read the input and output stores directly: chain inputs live in the
        # input grammar, chain outputs in the output grammar. Guard with `in`:
        # data may be absent if AUTO is called before the first execute.
        input_data = self.io.input_data
        get_input_size = self.io.input_grammar.data_converter.get_value_size
        n_inputs = sum(
            get_input_size(n, input_data[n]) for n in input_names if n in input_data
        )

        output_data = self.io.output_data
        get_output_size = self.io.output_grammar.data_converter.get_value_size
        n_outputs = sum(
            get_output_size(n, output_data[n]) for n in output_names if n in output_data
        )

        return (
            ChainDerivationMode.REVERSE
            if n_outputs <= n_inputs
            else ChainDerivationMode.FORWARD
        )

    def __accumulate_reverse_chain_rule(self, output_names: frozenset[str]) -> None:
        """Accumulate the chain rule from outputs to inputs (adjoint sweep).

        Args:
            output_names: The chain-level outputs to differentiate.
        """
        self.jac = {}
        # __discipline_to_ios is ordered by execution; reversed here for adjoint sweep.
        for discipline, ios in reversed(self.__discipline_to_ios.items()):
            discipline.linearize(discipline.io.get_input_data(), execute=False)

            for accumulated_row in self.jac.values():
                # Expand couplings this discipline produces that appear in the row.
                # Such entries can only come from downstream disciplines, so they
                # are derivatives with respect to this discipline's outputs.
                for coupling in ios.outputs & accumulated_row.keys():
                    # Pop, not get: coupling is intermediate and must not survive into
                    # the final Jacobian; leaving it would corrupt the addition when
                    # input_name == coupling (self-coupled discipline, ∂c/∂c term).
                    d_out_d_coupling = accumulated_row.pop(coupling)
                    local_jacobians = discipline.jac[coupling]
                    for input_name in ios.inputs & local_jacobians.keys():
                        d_coupling_d_in = local_jacobians[input_name]
                        # Perform ∂o/∂i = ∂o/∂c · ∂c/∂i
                        term = d_out_d_coupling @ d_coupling_d_in
                        if input_name in accumulated_row:
                            # ∂o/∂i += prior contribution from another coupling path
                            term = term + accumulated_row[input_name]
                        accumulated_row[input_name] = term

            # Seed self.jac for chain-level outputs owned by this discipline.
            # Seeding must follow the expansion above: a seeded row is already
            # expressed in this discipline's inputs, and a self-coupled entry
            # (a variable both consumed and produced here) must not be expanded
            # by the local ∂c/∂c term.
            # ios.inputs guards stale entries from prior add_differentiated_inputs.
            for output_name in output_names - self.jac.keys():
                if output_name in discipline.jac:
                    local_jacobians = discipline.jac[output_name]
                    # Copy: self.jac must never alias a sub-discipline's blocks,
                    # so that mutating it cannot corrupt discipline.jac.
                    self.jac[output_name] = {
                        name: local_jacobians[name].copy()
                        for name in ios.inputs
                        if name in local_jacobians
                    }

    def __accumulate_forward_chain_rule(
        self, input_names: frozenset[str], output_names: frozenset[str]
    ) -> None:
        """Accumulate the chain rule from inputs to outputs (tangent sweep).

        Args:
            input_names: The chain-level inputs to differentiate with respect to.
            output_names: The chain-level outputs to differentiate.
        """
        # No stale-entry guard needed: accumulated_jacobians is rebuilt fresh each
        # call, so inputs_from_upstream only contains cleanly-accumulated entries.
        accumulated_jacobians = {}
        # __discipline_to_ios is ordered by execution.
        for discipline, ios in self.__discipline_to_ios.items():
            discipline.linearize(discipline.io.get_input_data(), execute=False)

            # Discipline inputs already produced upstream (tangent accumulated).
            inputs_from_upstream = ios.inputs & accumulated_jacobians.keys()

            # Chain-level inputs not yet reached by any upstream coupling path.
            direct_inputs = input_names - accumulated_jacobians.keys()

            # Discipline outputs consumed downstream or at chain level;
            # ios.outputs already includes the chain-level outputs owned here.
            relevant_outputs = ios.outputs & discipline.jac.keys()
            for discipline_output in relevant_outputs:
                local_jacobians = discipline.jac[discipline_output]
                # ∂o/∂i for direct chain inputs (no upstream coupling path).
                # Copy: self.jac must never alias a sub-discipline's blocks,
                # so that mutating it cannot corrupt discipline.jac.
                d_out = {
                    chain_input: local_jacobians[chain_input].copy()
                    for chain_input in direct_inputs
                    if chain_input in local_jacobians
                }

                # Intersect with local_jacobians: a discipline may omit the
                # blocks of an output that does not depend on a given coupling.
                for coupling in inputs_from_upstream & local_jacobians.keys():
                    d_out_d_coupling = local_jacobians[coupling]
                    accumulated_row = accumulated_jacobians[coupling]
                    for chain_input, d_coupling_d_in in accumulated_row.items():
                        # Perform ∂o/∂i = ∂o/∂c · ∂c/∂i
                        term = d_out_d_coupling @ d_coupling_d_in
                        if chain_input in d_out:
                            # ∂o/∂i += prior contribution from another upstream coupling
                            term = term + d_out[chain_input]
                        d_out[chain_input] = term

                # Unconditional: producing a variable overwrites any upstream
                # accumulation, even when the new row is empty (shadowing).
                accumulated_jacobians[discipline_output] = d_out

        self.jac = {
            output_name: accumulated_jacobians.get(output_name, {})
            for output_name in output_names
        }

    def _compute_jacobian(  # noqa: D102
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        if not self.__execution_graph:
            self.__construct_execution_graph()

        input_names = frozenset(input_names)
        output_names = frozenset(output_names)

        io_key = (input_names, output_names)
        if self.__last_differentiated_io_key != io_key:
            self.__discipline_to_ios = set_differentiated_ios(
                self.__execution_graph,
                input_names,
                output_names,
            )
            self.__last_differentiated_io_key = io_key

        mode = self.linearization_mode
        if mode == ChainDerivationMode.AUTO:
            mode = self.__select_linearization_mode(input_names, output_names)

        if mode == ChainDerivationMode.FORWARD:
            self.__accumulate_forward_chain_rule(input_names, output_names)
        else:
            self.__accumulate_reverse_chain_rule(output_names)

        self._init_jacobian(
            input_names,
            output_names,
            fill_missing_keys=True,
            init_type=Discipline.InitJacobianType.SPARSE,
        )
