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
"""Graph traversal algorithms for differentiated IO analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

from gemseo.core.dependency_graph import DependencyGraph

if TYPE_CHECKING:
    from collections.abc import Iterable

    from networkx import DiGraph

    from gemseo.core.discipline import Discipline

_IO = DependencyGraph.IO


class DisciplineIOs(NamedTuple):
    """Coupling inputs and outputs of a discipline within a set of disciplines."""

    inputs: list[str]
    """Upstream coupling inputs and own differentiation inputs."""

    outputs: list[str]
    """Downstream coupling outputs and own differentiation outputs."""


def _compute_reachable_nodes(
    graph: DiGraph,
    sources: Iterable[Discipline],
) -> set[Discipline]:
    """Compute the disciplines reachable from sources in a coupling graph.

    Args:
        graph: The directed coupling graph.
        sources: The source disciplines to explore from.

    Returns:
        The set of disciplines reachable from the source disciplines,
        sources included.
    """
    reached = set(sources)
    stack = list(sources)
    while stack:
        for successor in graph.successors(stack.pop()):
            if successor not in reached:
                reached.add(successor)
                stack.append(successor)
    return reached


def set_differentiated_ios(
    graph: DiGraph,
    input_names: Iterable[str],
    output_names: Iterable[str],
) -> dict[Discipline, DisciplineIOs]:
    """Identify and register the coupling IOs of each discipline on an active path.

    A discipline is active iff it lies on a directed path from a discipline
    owning at least one name in `input_names` to a discipline owning at
    least one name in `output_names`. For each active discipline, the
    coupling inputs (outputs of an upstream active discipline) and coupling
    outputs (inputs of a downstream active discipline) are registered via
    [add_differentiated_inputs()][gemseo.core.discipline.discipline.Discipline.add_differentiated_inputs]
    and
    [add_differentiated_outputs()][gemseo.core.discipline.discipline.Discipline.add_differentiated_outputs].

    Args:
        graph: The directed dependency graph of the disciplines.
        input_names: The names of the differentiation inputs.
        output_names: The names of the differentiation outputs.

    Returns:
        Mapping from each active discipline to its coupling inputs and outputs.
    """
    input_names = set(input_names)
    output_names = set(output_names)

    sources = {}
    destinations = {}
    for discipline in graph.nodes:
        if names := input_names.intersection(discipline.io.input_grammar):
            sources[discipline] = names
        if names := output_names.intersection(discipline.io.output_grammar):
            destinations[discipline] = names

    forward = _compute_reachable_nodes(graph, sources)
    backward = _compute_reachable_nodes(graph.reverse(copy=False), destinations)

    successors = graph.successors
    predecessors = graph.predecessors
    get_edge_data = graph.get_edge_data

    discipline_to_ios = {}
    for discipline in forward & backward:
        diff_inputs = set(sources.get(discipline, ()))
        diff_outputs = set(destinations.get(discipline, ()))

        for predecessor in predecessors(discipline):
            if predecessor in forward:
                diff_inputs.update(get_edge_data(predecessor, discipline)[_IO])

        for successor in successors(discipline):
            if successor in backward:
                diff_outputs.update(get_edge_data(discipline, successor)[_IO])

        discipline.add_differentiated_inputs(diff_inputs)
        discipline.add_differentiated_outputs(diff_outputs)

        discipline_to_ios[discipline] = DisciplineIOs(
            inputs=sorted(diff_inputs),
            outputs=sorted(diff_outputs),
        )

    return discipline_to_ios


def set_mda_differentiated_ios(
    graph: DiGraph,
    input_names: Iterable[str],
    output_names: Iterable[str],
) -> dict[Discipline, DisciplineIOs]:
    """Identify and register the coupling IOs of each active discipline in an MDA.

    Extends
    [set_differentiated_ios()][gemseo.core.derivatives.graph_traversal.set_differentiated_ios]
    with two MDA-specific additions for each active discipline:

    - Self-coupling variables (`input_grammar ∩ output_grammar`) are added to
      both coupling inputs and outputs, as the dependency graph omits self-loops.
    - State variables and residuals from `residual_to_state_variable` are
      registered so the Newton solver can assemble the coupled system.

    Args:
        graph: The directed dependency graph of the disciplines.
        input_names: The names of the differentiation inputs.
        output_names: The names of the differentiation outputs.

    Returns:
        Mapping from each active discipline to its coupling inputs and outputs.
    """
    discipline_to_ios = set_differentiated_ios(graph, input_names, output_names)

    for discipline, ios in discipline_to_ios.items():
        differentiated_inputs = set(ios.inputs)
        differentiated_outputs = set(ios.outputs)

        io = discipline.io
        if self_coupling := set(io.input_grammar).intersection(io.output_grammar):
            differentiated_inputs |= self_coupling
            differentiated_outputs |= self_coupling

        if residual_to_state := io.residual_to_state_variable:
            differentiated_inputs.update(residual_to_state.values())
            differentiated_outputs.update(residual_to_state)

        discipline.add_differentiated_inputs(differentiated_inputs)
        discipline.add_differentiated_outputs(differentiated_outputs)

        discipline_to_ios[discipline] = DisciplineIOs(
            inputs=sorted(differentiated_inputs),
            outputs=sorted(differentiated_outputs),
        )

    return discipline_to_ios
