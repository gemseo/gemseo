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
"""The chain rule."""
from __future__ import annotations

from typing import Iterable
from typing import List
from typing import Mapping
from typing import Tuple
from typing import TYPE_CHECKING

from networkx import edge_bfs
from networkx import Graph
from networkx import reverse_view

if TYPE_CHECKING:
    from gemseo.core.discipline import MDODiscipline

    DisciplineIOMapping = Mapping[MDODiscipline, Tuple[List[str], List[str]]]


def _initialize_add_diff_io(
    graph: Graph, inputs: Iterable[str], outputs: Iterable[str]
) -> tuple[list[MDODiscipline], list[MDODiscipline], DisciplineIOMapping]:
    """Initialize the graph traversal algorithm.

    Detects the disciplines that have inputs and outputs to be differentiated
    because they have an input/output in the list of data to be differentiated.

    Args:
        graph: The data graph of the process, built from
            :class:`.DependencyGraph`.
        inputs: The inputs with respect to which the chain is differentiated.
        outputs: The chain outputs to be differentiated.

    Returns:
        The disciplines that have inputs with respect to which the
        differentiation is performed.
        The disciplines containing outputs to be differentiated.
        The input and output data names to differentiate.
    """
    input_sources = []
    output_sources = []
    diff_ios = {}
    for disc in graph.nodes:
        input_grammar = disc.input_grammar
        output_grammar = disc.output_grammar

        common_data = set(inputs).intersection(input_grammar.names)
        if common_data:
            diff_ios[disc] = (common_data, [])
            input_sources.append(disc)

        common_data = set(outputs).intersection(output_grammar.names)
        if common_data:
            output_sources.append(disc)
            diff_io_init_disc = diff_ios.get(disc)
            if diff_io_init_disc is None:
                diff_io_init_disc = ([], common_data)
                diff_ios[disc] = diff_io_init_disc
            else:
                diff_io_init_disc[1].extend(common_data)

    return input_sources, output_sources, diff_ios


def _bfs_one_way_diff_io(
    graph: Graph, source_disciplines: list[MDODiscipline], reverse: bool = False
) -> DisciplineIOMapping:
    """Traverse the graph using a BFS algorithm to set the differentiated IOs.

    Detects the disciplines that depend on outputs of the source disciplines,
    eventually indirectly through outputs of other disciplines.

    Args:
        graph: The data graph of the process, built from
            :class:`.DependencyGraph`.
        source_disciplines: The disciplines that have inputs with respect
            to which the differentiation is performed.
        reverse: Whether to reverse the graph direction and traverse it in reverse.

    Returns:
        The disciplines containing the outputs to be differentiated.
        The mapping of input and output data names to differentiate.
    """
    if reverse:
        graph = reverse_view(graph)
        inputs_source_edge_index = 1
        outputs_dest_edge_index = 0
    else:
        inputs_source_edge_index = 0
        outputs_dest_edge_index = 1

    diff_io = {}
    for source_disc in source_disciplines:
        for edge in edge_bfs(graph, source=source_disc):
            coupl_io = graph.get_edge_data(*edge)["io"]
            # The origin of the edge is the discipline that computes the outputs.
            # These outputs are added to the outputs to be differentiated.
            disc_1 = edge[0]
            disc_1_couplings = diff_io.get(disc_1)
            if disc_1_couplings is None:
                disc_1_couplings = ([], [])
                diff_io[disc_1] = disc_1_couplings
            disc_1_couplings[outputs_dest_edge_index].extend(coupl_io)

            # The destination of the edge is the discipline that takes the couplings as
            # inputs, these inputs are added to the  inputs to be differentiated.
            disc_2 = edge[1]
            disc_2_couplings = diff_io.get(disc_2)
            if disc_2_couplings is None:
                disc_2_couplings = ([], [])
                diff_io[disc_2] = disc_2_couplings
            disc_2_couplings[inputs_source_edge_index].extend(coupl_io)

    return diff_io


def _merge_diff_ios(
    diff_io_direct: DisciplineIOMapping,
    diff_io_reverse: DisciplineIOMapping,
    diff_io_init: DisciplineIOMapping,
) -> DisciplineIOMapping:
    """Merge the differentiated inputs and outputs mappings.

    These have been generated by the direct and
    reverse tree traversal, and the initial one generated by the sources (inputs and
    outputs to be differentiated of the MDA).

    Args:
        diff_io_direct: The differentiated inputs & outputs generated by the direct BFS.
        diff_io_reverse: The differentiated inputs & outputs generated by the
            reverse BFS.
        diff_io_init: The initial differentiated inputs & outputs.

    Returns:
        The merged differentiated inputs and outputs.
    """
    # Now set the final diff ios
    if len(diff_io_reverse) < len(diff_io_direct):
        diff_io_1 = diff_io_reverse
        diff_io_2 = diff_io_direct
    else:
        diff_io_1 = diff_io_direct
        diff_io_2 = diff_io_reverse

    diff_ios_merged = {}

    for disc, in_out_1 in diff_io_1.items():
        in_out_2 = diff_io_2.get(disc)
        if in_out_2 is not None:
            diff_ios_merged[disc] = [[], []]
            # If the inputs are presents in both the direct and reverse graph
            # traversal, add them to the differentiated ios.
            diff_inputs = set(in_out_1[0]).intersection(in_out_2[0])
            diff_ios_merged[disc][0].extend(diff_inputs)

            # Do the same for outputs.
            diff_outputs = set(in_out_1[1]).intersection(in_out_2[1])
            diff_ios_merged[disc][1].extend(diff_outputs)

            # Now treat the special case of the initial step of the algorithm, by
            # the inputs and outputs with respect to the derivation of the chain
            # is made.
            disc_ios_init = diff_io_init.get(disc)

            if disc_ios_init is not None:
                # The inputs ared added to the differentiated inputs of the discipline
                # only if the discipline has differentiated outputs.

                if diff_outputs and disc_ios_init[0]:
                    diff_inputs_of_disc = disc_ios_init[0]
                    diff_ios_merged[disc][0].extend(diff_inputs_of_disc)

                # Vice and versa.
                if diff_inputs and disc_ios_init[1]:
                    diff_outputs_of_disc = disc_ios_init[1]
                    diff_ios_merged[disc][1].extend(diff_outputs_of_disc)
    return diff_ios_merged


def _merge_diff_io_special(
    source_input_disc: Iterable[MDODiscipline],
    source_output_disc: Iterable[MDODiscipline],
    init_diff_ios: DisciplineIOMapping,
    diff_ios_merged: DisciplineIOMapping,
) -> None:
    """Merge the differentiated inputs and outputs.

    To treat the special case of the
    isolated disciplines that are not connected to others in the graph.
    This modifies inplace the diff_ios_merged.

    Args:
        source_input_disc: The disciplines that have inputs with respect to which the
            differentiation is performed.
        source_output_disc: The disciplines containing outputs to be differentiated.
            The input and output data names to differentiate.
        init_diff_ios: The initial differentiated inputs & outputs.
        diff_ios_merged: The merged differentiated inputs and outputs.
    """
    for disc_source in set(source_input_disc).intersection(source_output_disc):
        diff_inputs_of_disc = init_diff_ios[disc_source][0]
        diff_outputs_of_disc = init_diff_ios[disc_source][1]
        if disc_source in diff_ios_merged:
            diff_ios_merged[disc_source][0].extend(diff_inputs_of_disc)
            diff_ios_merged[disc_source][1].extend(diff_outputs_of_disc)
        else:
            diff_ios_merged[disc_source] = (diff_inputs_of_disc, diff_outputs_of_disc)


def apply_diff_ios(diff_ios: DisciplineIOMapping) -> None:
    """Apply the differentiated inputs and outputs.

    From the diff_ios specfication, add the differentiated inputs and outputs to the
    disciplines.

    Args:
        diff_ios: The differentiated inputs and outputs.
    """
    for disc, (diff_in, diff_out) in diff_ios.items():
        if diff_in:
            disc.add_differentiated_inputs(diff_in)
        if diff_out:
            disc.add_differentiated_outputs(diff_out)


def traverse_add_diff_io(
    graph: Graph,
    inputs: Iterable[str],
    outputs: Iterable[str],
    add_differentiated_ios: bool = True,
) -> None | dict[MDODiscipline, list[set, set]]:
    """Set the required differentiated IOs for the disciplines in a chain.

    Add the differentiated inputs and outputs to the disciplines in a chain of
    executions, given the list of inputs with respect to which the derivation is made
    and the list of outputs to be derived. The inputs and outputs are a subset of all
    the inputs and outputs of the chain. This allows to minimize the number of
    linearizations to perform in the disciplines.

    Uses a two ways (from inputs to outputs and then from outputs to inputs)
     breadth first search graph traversal algorithm.
    The graph is constructed by :class:`.DependencyGraph`, which represents the data
    connexions (edges) between the disciplines (nodes).

    Args:
        graph: The data graph of the process, built from
            :class:`.DependencyGraph`.
        inputs: The inputs with respect to which the chain chain is differentiated.
        outputs: The chain outputs to be differentiated.
        add_differentiated_ios: Whether to add the differentiated inputs and outputs
            to the discipline by calling `~.MDODiscipline.add_differentiated_inputs`
            and `~.MDODiscipline.add_differentiated_outputs`

    Returns:
        The merged differentiated inputs and outputs.
    """
    source_input_disc, source_output_disc, init_diff_ios = _initialize_add_diff_io(
        graph, inputs, outputs
    )

    # Traverse the graph from the inputs to the bottom.
    diff_io_direct = _bfs_one_way_diff_io(graph, source_input_disc, reverse=False)

    # Now the graph is traversed from the outputs to the inputs.
    # The graph edges are reversed.
    diff_io_reverse = _bfs_one_way_diff_io(graph, source_output_disc, reverse=True)

    diff_ios_merged = _merge_diff_ios(diff_io_direct, diff_io_reverse, init_diff_ios)

    # We treat the special case of isolated disciplines that are not connected to others
    # in the graph.
    _merge_diff_io_special(
        source_input_disc, source_output_disc, init_diff_ios, diff_ios_merged
    )

    if add_differentiated_ios:
        apply_diff_ios(diff_ios_merged)

    return diff_ios_merged
