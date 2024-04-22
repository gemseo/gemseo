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
#                         documentation
#        :author: Charlie Vanaret
"""Graphs of the disciplines dependencies and couplings."""

from __future__ import annotations

import logging
from typing import cast

from gemseo.core.discipline import MDODiscipline
from gemseo.utils.string_tools import pretty_str

try:
    # graphviz is an optional dependency.
    from gemseo.post._graph_view import GraphView
except ImportError:
    GRAPHVIZ_IS_MISSING = True
else:
    GRAPHVIZ_IS_MISSING = False

from typing import TYPE_CHECKING

from networkx import DiGraph
from networkx import condensation
from networkx import strongly_connected_components

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Sequence

LOGGER = logging.getLogger(__name__)

ExecutionSequence = list[list[tuple[MDODiscipline, ...]]]


class DependencyGraph:
    """Graph of dependencies between disciplines.

    This class can create the sequence of execution of the disciplines. This is done by
    determining the strongly connected components (scc) of the graph. The disciplines in
    the components of a scc have the same order as in the passed disciplines passed when
    instantiating the class.

    The couplings between the disciplines can also be computed.
    """

    __graph: DiGraph
    """The graph representing the disciplines data dependencies.

    The coupled inputs and outputs names are stored as an edge attributes named io.
    """

    def __init__(self, disciplines: Sequence[MDODiscipline]) -> None:
        """
        Args:
            disciplines: The disciplines to build the graph with.
        """  # noqa: D205, D212, D415
        self.__graph = self.__create_graph(disciplines)

    @property
    def disciplines(self) -> Iterator[MDODiscipline]:
        """The disciplines used to build the graph."""
        return iter(self.__graph.nodes)

    @property
    def graph(self) -> DiGraph:
        """The disciplines data graph."""
        return self.__graph

    def get_execution_sequence(self) -> list[list[tuple[MDODiscipline, ...]]]:
        """Compute the execution sequence of the disciplines.

        Returns:
            The execution sequence.
        """
        condensed_graph = self.__create_condensed_graph()
        execution_sequence = []

        while True:
            leaves = self.__get_leaves(condensed_graph)

            if not leaves:
                break

            parallel_tasks = [
                tuple(
                    cast(
                        list[MDODiscipline],
                        condensed_graph.nodes[node_id]["members"],
                    )
                )
                for node_id in leaves
            ]
            execution_sequence += [parallel_tasks]
            condensed_graph.remove_nodes_from(leaves)

        return list(reversed(execution_sequence))

    def __create_condensed_graph(self) -> DiGraph:
        """Return the condensed graph."""
        # scc = nx.kosaraju_strongly_connected_components(self.__graph)
        # scc = nx.strongly_connected_components_recursive(self.__graph)
        # fastest routine
        return condensation(
            self.__graph,
            scc=self.__get_ordered_scc(strongly_connected_components(self.__graph)),
        )

    def __get_ordered_scc(
        self, scc: Iterator[set[MDODiscipline]]
    ) -> Iterator[list[MDODiscipline]]:
        """Return the scc nodes ordered by the initial disciplines.

        Args:
            scc: The scc nodes.

        Yields:
            The ordered scc nodes.
        """
        disciplines = list(self.__graph.nodes)
        for components in scc:
            disc_indexes = {}

            for component in components:
                index = disciplines.index(component)
                disc_indexes[index] = component

            ordered_components = []

            for index in sorted(disc_indexes.keys()):
                ordered_components += [disc_indexes[index]]

            yield ordered_components

    def get_disciplines_couplings(
        self,
    ) -> list[tuple[MDODiscipline, MDODiscipline, list[str]]]:
        """Return the couplings between the disciplines.

        Returns:
            The disciplines couplings, a coupling is
            composed of a discipline, one of its successor and the sorted
            variables names.
        """
        couplings = []
        for from_disc, to_disc, edge_names in self.__graph.edges(data="io"):
            couplings += [(from_disc, to_disc, sorted(edge_names))]
        return couplings

    @staticmethod
    def __create_graph(disciplines: Iterable[MDODiscipline]) -> DiGraph:
        """Create the full graph.

        The coupled inputs and outputs names are stored as an edge attributes named io.

        Args:
            disciplines: The disciplines to build the graph with.

        Returns:
            The graph of disciplines.
        """
        nodes_to_ios = {}

        for disc in disciplines:
            nodes_to_ios[disc] = (
                set(disc.get_input_data_names()),
                set(disc.get_output_data_names()),
            )

        graph = DiGraph()
        graph.add_nodes_from(disciplines)
        graph_add_edge = graph.add_edge
        # Create the graph edges by discovering the coupled disciplines
        for disc_i, (_, outputs_i) in nodes_to_ios.items():
            for disc_j, (inputs_j, _) in nodes_to_ios.items():
                if disc_i != disc_j:
                    coupled_io = outputs_i & inputs_j
                    if coupled_io:
                        graph_add_edge(disc_i, disc_j, io=coupled_io)

        return graph

    @staticmethod
    def __get_node_name(graph: DiGraph, node: int | MDODiscipline) -> str:
        """Return the name of a node for the representation of a graph.

        Args:
            graph: A full or condensed graph.
            node: A node of the graph.

        Returns:
            The name of the node.
        """
        if isinstance(node, MDODiscipline):
            # not a scc node
            return str(node)

        # networkx stores the scc nodes under the members node attribute
        condensed_discs = graph.nodes[node]["members"]

        if len(condensed_discs) == 1:
            # not a scc node in a scc graph
            return str(next(iter(condensed_discs)))

        # scc node
        return "MDA of {}".format(", ".join(map(str, condensed_discs)))

    @staticmethod
    def __get_scc_edge_names(
        graph: DiGraph,
        node_from: int,
        node_to: int | None = None,
    ) -> set[str]:
        """Return the names of an edge in a condensed graph.

        Args:
            graph: A condensed graph.
            node_from: The predecessor node in the graph.
            node_to: The successor node in the graph.

        Returns:
            The names of the edge.
        """
        output_names = set()
        for disc in graph.nodes[node_from]["members"]:
            output_names.update(disc.get_output_data_names())

        if node_to is None:
            return output_names

        input_names = set()
        for disc in graph.nodes[node_to]["members"]:
            input_names.update(disc.get_input_data_names())
        return output_names & input_names

    def __write_graph(
        self,
        graph: DiGraph,
        file_path: str,
        is_full: bool,
    ) -> GraphView | None:
        """Write the representation of a graph.

        Args:
            graph: A graph.
            file_path: The file path to save the visualization.
            is_full: Whether the graph is full.
        """
        if GRAPHVIZ_IS_MISSING:
            LOGGER.warning(
                "Cannot write graph: "
                "GraphView cannot be imported because graphviz is not installed."
            )
            return None

        graph_view = GraphView()

        # 1. Add the edges with different head and tail nodes
        #    (case: some outputs of a discipline are inputs of another one)
        for head_node, tail_node, coupling_names in graph.edges(data="io"):
            head_name = self.__get_node_name(graph, head_node)
            tail_name = self.__get_node_name(graph, tail_node)
            if not isinstance(tail_node, MDODiscipline):
                # a scc edge
                coupling_names = self.__get_scc_edge_names(graph, head_node, tail_node)

            graph_view.edge(head_name, tail_name, pretty_str(coupling_names, ","))

        # 2. Add the edges with same head and tail nodes
        #    (case: some outputs of a discipline are inputs of itself)
        if is_full:
            for discipline in self.__graph.nodes:
                coupling_names = set(discipline.get_input_data_names()).intersection(
                    discipline.get_output_data_names()
                )
                if coupling_names:
                    name = discipline.name
                    graph_view.edge(name, name, pretty_str(coupling_names, ","))

        # 3. Add the edges without head node
        #    (case: some output variables of discipline are not coupling variables).
        for leaf_node in self.__get_leaves(graph):
            if isinstance(leaf_node, MDODiscipline):
                output_names = leaf_node.get_output_data_names()
                node_name = str(leaf_node)
            else:
                # a scc edge
                output_names = list(self.__get_scc_edge_names(graph, leaf_node))
                node_name = self.__get_node_name(graph, leaf_node)

            if not output_names:
                continue

            tail_name = f"_{leaf_node}"
            graph_view.edge(node_name, tail_name, pretty_str(output_names, ","))
            graph_view.hide_node(tail_name)

        # 4. Write the dot and target files.
        graph_view.visualize(show=False, file_path=file_path, clean_up=False)
        return graph_view

    def write_full_graph(self, file_path: str) -> GraphView | None:
        """Write a representation of the full graph.

        Args:
            file_path: A path to the file.

        Returns:
            The full graph or ``None`` otherwise.
        """
        return self.__write_graph(self.__graph, file_path, True)

    # TODO: API: remove.
    export_initial_graph = write_full_graph

    def write_condensed_graph(self, file_path: str) -> GraphView | None:
        """Write a representation of the condensed graph.

        Args:
            file_path: A path to the file.

        Returns:
            The condensed graph or ``None`` otherwise.
        """
        return self.__write_graph(self.__create_condensed_graph(), file_path, False)

    # TODO: API: remove.
    export_reduced_graph = write_condensed_graph

    @staticmethod
    def __get_leaves(graph: DiGraph) -> list[MDODiscipline] | list[int]:
        """Return the leaf nodes of a graph.

        Args:
            graph: A graph.

        Returns:
            The graph leaf nodes.
        """
        return [n for n in graph.nodes if graph.out_degree(n) == 0]
