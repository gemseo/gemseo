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
from typing import Iterable
from typing import Iterator

from gemseo.core.discipline import MDODiscipline
from gemseo.utils.string_tools import pretty_str

# graphviz is an optional dependency

try:
    from gemseo.post._graph_view import GraphView
except ImportError:
    GraphView = None

from networkx import Graph
from networkx import DiGraph
from networkx import strongly_connected_components
from networkx import condensation


LOGGER = logging.getLogger(__name__)


class DependencyGraph:
    """Graph of dependencies between disciplines.

    This class can create the sequence of execution of the disciplines.
    This is done by determining the strongly connected components (scc) of the graph.
    The disciplines in the components of a scc have the same order
    as in the passed disciplines passed when instantiating the class.

    The couplings between the disciplines can also be computed.
    """

    def __init__(self, disciplines) -> None:
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
    def graph(self) -> Graph:
        """The disciplines data graph."""
        return self.__graph

    def get_execution_sequence(self):
        """Compute the execution sequence of the disciplines.

        Returns:
            list(set(tuple(set(MDODisciplines))))
        """
        condensed_graph = self.__create_condensed_graph()
        execution_sequence = []

        while True:
            leaves = self.__get_leaves(condensed_graph)

            if not leaves:
                break

            parallel_tasks = list(
                tuple(condensed_graph.nodes[node_id]["members"]) for node_id in leaves
            )
            execution_sequence += [parallel_tasks]
            condensed_graph.remove_nodes_from(leaves)

        return list(reversed(execution_sequence))

    def __create_condensed_graph(self):
        """Return the condensed graph."""
        # scc = nx.kosaraju_strongly_connected_components(self.__graph)
        # scc = nx.strongly_connected_components_recursive(self.__graph)
        # fastest routine
        return condensation(
            self.__graph,
            scc=self.__get_ordered_scc(strongly_connected_components(self.__graph)),
        )

    def __get_ordered_scc(self, scc):
        """Return the scc nodes ordered by the initial disciplines.

        Args:
            scc: The scc nodes.

        Yields:
            List[MDODisciplines]: The ordered scc nodes.
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

    def get_disciplines_couplings(self):
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
            disciplines (list): The disciplines to build the graph with.

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
    def __get_node_name(graph, node):
        """Return the name of a node for the representation of a graph.

        Args:
            graph (networkx.DiGraph): A full or condensed graph.
            node (int, MDODiscipline): A node of the graph.

        Returns:
            str: The name of the node.
        """
        if isinstance(node, MDODiscipline):
            # not a scc node
            return str(node)

        # networkx stores the scc nodes under the members node attribute
        condensed_discs = graph.nodes[node]["members"]

        if len(condensed_discs) == 1:
            # not a scc node in a scc graph
            return str(list(condensed_discs)[0])

        # scc node
        return "MDA of {}".format(", ".join(map(str, condensed_discs)))

    @staticmethod
    def __get_scc_edge_names(graph, node_from, node_to=None):
        """Return the names of an edge in a condensed graph.

        Args:
            graph (networkx.DiGraph): A condensed graph.
            node_from (int): The predecessor node in the graph.
            node_to (int, optional): The successor node in the graph.

        Returns:
            set(str): The names of the edge.
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

    def __write_graph(self, graph: DiGraph, file_path: str, is_full: bool) -> None:
        """Write the representation of a graph.

        Args:
            graph: A graph.
            file_path: The file path to save the visualization.
            is_full: Whether the graph is full.
        """
        if GraphView is None:
            LOGGER.warning(
                "Cannot write graph: "
                "GraphView cannot be imported because graphviz is not installed."
            )
            return

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
        for head_name in self.__get_leaves(graph):
            if isinstance(head_name, MDODiscipline):
                output_names = head_name.get_output_data_names()
                node_name = str(head_name)
            else:
                # a scc edge
                output_names = self.__get_scc_edge_names(graph, head_name)
                node_name = self.__get_node_name(graph, head_name)

            if not output_names:
                continue

            tail_name = f"_{head_name}"
            graph_view.edge(node_name, tail_name, pretty_str(output_names, ","))
            graph_view.hide_node(tail_name)

        # 4. Write the dot and target files.
        graph_view.visualize(show=False, file_path=file_path, clean_up=False)

    def write_full_graph(self, file_path):
        """Write a representation of the full graph.

        Args:
            file_path (str): A path to the file.
        """
        self.__write_graph(self.__graph, file_path, True)

    export_initial_graph = write_full_graph

    def write_condensed_graph(self, file_path):
        """Write a representation of the condensed graph.

        Args:
            file_path (str): A path to the file.
        """
        self.__write_graph(self.__create_condensed_graph(), file_path, False)

    export_reduced_graph = write_condensed_graph

    @staticmethod
    def __get_leaves(graph):
        """Return the leaf nodes of a graph.

        Args:
            graph (networkx.DiGraph): A graph.
        """
        return [n for n in graph.nodes if graph.out_degree(n) == 0]
