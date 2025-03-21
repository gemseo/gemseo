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
from typing import Final
from typing import cast

from gemseo.core.discipline import Discipline
from gemseo.utils.discipline import DisciplineVariableProperties
from gemseo.utils.discipline import get_discipline_variable_properties
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
    from collections.abc import Mapping
    from collections.abc import Sequence
    from pathlib import Path

LOGGER = logging.getLogger(__name__)

ExecutionSequence = list[list[tuple[Discipline, ...]]]


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

    IO: Final[str] = "io"
    """The argument name for the coupling variables associated with an edge."""

    def __init__(self, disciplines: Sequence[Discipline]) -> None:
        """
        Args:
            disciplines: The disciplines to build the graph with.
        """  # noqa: D205, D212, D415
        self.__graph = self.__create_graph(disciplines)
        if len({d.name for d in disciplines}) == len(disciplines):
            self.__get_node_name_from_discipline = self._get_node_name_from_disc_name
        else:
            self.__get_node_name_from_discipline = self._get_node_name_from_disc_id

    @staticmethod
    def _get_node_name_from_disc_name(discipline: Discipline) -> str:
        """Return the name of a node from the name of the associated discipline.

        Args:
            discipline: The discipline.

        Returns:
            The name of the node.
        """
        return discipline.name

    @staticmethod
    def _get_node_name_from_disc_id(discipline: Discipline) -> str:
        """Return the name of a node from the id of the associated discipline.

        Args:
            discipline: The discipline.

        Returns:
            The name of the node.
        """
        return str(id(discipline))

    @property
    def disciplines(self) -> Iterator[Discipline]:
        """The disciplines used to build the graph."""
        return iter(self.__graph.nodes)

    @property
    def graph(self) -> DiGraph:
        """The disciplines data graph."""
        return self.__graph

    def get_execution_sequence(self) -> list[list[tuple[Discipline, ...]]]:
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
                        "list[Discipline]",
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
        self, scc: Iterator[set[Discipline]]
    ) -> Iterator[list[Discipline]]:
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
    ) -> list[tuple[Discipline, Discipline, list[str]]]:
        """Return the couplings between the disciplines.

        Returns:
            The disciplines couplings, a coupling is
            composed of a discipline, one of its successor and the sorted
            variables names.
        """
        couplings = []
        for from_disc, to_disc, edge_names in self.__graph.edges(data=self.IO):
            couplings += [(from_disc, to_disc, sorted(edge_names))]
        return couplings

    @staticmethod
    def __create_graph(disciplines: Iterable[Discipline]) -> DiGraph:
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
                set(disc.io.input_grammar),
                set(disc.io.output_grammar),
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

    def __get_node_name(self, graph: DiGraph, node: int | Discipline) -> str:
        """Return the name of a node for the representation of a graph.

        Args:
            graph: A full or condensed graph.
            node: A node of the graph.

        Returns:
            The name of the node.
        """
        if isinstance(node, Discipline):
            # not a scc node
            return self.__get_node_name_from_discipline(node)

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
            output_names.update(disc.io.output_grammar)

        if node_to is None:
            return output_names

        input_names = set()
        for disc in graph.nodes[node_to]["members"]:
            input_names.update(disc.io.input_grammar)
        return output_names & input_names

    def __render_graph(
        self,
        graph: DiGraph,
        file_path: str | Path,
        is_full: bool,
    ) -> GraphView | None:
        """Render the graph when graphviz is installed.

        Args:
            graph: A graph.
            file_path: The file path to save the graphical representation of the graph.
                If empty, the graphical representation is not saved.
            is_full: Whether the graph is full.

        Returns:
            Either the graph or ``None`` when graphviz is not installed.
        """
        if GRAPHVIZ_IS_MISSING:
            LOGGER.warning(
                "Cannot render graph: "
                "GraphView cannot be imported because graphviz is not installed."
            )
            return None

        graph_view = GraphView()
        get_node_name_from_discipline = self.__get_node_name_from_discipline
        add_tooltip = (
            # This is a full graph.
            is_full
            # There are no homonymous disciplines.
            and get_node_name_from_discipline != self._get_node_name_from_disc_id
        )
        if add_tooltip:
            get_properties = get_discipline_variable_properties
            discipline_names_to_properties = {
                discipline.name: get_properties(discipline)
                for discipline in self.__graph.nodes
            }
        else:
            discipline_names_to_properties = {}

        if get_node_name_from_discipline == self._get_node_name_from_disc_id:
            for discipline in graph.nodes:
                graph_view.node(
                    get_node_name_from_discipline(discipline), discipline.name
                )

        # 1. Add the edges with different head and tail nodes
        #    (case: some outputs of a discipline are inputs of another one)
        for tail_node, head_node, coupling_names in graph.edges(data=self.IO):
            tail_name = self.__get_node_name(graph, tail_node)
            head_name = self.__get_node_name(graph, head_node)
            if not isinstance(head_node, Discipline):
                # a scc edge
                coupling_names = self.__get_scc_edge_names(graph, tail_node, head_node)

            self.__add_edge(
                graph_view,
                coupling_names,
                discipline_names_to_properties,
                tail_name,
                head_name,
                add_tooltip,
            )

        # 2. Add the edges with same head and tail nodes
        #    (case: some outputs of a discipline are inputs of itself)
        if is_full:
            for discipline in self.__graph.nodes:
                coupling_names = set(discipline.io.input_grammar).intersection(
                    discipline.io.output_grammar
                )
                if coupling_names:
                    name = get_node_name_from_discipline(discipline)
                    self.__add_edge(
                        graph_view,
                        coupling_names,
                        discipline_names_to_properties,
                        name,
                        name,
                        add_tooltip,
                    )

        # 3. Add the edges without head node
        #    (case: some output variables of discipline are not coupling variables).
        for leaf_node in self.__get_leaves(graph):
            if isinstance(leaf_node, Discipline):
                output_names = tuple(leaf_node.io.output_grammar)
                node_name = get_node_name_from_discipline(leaf_node)
            else:
                # a scc edge
                output_names = tuple(self.__get_scc_edge_names(graph, leaf_node))
                node_name = self.__get_node_name(graph, leaf_node)

            if not output_names:
                continue

            self.__add_edge(
                graph_view,
                output_names,
                discipline_names_to_properties,
                node_name,
                f"_{leaf_node}",
                add_tooltip,
                hide_head=True,
            )

        if file_path:
            # 4. Write the dot and target files.
            graph_view.visualize(show=False, file_path=file_path, clean_up=False)

        return graph_view

    @staticmethod
    def __add_edge(
        graph_view: GraphView,
        coupling_names: Iterable[str],
        discipline_names_to_properties: Mapping[
            str,
            tuple[
                Mapping[str, DisciplineVariableProperties],
                Mapping[str, DisciplineVariableProperties],
            ],
        ],
        tail_name: str,
        head_name: str,
        add_tooltip: bool,
        hide_head: bool = False,
    ) -> None:
        """Add an edge to a graph view.

        Args:
            graph_view: The graph view.
            coupling_names: The names of the coupling variables.
            discipline_names_to_properties: The variables properties
                associated with the discipline names.
            tail_name: The name of the tail discipline.
            head_name: The name of the head discipline.
            add_tooltip: Whether to display
                the original and current names of the coupling variables
                when hovered over.
            hide_head: Whether to hide the head node.
        """
        if add_tooltip:
            lines = []
            sep = ", "
            if hide_head:
                lines.append(f"Global name{sep}Name in discipline {tail_name!r}\n")
            else:
                lines.append(
                    f"Global name{sep}"
                    f"Name in discipline {tail_name!r}{sep}"
                    f"Name in discipline {head_name!r}\n"
                )

            tail_properties = discipline_names_to_properties[tail_name][1]
            if hide_head:
                for coupling_name in coupling_names:
                    tail_original_name = tail_properties[coupling_name].original_name
                    lines.append(f"{coupling_name}{sep}{tail_original_name}")
            else:
                head_properties = discipline_names_to_properties[head_name][0]
                for coupling_name in coupling_names:
                    head_original_name = head_properties[coupling_name].original_name
                    tail_original_name = tail_properties[coupling_name].original_name
                    lines.append(
                        f"{coupling_name}{sep}"
                        f"{tail_original_name}{sep}"
                        f"{head_original_name}"
                    )

            kwargs = {"labeltooltip": "\n".join(lines)}
        else:
            kwargs = {}

        graph_view.edge(tail_name, head_name, pretty_str(coupling_names), **kwargs)
        if hide_head:
            graph_view.hide_node(head_name)

    def render_full_graph(self, file_path: str | Path) -> GraphView | None:
        """Render the full graph.

        Args:
            file_path: The file path
                to save the graphical representation of the full graph.
                If empty, the graphical representation is not saved.

        Returns:
            Either the full graph or ``None`` when graphviz is not installed.
        """
        return self.__render_graph(self.__graph, file_path, True)

    def render_condensed_graph(self, file_path: str | Path) -> GraphView | None:
        """Render the condensed graph.

        Args:
            file_path: The file path
                to save the graphical representation of the condensed graph.
                If empty, the graphical representation is not saved.

        Returns:
            Either the condensed graph or ``None`` when graphviz is not installed.
        """
        return self.__render_graph(self.__create_condensed_graph(), file_path, False)

    @staticmethod
    def __get_leaves(graph: DiGraph) -> list[Discipline] | list[int]:
        """Return the leaf nodes of a graph.

        Args:
            graph: A graph.

        Returns:
            The graph leaf nodes.
        """
        return [n for n in graph.nodes if graph.out_degree(n) == 0]
