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
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Final
from typing import cast

from networkx import DiGraph
from networkx import condensation
from networkx import strongly_connected_components

from gemseo.core.discipline.base_discipline import BaseDiscipline

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Sequence

    from graphviz import Digraph

    from gemseo.util.typing import StrPath

# graphviz is an optional dependency;
# the gemseo.post.graph import (which imports graphviz) happens
# lazily where the graph is drawn so that gemseo.core does not depend on gemseo.post
# at module level.
graphviz_is_missing: Final[bool] = find_spec("graphviz") is None

logger = logging.getLogger(__name__)

ExecutionSequence = list[list[tuple[BaseDiscipline, ...]]]


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

    io: Final[str] = "io"
    """The argument name for the coupling variables associated with an edge."""

    def __init__(self, disciplines: Sequence[BaseDiscipline]) -> None:
        """
        Args:
            disciplines: The disciplines to build the graph with.
        """  # noqa: D205, D212, D415
        self.__graph = self.__create_graph(disciplines)

    @property
    def disciplines(self) -> Iterator[BaseDiscipline]:
        """The disciplines used to build the graph."""
        return iter(self.__graph.nodes)

    @property
    def graph(self) -> DiGraph:
        """The disciplines data graph."""
        return self.__graph

    def get_execution_sequence(self) -> ExecutionSequence:
        """Compute the execution sequence of the disciplines.

        Returns:
            The execution sequence.
        """
        condensed_graph = self.create_condensed_graph()
        execution_sequence = []

        while True:
            leaves = self.__get_leaves(condensed_graph)

            if not leaves:
                break

            parallel_tasks = [
                tuple(
                    cast(
                        "list[BaseDiscipline]",
                        condensed_graph.nodes[node_id]["members"],
                    )
                )
                for node_id in leaves
            ]
            execution_sequence.append(parallel_tasks)
            condensed_graph.remove_nodes_from(leaves)

        return list(reversed(execution_sequence))

    def create_condensed_graph(self) -> DiGraph:
        """Create the condensed graph.

        The condensed graph merges each strongly connected component
        of the graph of the disciplines into a single node.

        Returns:
            The condensed graph,
            whose nodes are the indices of the strongly connected components,
            from 0,
            and whose `members` node attribute is the list of the disciplines
            of the component,
            in the order of the disciplines passed at instantiation.
            The `mapping` graph attribute maps each discipline
            to the index of its component.
            The `io` edge attribute is the set of the names of the coupling variables
            from the disciplines of the tail component
            to those of the head component.
        """
        # scc = nx.kosaraju_strongly_connected_components(self.__graph)
        # scc = nx.strongly_connected_components_recursive(self.__graph)
        # fastest routine
        condensed_graph = condensation(
            self.__graph,
            scc=self.__get_ordered_scc(strongly_connected_components(self.__graph)),
        )
        mapping = condensed_graph.graph["mapping"]
        # Each edge of the condensed graph gathers the edges of the full graph
        # between the disciplines of two different components.
        edges = condensed_graph.edges
        for tail, head, names in self.__graph.edges(data=self.io):
            tail_component = mapping[tail]
            head_component = mapping[head]
            if tail_component != head_component:
                edge = edges[tail_component, head_component]
                edge.setdefault(self.io, set()).update(names)

        return condensed_graph

    def __get_ordered_scc(
        self, scc: Iterator[set[BaseDiscipline]]
    ) -> Iterator[list[BaseDiscipline]]:
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
    ) -> list[tuple[BaseDiscipline, BaseDiscipline, list[str]]]:
        """Return the couplings between the disciplines.

        Returns:
            The disciplines couplings, a coupling is
            composed of a discipline, one of its successor and the sorted
            variables names.
        """
        couplings = []
        for from_disc, to_disc, edge_names in self.__graph.edges(data=self.io):
            couplings += [(from_disc, to_disc, sorted(edge_names))]
        return couplings

    @staticmethod
    def __create_graph(disciplines: Iterable[BaseDiscipline]) -> DiGraph:
        """Create the full graph.

        The coupled inputs and outputs names are stored as an edge attributes named io.

        Args:
            disciplines: The disciplines to build the graph with.

        Returns:
            The graph of disciplines.
        """
        node_to_ios = {}

        for disc in disciplines:
            node_to_ios[disc] = (
                set(disc.io.input_grammar),
                set(disc.io.output_grammar),
            )

        graph = DiGraph()
        graph.add_nodes_from(disciplines)
        graph_add_edge = graph.add_edge
        # Create the graph edges by discovering the coupled disciplines
        for disc_i, (_, outputs_i) in node_to_ios.items():
            for disc_j, (inputs_j, _) in node_to_ios.items():
                if disc_i != disc_j:
                    coupled_io = outputs_i & inputs_j
                    if coupled_io:
                        graph_add_edge(disc_i, disc_j, io=coupled_io)

        return graph

    def __render_graph(
        self,
        full: bool,
        file_path: StrPath,
        clean_up: bool,
        show_edge_labels: bool,
    ) -> Digraph | None:
        """Render the graph when graphviz is installed.

        Args:
            full: Whether to render the full graph, otherwise the condensed one.
            file_path: The file path to save the graphical representation of the graph.
                Its suffix sets the format, e.g. `".pdf"`.
                If empty, the graphical representation is not saved.
            clean_up: Whether to remove the DOT source file;
                otherwise, keep it at `file_path` with the suffix `.gv`,
                or `.dot` when that of `file_path` is `.gv`.
            show_edge_labels: Whether to display the coupling variable names on edges.

        Returns:
            Either the graph or `None` when graphviz is not installed.

        Raises:
            ValueError: When graphviz is installed
                and `file_path` is not empty and has no suffix,
                a suffix that is not a Graphviz format
                or that of a Graphviz format opening a window.
        """
        if graphviz_is_missing:
            logger.warning("Cannot render graph: graphviz is not installed.")
            return None

        from gemseo.post.graph import _check_coupling_graph_file_path
        from gemseo.post.graph import draw_coupling_graph
        from gemseo.post.graph import save_coupling_graph

        # An empty path means that nothing is saved:
        # "" as a string, Path("") as a path-like object,
        # which cannot be told apart from Path(".")
        # whereas "." as a string is a path without extension.
        if isinstance(file_path, str):
            save = file_path != ""
        else:
            save = Path(file_path) != Path()

        if save:
            # Check the file path before drawing the graph, which can take long.
            _check_coupling_graph_file_path(file_path)

        digraph = draw_coupling_graph(
            self, full=full, show_edge_labels=show_edge_labels
        )
        if save:
            save_coupling_graph(digraph, file_path, clean_up=clean_up)

        return digraph

    def render_full_graph(
        self,
        file_path: StrPath,
        clean_up: bool = True,
        show_edge_labels: bool = True,
    ) -> Digraph | None:
        """Render the full graph.

        Args:
            file_path: The file path
                to save the graphical representation of the full graph.
                Its suffix sets the format, e.g. `".pdf"`.
                If empty, the graphical representation is not saved.
            clean_up: Whether to remove the DOT source file;
                otherwise, keep it at `file_path` with the suffix `.gv`,
                or `.dot` when that of `file_path` is `.gv`.
            show_edge_labels: Whether to display the coupling variable names on edges.

        Returns:
            Either the full graph or `None` when graphviz is not installed.
            The nodes of the graph are named as described in
            [draw_coupling_graph()][gemseo.post.graph.draw_coupling_graph].

        Raises:
            ValueError: When graphviz is installed
                and `file_path` is not empty and has no suffix,
                a suffix that is not a Graphviz format
                or that of a Graphviz format opening a window.
        """
        return self.__render_graph(True, file_path, clean_up, show_edge_labels)

    def render_condensed_graph(
        self,
        file_path: StrPath,
        clean_up: bool = True,
        show_edge_labels: bool = True,
    ) -> Digraph | None:
        """Render the condensed graph.

        Args:
            file_path: The file path
                to save the graphical representation of the condensed graph.
                Its suffix sets the format, e.g. `".pdf"`.
                If empty, the graphical representation is not saved.
            clean_up: Whether to remove the DOT source file;
                otherwise, keep it at `file_path` with the suffix `.gv`,
                or `.dot` when that of `file_path` is `.gv`.
            show_edge_labels: Whether to display the coupling variable names on edges.

        Returns:
            Either the condensed graph or `None` when graphviz is not installed.
            The nodes of the graph are named as described in
            [draw_coupling_graph()][gemseo.post.graph.draw_coupling_graph].

        Raises:
            ValueError: When graphviz is installed
                and `file_path` is not empty and has no suffix,
                a suffix that is not a Graphviz format
                or that of a Graphviz format opening a window.
        """
        return self.__render_graph(False, file_path, clean_up, show_edge_labels)

    @staticmethod
    def __get_leaves(graph: DiGraph) -> list[BaseDiscipline] | list[int]:
        """Return the leaf nodes of a graph.

        Args:
            graph: A graph.

        Returns:
            The graph leaf nodes.
        """
        return [n for n in graph.nodes if graph.out_degree(n) == 0]
