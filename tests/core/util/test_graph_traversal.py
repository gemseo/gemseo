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
#        :author: Gilberto Ruiz Jimenez
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from networkx import DiGraph

from gemseo.core.util._graph_traversal import compute_reachable_nodes


def test_sources_belong_to_the_reachable_set() -> None:
    """Pin that the sources and everything downstream of them are returned."""
    graph = DiGraph()
    graph.add_edges_from([("a", "b"), ("b", "c")])
    graph.add_node("d")

    assert compute_reachable_nodes(graph, {"a"}) == {"a", "b", "c"}


def test_cycle_is_fully_reached_without_looping_forever() -> None:
    """Pin that a cycle terminates and reaches the whole cycle plus downstream nodes.

    The ``reached`` membership check is what stops the while loop from
    revisiting the nodes of the cycle indefinitely.
    """
    graph = DiGraph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])

    assert compute_reachable_nodes(graph, {"a"}) == {"a", "b", "c", "d"}


def test_empty_sources_yield_an_empty_set() -> None:
    """Pin that an empty set of sources yields an empty set."""
    graph = DiGraph()
    graph.add_edges_from([("a", "b")])

    assert compute_reachable_nodes(graph, set()) == set()
