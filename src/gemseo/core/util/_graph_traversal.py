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
"""Graph traversal algorithms.

The building blocks shared by the modules that walk a
[DependencyGraph][gemseo.core.dependency_graph.DependencyGraph],
whatever they do with the traversal result:
differentiation, namespace propagation, etc.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import TypeVar

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    from networkx import DiGraph

_NodeT = TypeVar("_NodeT", bound=Hashable)


class DisciplineIOs(NamedTuple):
    """A selection of the inputs and outputs of a discipline.

    The meaning of the selection is defined by the algorithm that builds it.
    """

    inputs: frozenset[str]
    """The names of the selected inputs."""

    outputs: frozenset[str]
    """The names of the selected outputs."""


def compute_reachable_nodes(
    graph: DiGraph,
    sources: AbstractSet[_NodeT],
) -> set[_NodeT]:
    """Compute the nodes reachable from sources in a directed graph.

    Args:
        graph: The directed graph.
        sources: The set of source nodes to explore from.

    Returns:
        The set of nodes reachable from the source nodes, sources included.
    """
    reached = set(sources)
    stack = list(reached)
    while stack:
        for successor in graph.successors(stack.pop()):
            if successor not in reached:
                reached.add(successor)
                stack.append(successor)
    return reached
