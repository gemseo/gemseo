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
from __future__ import annotations

from gemseo.core.coupling_structure import CouplingStructure
from gemseo.core.dependency_graph import DependencyGraph
from gemseo.core.derivatives.graph_traversal import set_differentiated_ios
from gemseo.core.derivatives.graph_traversal import set_mda_differentiated_ios
from gemseo.problems.mdo.scalable.linear.disciplines_generator import (
    create_disciplines_from_desc,
)


def test_set_differentiated_ios_basic() -> None:
    """Test the differentiated inputs and outputs graph calculations."""
    a, b, c, d, e, h, i = create_disciplines_from_desc([
        ("A", ["x", "a"], ["p", "q", "xx"]),
        ("B", ["x", "y"], ["r"]),
        ("C", ["x"], ["m"]),
        ("D", ["q", "r", "w"], ["s"]),
        ("E", ["s"], ["o"]),
        ("H", ["z"], ["w"]),
        ("I", ["m"], ["n"]),
    ])

    graph = DependencyGraph([a, b, c, d, e]).graph
    mapping = set_differentiated_ios(graph, ["x"], ["o"])

    assert a._differentiated_input_names == ["x"]
    assert a._differentiated_output_names == ["q"]

    assert b._differentiated_input_names == ["x"]
    assert b._differentiated_output_names == ["r"]

    assert not c._differentiated_input_names
    assert not c._differentiated_output_names

    assert sorted(d._differentiated_input_names) == ["q", "r"]
    assert d._differentiated_output_names == ["s"]

    assert e._differentiated_input_names == ["s"]
    assert e._differentiated_output_names == ["o"]

    assert not h._differentiated_input_names
    assert not h._differentiated_output_names

    assert not i._differentiated_input_names
    assert not i._differentiated_output_names

    assert set(mapping) == {a, b, d, e}
    assert mapping[a] == (frozenset(["x"]), frozenset(["q"]))
    assert mapping[b] == (frozenset(["x"]), frozenset(["r"]))
    assert mapping[d] == (frozenset(["q", "r"]), frozenset(["s"]))
    assert mapping[e] == (frozenset(["s"]), frozenset(["o"]))


def test_set_mda_differentiated_ios_basic() -> None:
    """Test the differentiated inputs and outputs graph calculations."""
    a, b, c, d, e = create_disciplines_from_desc([
        ("A", ["x1", "x2"], ["a"]),
        ("B", ["x3"], ["b"]),
        ("C", ["a", "b", "y1"], ["y2", "y3"]),
        ("D", ["y2"], ["y1", "o1"]),
        ("E", ["y1", "y2", "y3", "x1"], ["o2"]),
    ])
    coupl = CouplingStructure([a, b, c, d, e])
    mapping = set_mda_differentiated_ios(coupl.graph.graph, ["x1"], ["o1"])

    assert a._differentiated_input_names == ["x1"]
    assert a._differentiated_output_names == ["a"]

    assert not b._differentiated_input_names
    assert not b._differentiated_output_names

    assert sorted(c._differentiated_input_names) == ["a", "y1"]
    assert c._differentiated_output_names == ["y2"]

    assert d._differentiated_input_names == ["y2"]
    assert sorted(d._differentiated_output_names) == ["o1", "y1"]

    assert not e._differentiated_input_names
    assert not e._differentiated_output_names

    assert set(mapping) == {a, c, d}
    assert mapping[a] == (frozenset(["x1"]), frozenset(["a"]))
    assert mapping[c] == (frozenset(["a", "y1"]), frozenset(["y2"]))
    assert mapping[d] == (frozenset(["y2"]), frozenset(["o1", "y1"]))


def test_set_differentiated_ios_seed_and_edge() -> None:
    """Test a discipline whose diff_inputs/outputs come from both seed and edge data.

    B owns "x" (src_input seed) and also receives "a" from A via edge,
    so diff_inputs = {"x", "a"}. B owns "y" (src_output seed) and also
    feeds "b" to C via edge, so diff_outputs = {"b", "y"}.
    """
    a, b, c = create_disciplines_from_desc([
        ("A", ["x"], ["a"]),
        ("B", ["x", "a"], ["b", "y"]),
        ("C", ["b"], ["y"]),
    ])
    graph = DependencyGraph([a, b, c]).graph
    mapping = set_differentiated_ios(graph, ["x"], ["y"])

    assert set(mapping) == {a, b, c}
    assert mapping[a] == (frozenset(["x"]), frozenset(["a"]))
    assert mapping[b] == (frozenset(["a", "x"]), frozenset(["b", "y"]))
    assert mapping[c] == (frozenset(["b"]), frozenset(["y"]))


def test_set_differentiated_ios_no_active_path() -> None:
    """Test that the mapping is empty when no path connects inputs to outputs."""
    a, b = create_disciplines_from_desc([
        ("A", ["x"], ["a"]),
        ("B", ["b"], ["y"]),
    ])
    graph = DependencyGraph([a, b]).graph
    assert set_differentiated_ios(graph, ["x"], ["y"]) == {}


def test_set_mda_differentiated_ios_self_coupling() -> None:
    """Test that self-coupling variables are added to both inputs and outputs."""
    a, b = create_disciplines_from_desc([
        ("A", ["x", "y"], ["a", "y"]),
        ("B", ["a"], ["o"]),
    ])
    coupl = CouplingStructure([a, b])
    mapping = set_mda_differentiated_ios(coupl.graph.graph, ["x"], ["o"])

    assert set(mapping) == {a, b}
    assert mapping[a] == (frozenset(["x", "y"]), frozenset(["a", "y"]))
    assert mapping[b] == (frozenset(["a"]), frozenset(["o"]))


def test_set_mda_differentiated_ios_with_residuals() -> None:
    """Test that residual/state pairs from residual_to_state_variable are registered."""
    a, b = create_disciplines_from_desc(
        [
            ("A", ["x", "w"], ["a", "r", "w"]),
            ("B", ["a", "w"], ["y"]),
        ],
    )
    a.io.residual_to_state_variable = {"r": "w"}
    coupl = CouplingStructure([a, b])
    mapping = set_mda_differentiated_ios(coupl.graph.graph, ["x"], ["y"])

    assert set(mapping) == {a, b}
    assert mapping[a] == (frozenset(["w", "x"]), frozenset(["a", "r", "w"]))
    assert mapping[b] == (frozenset(["a", "w"]), frozenset(["y"]))
