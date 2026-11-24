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
from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path
from subprocess import CalledProcessError
from typing import get_type_hints

import pytest
from graphviz import Digraph
from graphviz import Graph

import gemseo.post.graph
from gemseo.core.dependency_graph import DependencyGraph
from gemseo.discipline import AnalyticDiscipline
from gemseo.post.graph import draw_coupling_graph
from gemseo.post.graph import save_graph
from gemseo.problem.mdo.sellar import Sellar1
from gemseo.util.discipline import DummyDiscipline
from gemseo.util.testing.helper import assert_exception


@pytest.fixture
def recorded_get_edges():
    """A function to draw one edge per variable and the calls it recorded.

    Returns:
        The calls, as tuples of the tail name, the head name and the variable name,
        and the function.
    """
    calls = []

    def get_edges(tail, head, name):
        calls.append((
            None if tail is None else tail.name,
            None if head is None else head.name,
            name,
        ))
        return [{}]

    return calls, get_edges


def test_get_edges_one_edge_per_mapping(sellar_dependency_graph, snapshot) -> None:
    """Check that one edge is drawn per mapping returned for each variable."""

    def get_edges(tail, head, name):
        return [{"label": f"{name}[0]", "penwidth": "2"}, {"label": f"{name}[1]"}]

    source = draw_coupling_graph(sellar_dependency_graph, get_edges=get_edges).source
    assert source == snapshot
    assert len(re.findall(r"-> \w+ \[label=", source)) == 2 * 7
    assert source.count("penwidth=2") == 7
    assert source.count("h2 [style=invis]") == 1


@pytest.mark.parametrize(
    "edges",
    [{"label": "y_1"}, None, 1, "y_1", ["y_1"], [None]],
    ids=["mapping", "none", "integer", "string", "list_of_strings", "list_of_none"],
)
def test_get_edges_returning_no_sequence(
    sellar_dependency_graph, snapshot, edges
) -> None:
    """Check the error raised when get_edges does not return a sequence."""
    with assert_exception(TypeError, snapshot):
        draw_coupling_graph(
            sellar_dependency_graph, get_edges=lambda tail, head, name: edges
        )


@pytest.mark.parametrize(
    ("get_edges", "present", "absent"),
    [
        (
            lambda tail, head, name: [] if name.startswith("y") else [{}],
            ["label=c_1", "h2 [style=invis]"],
            ["y_1", "y_2"],
        ),
        (
            lambda tail, head, name: [{}] if head else [],
            ["label=y_1"],
            ["h2", "invis"],
        ),
    ],
    ids=["empty_list", "hidden_head"],
)
def test_get_edges_filtering(
    sellar_dependency_graph, get_edges, present, absent
) -> None:
    """Check that no edge is drawn when the function returns an empty list.

    And that a fully filtered leaf adds no hidden node.
    """
    source = draw_coupling_graph(sellar_dependency_graph, get_edges=get_edges).source
    for substring in present:
        assert substring in source
    for substring in absent:
        assert substring not in source


@pytest.fixture
def self_coupled_dependency_graph() -> DependencyGraph:
    """The dependency graph of a self-coupled discipline."""
    return DependencyGraph([AnalyticDiscipline({"x": "x/2"}, name="D")])


@pytest.mark.parametrize(
    ("dependency_graph", "expected_calls"),
    [
        (
            "sellar_dependency_graph",
            [
                ("Sellar1", "Sellar2", "y_1"),
                ("Sellar1", "SellarSystem", "y_1"),
                ("Sellar2", "Sellar1", "y_2"),
                ("Sellar2", "SellarSystem", "y_2"),
                ("SellarSystem", None, "c_1"),
                ("SellarSystem", None, "c_2"),
                ("SellarSystem", None, "obj"),
            ],
        ),
        # The same discipline is the tail and head of a self-coupling
        # and the output is also drawn as a leaf output, without head.
        ("self_coupled_dependency_graph", [("D", "D", "x"), ("D", None, "x")]),
    ],
    ids=["sellar", "self_coupling"],
)
def test_get_edges_arguments(
    request, recorded_get_edges, dependency_graph, expected_calls
) -> None:
    """Check the tail, head and name passed to the function."""
    calls, get_edges = recorded_get_edges
    draw_coupling_graph(request.getfixturevalue(dependency_graph), get_edges=get_edges)
    assert calls == expected_calls


def test_get_edges_self_coupling_effect(self_coupled_dependency_graph) -> None:
    """Check that get_edges draws the edge of a self-coupling."""
    source = draw_coupling_graph(
        self_coupled_dependency_graph,
        get_edges=lambda tail, head, name: [{"penwidth": "2"}],
    ).source
    assert "d0 -> d0 [label=x" in source
    assert source.count("penwidth=2") == 2


def test_get_edges_label_overrides_show_edge_labels(sellar_dependency_graph) -> None:
    """Check that a label returned by the function wins over show_edge_labels."""
    source = draw_coupling_graph(
        sellar_dependency_graph,
        show_edge_labels=False,
        get_edges=lambda tail, head, name: [{"label": f"L_{name}"}, {}],
    ).source
    assert "label=L_y_1" in source
    # The edges with a label have a label tooltip, the others an edge tooltip.
    assert source.count("labeltooltip") == 7
    assert source.count("edgetooltip") == 7


@pytest.mark.parametrize(
    ("show_edge_labels", "present_key", "absent_key"),
    [(True, "labeltooltip", "edgetooltip"), (False, "edgetooltip", "labeltooltip")],
    ids=["labelled", "unlabelled"],
)
def test_get_edges_tooltips(
    sellar_dependency_graph, snapshot, show_edge_labels, present_key, absent_key
) -> None:
    """Check the label tooltips of the labelled edges.

    And the edge tooltips of the unlabelled edges.
    """
    source = draw_coupling_graph(
        sellar_dependency_graph,
        show_edge_labels=show_edge_labels,
        get_edges=lambda tail, head, name: [{}],
    ).source
    assert absent_key not in source
    assert source.count(present_key) == 7
    assert bool(re.search(r"-> \w+ \[label=", source)) is show_edge_labels
    assert "Name in discipline 'Sellar2'\n\ny_1, y_1, y_1" in source
    assert "Name in discipline 'SellarSystem'\n\nc_1, c_1" in source
    assert "c_1, c_1, c_2" not in source
    assert source == snapshot


@pytest.mark.parametrize(
    ("kwargs", "counts"),
    [
        (
            {"get_edges": lambda tail, head, name: [{"labeltooltip": "T"}]},
            {"labeltooltip=T": 7},
        ),
        (
            {"get_edges": lambda tail, head, name: [{"label": ""}]},
            {"labeltooltip": 0, "edgetooltip": 7},
        ),
    ],
    ids=["overridden_tooltip", "empty_label"],
)
def test_get_edges_tooltip_counts(sellar_dependency_graph, kwargs, counts) -> None:
    """Check the number of edge tooltips when get_edges gives the attributes."""
    source = draw_coupling_graph(sellar_dependency_graph, **kwargs).source
    for substring, count in counts.items():
        assert source.count(substring) == count


def test_get_edges_label_tooltip_of_variable_edge(sellar_dependency_graph) -> None:
    """Check the label tooltip of a variable edge whose label comes from get_edges."""
    source = draw_coupling_graph(
        sellar_dependency_graph,
        get_edges=lambda tail, head, name: [{"label": "L"}],
        input_names=["x_1"],
    ).source
    assert "label=L" in source
    assert "labeltooltip=\"Global name, Name in discipline 'Sellar1'" in source


def test_input_and_output_nodes(sellar_dependency_graph, snapshot) -> None:
    """Check the nodes of the inputs and outputs and their edges."""
    source = draw_coupling_graph(
        sellar_dependency_graph,
        input_names=["x_shared", "x_1"],
        output_names=["obj", "c_1"],
    ).source
    assert source == snapshot


def test_input_and_output_names_order(sellar_dependency_graph) -> None:
    """Check that the graph does not depend on the order of the variable names."""
    assert (
        draw_coupling_graph(
            sellar_dependency_graph,
            input_names=["x_shared", "x_1"],
            output_names=["obj", "c_1"],
        ).source
        == draw_coupling_graph(
            sellar_dependency_graph,
            input_names=["x_1", "x_shared"],
            output_names=["c_1", "obj"],
        ).source
    )


def test_input_and_output_node_ranks(sellar_dependency_graph) -> None:
    """Check that the input and output nodes are on the first and last ranks."""
    source = draw_coupling_graph(
        sellar_dependency_graph, input_names=["x_1"], output_names=["obj"]
    ).source
    assert "subgraph source_nodes {\n\t\trank=source" in source
    assert "subgraph sink_nodes {\n\t\trank=sink" in source


@pytest.mark.parametrize(
    ("kwargs", "has_source_nodes"),
    [({"input_names": ["x_1"]}, True), ({"output_names": ["obj"]}, False)],
    ids=["input_names", "output_names"],
)
def test_no_subgraph_without_nodes(
    sellar_dependency_graph, kwargs, has_source_nodes
) -> None:
    """Check that a subgraph is only created when it has nodes."""
    source = draw_coupling_graph(sellar_dependency_graph, **kwargs).source
    assert ("source_nodes" in source) is has_source_nodes
    assert ("sink_nodes" in source) is not has_source_nodes


@pytest.mark.parametrize(
    ("kwargs", "present", "absent"),
    [
        ({}, ["h2 [style=invis]"], []),
        ({"input_names": ["x_1"]}, ["h2 [style=invis]"], []),
        # The output nodes replace the hidden head nodes
        # and the outputs that are not selected are not drawn.
        ({"output_names": ["obj"]}, [], ["h2", "invis", "c_1"]),
    ],
    ids=["default", "input_names", "output_names"],
)
def test_hidden_nodes(sellar_dependency_graph, kwargs, present, absent) -> None:
    """Check that the hidden head nodes are drawn unless there are output nodes."""
    source = draw_coupling_graph(sellar_dependency_graph, **kwargs).source
    for substring in present:
        assert substring in source
    for substring in absent:
        assert substring not in source


def test_output_of_non_leaf_discipline(sellar_dependency_graph) -> None:
    """Check that an output of a discipline that is not a leaf has a sink edge."""
    source = draw_coupling_graph(sellar_dependency_graph, output_names=["y_2"]).source
    assert "d1 -> out0" in source
    assert source.count("out0") == 2  # the node and its edge
    # The coupling edges are kept.
    assert "d1 -> d0" in source


def test_discipline_without_drawn_edge() -> None:
    """Check that a discipline without any drawn edge is drawn."""
    graph = draw_coupling_graph(
        DependencyGraph([
            AnalyticDiscipline({"y": "x"}, name="A"),
            AnalyticDiscipline({"z": "y"}, name="B"),
            AnalyticDiscipline({"w": "v"}, name="C"),
        ]),
        output_names=["z"],
    )
    node_names, edges = get_nodes_and_edges(graph)
    assert node_names == ["d0", "d1", "d2", "out0"]
    assert edges == [("d0", "d1"), ("d1", "out0")]
    assert "\td2 [label=C]" in graph.source


def test_get_node_attributes_arguments(sellar_dependency_graph) -> None:
    """Check the arguments passed to get_node_attributes."""
    calls = []

    def get_node_attributes(name, is_input):
        calls.append((name, is_input))
        return {}

    draw_coupling_graph(
        sellar_dependency_graph,
        input_names=["x_1"],
        output_names=["obj"],
        get_node_attributes=get_node_attributes,
    )
    assert sorted(calls) == [("obj", False), ("x_1", True)]


def test_get_node_attributes_effect(sellar_dependency_graph) -> None:
    """Check that the attributes of get_node_attributes override the default ones."""
    source = draw_coupling_graph(
        sellar_dependency_graph,
        input_names=["x_1"],
        output_names=["obj"],
        get_node_attributes=lambda name, is_input: (
            {"color": "blue", "label": f"{name} (in)"} if is_input else {"shape": "box"}
        ),
    ).source
    assert 'in0 [label="x_1 (in)" color=blue' in source
    assert "out0 [label=obj" in source
    # Only the attributes returned by the function override the default ones.
    assert source.count("shape=box") == 1
    assert source.count("shape=plaintext") == 1


def test_get_edges_with_variable_nodes(
    sellar_dependency_graph, recorded_get_edges
) -> None:
    """Check the tail and head passed to get_edges for the variable nodes."""
    calls, get_edges = recorded_get_edges
    draw_coupling_graph(
        sellar_dependency_graph,
        get_edges=get_edges,
        input_names=["x_1"],
        output_names=["obj", "y_2"],
    )
    # The coupling edges come first, there are no calls for hidden heads.
    assert len(calls) == 4 + 4
    assert calls[4:] == [
        (None, "Sellar1", "x_1"),
        (None, "SellarSystem", "x_1"),
        ("SellarSystem", None, "obj"),
        ("Sellar2", None, "y_2"),
    ]


def test_get_edges_filters_variable_nodes(sellar_dependency_graph) -> None:
    """Check that get_edges can filter and style the edges of the variable nodes."""
    source = draw_coupling_graph(
        sellar_dependency_graph,
        get_edges=lambda tail, head, name: (
            []
            if tail is None and head is not None and head.name == "Sellar1"
            else [{"penwidth": "3"}]
        ),
        input_names=["x_1"],
        output_names=["obj"],
    ).source
    assert "in0 -> d0" not in source
    assert "in0 -> d2" in source
    assert "d2 -> out0" in source
    assert "penwidth=3" in source
    # The nodes are drawn even if their edges are filtered.
    assert "in0 [" in source


def test_variable_edge_tooltips(sellar_dependency_graph) -> None:
    """Check the tooltips of the edges of the variable nodes."""
    source = draw_coupling_graph(
        sellar_dependency_graph, input_names=["x_1"], output_names=["obj"]
    ).source
    assert "Global name, Name in discipline 'Sellar1'\n\nx_1, x_1" in source
    assert "Global name, Name in discipline 'SellarSystem'\n\nobj, obj" in source
    assert "edgetooltip" in source


def get_nodes_and_edges(graph: Digraph) -> tuple[list[str], list[tuple[str, str]]]:
    """Return the nodes and edges of a graph as laid out by Graphviz.

    Args:
        graph: The graph.

    Returns:
        The sorted names of the nodes and the sorted tail and head names of the edges.
    """
    layout = json.loads(graph.pipe(format="json"))
    node_names = {
        node["_gvid"]: node["name"] for node in layout["objects"] if "nodes" not in node
    }
    edges = sorted(
        (node_names[edge["tail"]], node_names[edge["head"]])
        for edge in layout.get("edges", [])
    )
    return sorted(node_names.values()), edges


def test_variable_edges_of_homonymous_disciplines() -> None:
    """Check the edges of homonymous disciplines and their tooltips."""
    graph = DependencyGraph([Sellar1(), Sellar1()])
    source = draw_coupling_graph(graph, input_names=["x_1"]).source
    assert source.count("edgetooltip=\"Global name, Name in discipline 'Sellar1'") == 2
    assert source.count("labeltooltip=\"Global name, Name in discipline 'Sellar1'") == 2
    assert source.count("in0 ->") == 2


def test_homonymous_disciplines_tooltip() -> None:
    """Check the tooltip headers of an edge between homonymous disciplines."""
    graph = DependencyGraph([
        AnalyticDiscipline({"y": "x"}, name="A"),
        AnalyticDiscipline({"z": "y"}, name="A"),
    ])
    source = draw_coupling_graph(graph).source
    # The headers tell the two disciplines apart
    # and do not read as a self-coupling.
    assert (
        'd0 -> d1 [label=y labeltooltip="Global name, '
        "Name in source discipline 'A', Name in target discipline 'A'"
    ) in source
    assert "Output name" not in source
    # The tooltip of the leaf has a single discipline column.
    assert "d1 -> h1 [label=z labeltooltip=\"Global name, Name in discipline 'A'" in (
        source
    )


def test_homonymous_disciplines_condensed_graph() -> None:
    """Check that the condensed graph draws one node per homonymous discipline."""
    graph = draw_coupling_graph(DependencyGraph([Sellar1(), Sellar1()]), full=False)
    node_names, edges = get_nodes_and_edges(graph)
    assert node_names == ["d0", "d1", "h0", "h1"]
    assert sorted(tail for tail, _ in edges) == ["d0", "d1"]
    assert "\td0 [label=Sellar1]" in graph.source
    assert "\td1 [label=Sellar1]" in graph.source


@pytest.fixture
def disciplines_with_escaped_names() -> list[AnalyticDiscipline]:
    """Disciplines whose names must be escaped, the first two being coupled."""
    return [
        AnalyticDiscipline({"y": "x+z"}, name="ns:a"),
        AnalyticDiscipline({"z": "y"}, name="b%"),
        AnalyticDiscipline({"t": "y"}, name="c"),
    ]


@pytest.mark.parametrize(
    ("full", "expected_node_names", "expected_edges", "expected_label_lines"),
    [
        (
            True,
            ["d0", "d1", "d2", "h2"],
            [("d0", "d1"), ("d0", "d2"), ("d1", "d0"), ("d2", "h2")],
            ['\td0 [label="ns:a"]', '\td1 [label="b%"]'],
        ),
        # The node c is the node 0 of the condensed graph.
        (
            False,
            ["d2", "h0", "scc1"],
            [("d2", "h0"), ("scc1", "d2")],
            ['\tscc1 [label="MDA of ns:a, b%"]'],
        ),
    ],
    ids=["full", "condensed"],
)
def test_escaped_discipline_name(
    disciplines_with_escaped_names,
    full,
    expected_node_names,
    expected_edges,
    expected_label_lines,
) -> None:
    """Check that a discipline name to escape is a single node with this label.

    And that an MDA of such disciplines is a single node.
    """
    graph = draw_coupling_graph(
        DependencyGraph(disciplines_with_escaped_names), full=full
    )
    node_names, edges = get_nodes_and_edges(graph)
    assert node_names == expected_node_names
    assert edges == expected_edges
    for line in expected_label_lines:
        assert line in graph.source


@pytest.mark.parametrize(
    ("full", "kwargs", "expected_lines"),
    [
        (
            True,
            {"input_names": ["<x>"]},
            ['\td0 [label="<A>"]', '\t\tin0 [label="<x>"', '\td0 -> h0 [label="<y>"'],
        ),
        (
            True,
            {
                "input_names": ["<x>"],
                "get_node_attributes": lambda name, is_input: {"label": "<<b>x</b>>"},
                "get_edges": lambda tail, head, name: [{"label": "<<i>y</i>>"}],
            },
            ["\t\tin0 [label=<<b>x</b>>", "\td0 -> h0 [label=<<i>y</i>>"],
        ),
        (False, {}, ['\td0 [label="<A>"]', '\td0 -> h0 [label="<y>"']),
    ],
    ids=["full", "user_labels", "condensed"],
)
def test_html_like_names(full, kwargs, expected_lines) -> None:
    """Check that the names looking like HTML-like labels are labels as is.

    And that the labels returned by the user functions are left as is.
    """
    discipline = DummyDiscipline("<A>", ["<x>"], ["<y>"])
    graph = draw_coupling_graph(DependencyGraph([discipline]), full=full, **kwargs)
    for line in expected_lines:
        assert line in graph.source

    # Graphviz can lay the graph out.
    node_names, _ = get_nodes_and_edges(graph)
    assert "d0" in node_names


def test_tooltip_order() -> None:
    """Check that the tooltip of an edge lists its variables in alphabetical order."""
    names = ["f", "c", "a", "e", "b", "d"]
    graph = DependencyGraph([
        AnalyticDiscipline(dict.fromkeys(names, "x"), name="A"),
        AnalyticDiscipline({"z": "+".join(names)}, name="B"),
    ])
    source = draw_coupling_graph(graph).source
    assert '\n\na, a, a\nb, b, b\nc, c, c\nd, d, d\ne, e, e\nf, f, f"' in source


def test_hidden_node_name_clash() -> None:
    """Check that the hidden head node of a leaf does not hide another discipline."""
    graph = draw_coupling_graph(
        DependencyGraph([
            AnalyticDiscipline({"y": "x"}, name="A"),
            AnalyticDiscipline({"z": "x"}, name="_A"),
        ])
    )
    node_names, edges = get_nodes_and_edges(graph)
    assert node_names == ["d0", "d1", "h0", "h1"]
    assert edges == [("d0", "h0"), ("d1", "h1")]
    assert "\td1 [label=_A]" in graph.source
    assert "\th0 [style=invis]" in graph.source
    assert "\th1 [style=invis]" in graph.source


def test_namespaced_variable_nodes() -> None:
    """Check that a namespaced variable has a single node attached to its edge."""
    discipline = AnalyticDiscipline({"y": "x"}, name="D")
    discipline.add_namespace_to_input("x", "ns")
    discipline.add_namespace_to_output("y", "ns")
    graph = draw_coupling_graph(
        DependencyGraph([discipline]), input_names=["ns:x"], output_names=["ns:y"]
    )
    node_names, edges = get_nodes_and_edges(graph)
    assert node_names == ["d0", "in0", "out0"]
    assert edges == [("d0", "out0"), ("in0", "d0")]
    assert '\t\tin0 [label="ns:x"' in graph.source
    assert '\t\tout0 [label="ns:y"' in graph.source


def test_variable_node_names(sellar_dependency_graph) -> None:
    """Check the names of the variable nodes follow the sorted variable names."""
    source = draw_coupling_graph(
        sellar_dependency_graph,
        input_names=["x_shared", "x_1", "x_1"],
        output_names=["obj", "c_1"],
    ).source
    assert "\t\tin0 [label=x_1" in source
    assert "\t\tin1 [label=x_shared" in source
    assert "\t\tout0 [label=c_1" in source
    assert "\t\tout1 [label=obj" in source


def test_coupling_variable_as_input(sellar_dependency_graph) -> None:
    """Check that a coupling given as input keeps its coupling edges."""
    source = draw_coupling_graph(sellar_dependency_graph, input_names=["y_1"]).source
    assert "in0 -> d1" in source
    assert "in0 -> d2" in source
    assert "d0 -> d1 [label=y_1" in source
    assert "d0 -> d2 [label=y_1" in source


def test_copy(sellar_dependency_graph) -> None:
    """Check that a copy of the graph has the same source."""
    graph = draw_coupling_graph(sellar_dependency_graph)
    assert graph.copy().source == graph.source


def test_subgraph(sellar_dependency_graph) -> None:
    """Check that the graph can have subgraphs."""
    graph = draw_coupling_graph(sellar_dependency_graph)
    with graph.subgraph(name="cluster_0") as subgraph:
        subgraph.node("d0")

    assert "subgraph cluster_0 {" in graph.source


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_names": ["x_1", "obj", "a"]},
        {"output_names": ["obj", "x_1", "b"]},
    ],
    ids=["input_names", "output_names"],
)
def test_unknown_names(sellar_dependency_graph, snapshot, kwargs) -> None:
    """Check the error raised for names that are not inputs or outputs."""
    with assert_exception(ValueError, snapshot):
        draw_coupling_graph(sellar_dependency_graph, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"get_edges": lambda tail, head, name: [{}]},
        {"input_names": ["x_1"]},
        {"output_names": ["obj"]},
        {"get_node_attributes": lambda name, is_input: {}},
    ],
    ids=["get_edges", "input_names", "output_names", "get_node_attributes"],
)
def test_condensed_graph(sellar_dependency_graph, snapshot, kwargs) -> None:
    """Check that the edges and variable names are rejected for the condensed graph."""
    with assert_exception(ValueError, snapshot):
        draw_coupling_graph(sellar_dependency_graph, full=False, **kwargs)


def test_type_hints() -> None:
    """Check that the type hints of the module can be resolved at runtime."""
    assert set(get_type_hints(gemseo.post.graph)) == {
        "_device_formats",
        "EdgeGetter",
        "NodeAttributeGetter",
    }


@pytest.fixture
def coupling_graph(sellar_dependency_graph) -> Digraph:
    """The coupling graph of the Sellar disciplines."""
    return draw_coupling_graph(sellar_dependency_graph)


@pytest.mark.parametrize("clean_up", [False, True], ids=["keep_source", "clean_up"])
@pytest.mark.parametrize(
    ("file_name", "source_name"),
    [("graph.dot", "graph.gv"), ("graph.gv", "graph.dot")],
    ids=["dot", "gv"],
)
@pytest.mark.parametrize("directed", [False, True], ids=["undirected", "directed"])
def test_save_dot_language_file(
    request, tmp_wd, directed, file_name, source_name, clean_up
):
    """Check that a directed or undirected graph can be saved in the DOT language.

    The source file is kept or not;
    when kept,
    it has the other extension.
    """
    if directed:
        graph = request.getfixturevalue("coupling_graph")
        header = "digraph {"
    else:
        graph = Graph()
        graph.edge("a", "b")
        header = "graph {"

    save_graph(graph, file_name, clean_up=clean_up)
    # The output file is laid out from the source file.
    output = Path(file_name).read_text()
    assert output.startswith(header)
    assert "pos=" in output
    sources = [] if clean_up else [source_name]
    assert sorted(os.listdir(tmp_wd)) == sorted([file_name, *sources])
    for source in sources:
        assert Path(source).read_text() == graph.source


def test_save_ignores_graph_directory(tmp_wd):
    """Check that the files are saved relative to the working directory.

    And not relative to the directory of the graph.
    """
    digraph = Digraph(directory="sub")
    digraph.node("a")
    save_graph(digraph, "graph.svg", clean_up=False)
    assert sorted(os.listdir(tmp_wd)) == ["graph.gv", "graph.svg"]


@pytest.mark.parametrize("clean_up", [False, True], ids=["keep_source", "clean_up"])
def test_save_in_missing_directory(tmp_wd, coupling_graph, clean_up):
    """Check that a graph can be saved in a directory that does not exist yet."""
    save_graph(coupling_graph, "sub/sub/graph.svg", clean_up=clean_up)
    assert Path("sub/sub/graph.svg").read_text().startswith("<?xml")


@pytest.mark.parametrize(
    ("clean_up", "existing_directories", "foreign_files", "expected_paths"),
    [
        (True, (), (), ()),
        (False, (), (), ()),
        (True, (), ("sub/foreign.txt",), ("sub", "sub/foreign.txt")),
        (
            True,
            (),
            ("sub/sub/foreign.txt",),
            ("sub", "sub/sub", "sub/sub/foreign.txt"),
        ),
        (True, ("sub",), (), ("sub",)),
        (True, ("sub", "sub/sub"), (), ("sub", "sub/sub")),
    ],
    ids=[
        "created",
        "created_keep_source",
        "foreign_file_in_sub",
        "foreign_file_in_sub_sub",
        "existing_sub",
        "existing_sub_sub",
    ],
)
def test_save_failure_removes_created_directories(
    tmp_wd,
    coupling_graph,
    monkeypatch,
    clean_up,
    existing_directories,
    foreign_files,
    expected_paths,
):
    """Check that the directories created to save a graph are removed on failure.

    And that the directories that existed before are kept,
    as well as the created directories
    to which another process has written a file meanwhile.
    """
    for directory in existing_directories:
        Path(directory).mkdir()

    def render(self, filename, outfile, format):  # noqa: A002
        # Write the source and output files, then those of another process, then fail.
        Path(filename).write_text("foo")
        Path(outfile).write_text("foo")
        for foreign_file in foreign_files:
            Path(foreign_file).write_text("foo")

        msg = "dot failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(Digraph, "render", render)
    with pytest.raises(RuntimeError):
        save_graph(coupling_graph, "sub/sub/graph.svg", clean_up=clean_up)

    assert sorted(path.as_posix() for path in Path().rglob("*")) == sorted(
        expected_paths
    )


def test_save_mkdir_failure_removes_created_directories(
    tmp_wd, coupling_graph, monkeypatch
):
    """Check that the created directories are removed when creating one fails."""
    mkdir = Path.mkdir
    failing_directory = Path("sub/sub").absolute()

    def mkdir_failing_at_sub_sub(self, *args, **kwargs):
        # Create "sub", then fail to create "sub/sub".
        if self == failing_directory:
            mkdir(self.parent, parents=True, exist_ok=True)
            msg = "mkdir failed"
            raise PermissionError(msg)

        mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir_failing_at_sub_sub)
    with pytest.raises(PermissionError, match="mkdir failed"):
        save_graph(coupling_graph, "sub/sub/graph.svg")

    assert list(Path().rglob("*")) == []


@pytest.mark.parametrize("clean_up", [False, True], ids=["keep_source", "clean_up"])
@pytest.mark.parametrize(
    "existing_files", [(), ("graph.gv", "graph.svg")], ids=["none", "both"]
)
def test_save_failure_removes_written_files(tmp_wd, clean_up, existing_files):
    """Check that the files written to save a graph are removed on failure.

    And that the files that existed before are kept.
    """
    for file_name in existing_files:
        Path(file_name).write_text("foo")

    digraph = Digraph()
    digraph.node("a")
    # Make dot fail with a syntax error.
    digraph.body.append("{{{\n")
    with pytest.raises(CalledProcessError):
        save_graph(digraph, "graph.svg", clean_up=clean_up)

    assert sorted(os.listdir(tmp_wd)) == sorted(existing_files)


@pytest.mark.parametrize("file_format", ["xdot1.2", "xdot1.4"])
def test_save_format_with_dots(tmp_wd, coupling_graph, file_format):
    """Check that a graph can be saved in a format whose name contains dots."""
    file_name = f"graph.{file_format}"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save_graph(coupling_graph, file_name, clean_up=False)

    assert f"xdotversion={file_format[4:]}" in Path(file_name).read_text()
    assert sorted(os.listdir(tmp_wd)) == sorted([file_name, "graph.gv"])


def test_save_preserves_gv_file(tmp_wd, coupling_graph):
    """Check that saving with clean_up=True preserves a sibling GV file."""
    source_path = Path("graph.gv")
    source_path.write_text("foo")
    save_graph(coupling_graph, "graph.pdf")
    assert source_path.read_text() == "foo"
    assert Path("graph.pdf").exists()


def test_save_keeps_filename(tmp_wd, coupling_graph):
    """Check that saving the graph does not change its filename."""
    filename = coupling_graph.filename
    save_graph(coupling_graph, "graph.pdf")
    assert coupling_graph.filename == filename


@pytest.mark.parametrize(
    "file_path",
    [
        "coupling_graph",
        "coupling_graph.v2",
        "coupling_graph.gtk",
        "coupling_graph.x11",
        "coupling_graph.XLIB",
        ".",
    ],
    ids=["no_suffix", "unknown_suffix", "gtk", "x11", "xlib", "current_directory"],
)
def test_save_with_wrong_suffix(tmp_wd, coupling_graph, file_path, snapshot):
    """Check the error raised when the suffix of the file path is not a format."""
    with assert_exception(ValueError, snapshot):
        save_graph(coupling_graph, file_path)

    assert not os.listdir(tmp_wd)
