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

from pathlib import Path
from typing import Any
from typing import Final

import pytest

from gemseo.post.graph_view import GraphView

parametrize_edge_dir: Final = pytest.mark.parametrize(
    ("is_directed", "edge_dir"),
    [(False, "none"), (True, "forward")],
    ids=["undirected", "directed"],
)
"""Parametrize a test with the argument is_directed and the direction written on each
edge."""


def test_graph_view(tmp_wd) -> None:
    """Check GraphView."""
    graph = GraphView()
    graph.edge("A", "B", "foo")
    graph.edge(
        "A",
        "C",
        "bar",
        color="red",
        fontcolor="blue",
        penwidth="2.0",
        dir="none",
        fontsize="7",
    )
    graph.edge("A", "A", "baz")
    graph.edge("B", "B")
    graph.node(
        "B",
        fillcolor="black",
        color="red",
        fontcolor="white",
        penwidth="4.0",
        shape="box",
    )
    graph.hide_node("C")
    graph.node("D", fillcolor="yellow")

    file_path = Path("graph_view.png")
    graph.visualize(file_path=file_path, clean_up=False)
    file_path = file_path.with_suffix(".dot")
    assert (
        file_path.read_text().strip()
        == (Path(__file__).parent / file_path).read_text().strip()
    )
    assert file_path.exists()


@pytest.mark.parametrize("use_directed_edges", [False, True])
def test_use_directed_edges(tmp_wd, use_directed_edges) -> None:
    """Check the argument use_directed_edges passed at instantiation."""
    graph = GraphView(use_directed_edges)
    graph.edge("A", "B")
    file_path = Path(f"use_directed_edges_{str(use_directed_edges).lower()}.png")
    graph.visualize(file_path=file_path, clean_up=False)
    file_path = file_path.with_suffix(".dot")
    assert (
        file_path.read_text().strip()
        == (Path(__file__).parent / file_path).read_text().strip()
    )


@pytest.mark.parametrize("clean_up", [False, True])
def test_clean_up(tmp_wd, clean_up) -> None:
    """Check the argument clean_up."""
    GraphView().visualize(clean_up=clean_up)
    assert Path("graph_view.dot").exists() is not clean_up


def test_repr_html() -> None:
    """Check _repr_html_."""
    graph = GraphView()
    graph.edge("A", "B", "foo")
    assert "<title>B</title>" in graph._repr_html_()


@parametrize_edge_dir
def test_subgraph(is_directed, edge_dir) -> None:
    """Check that a subgraph is a GraphView with the settings of its parent."""
    graph = GraphView(is_directed, engine="neato", format="svg")
    with graph.subgraph(name="cluster_0") as subgraph:
        subgraph.node("A")
        subgraph.edge("A", "B")

    assert isinstance(subgraph, GraphView)
    assert subgraph.engine == "neato"
    assert subgraph.format == "svg"
    assert "subgraph cluster_0 {" in graph.source
    assert "A [color=black fillcolor=white" in graph.source
    assert f"A -> B [color=black dir={edge_dir} fontcolor=black" in graph.source


@parametrize_edge_dir
def test_copy(is_directed, edge_dir) -> None:
    """Check that a copy is a GraphView with the content and edge direction of the
    original."""
    graph = GraphView(is_directed)
    graph.node("C", fillcolor="yellow")
    graph.edge("A", "B")
    graph = graph.copy()
    graph.edge("D", "E")
    assert isinstance(graph, GraphView)
    assert "C [color=black fillcolor=yellow" in graph.source
    assert f"A -> B [color=black dir={edge_dir} fontcolor=black" in graph.source
    assert f"D -> E [color=black dir={edge_dir} fontcolor=black" in graph.source


@parametrize_edge_dir
def test_clear(is_directed, edge_dir) -> None:
    """Check that the edge direction survives clear()."""
    graph = GraphView(is_directed)
    graph.edge("A", "B")
    graph.clear()
    graph.edge("C", "D")
    assert f"C -> D [color=black dir={edge_dir} fontcolor=black" in graph.source


@pytest.mark.parametrize(
    ("is_directed", "edge_dir"),
    [(False, "forward"), (True, "none")],
    ids=["directed_in_undirected", "undirected_in_directed"],
)
def test_subgraph_with_other_direction(is_directed, edge_dir) -> None:
    """Check that a subgraph keeps its edge direction in a parent with the other
    one."""
    graph = GraphView(is_directed)
    subgraph = GraphView(not is_directed, name="cluster_0")
    subgraph.edge("A", "B")
    graph.subgraph(subgraph)
    assert f"A -> B [color=black dir={edge_dir} fontcolor=black" in graph.source


def test_visualize_show_default(monkeypatch) -> None:
    """Check that visualize does not show the graph by default."""
    render_kwargs = {}
    monkeypatch.setattr(
        GraphView, "render", lambda self, *args, **kwargs: render_kwargs.update(kwargs)
    )
    GraphView().visualize()
    assert render_kwargs["view"] is False


class _GraphViewWithOption(GraphView):
    """A graph view with an extra option, copied graphviz-style."""

    def __init__(self, option: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.option = option

    def _copy_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        return super()._copy_kwargs(option=self.option, **kwargs)


class _UndirectedGraphView(GraphView):
    """A graph view whose edge direction is fixed by its own __init__."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(is_directed=False, **kwargs)

    def _copy_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        # The __init__ of this subclass does not take is_directed.
        copy_kwargs = super()._copy_kwargs(**kwargs)
        del copy_kwargs["is_directed"]
        return copy_kwargs


_SUBCLASS_PARAMETERS: Final = pytest.mark.parametrize(
    ("cls", "kwargs", "attributes"),
    [
        (
            _GraphViewWithOption,
            {"option": "foo", "is_directed": False},
            {"option": "foo"},
        ),
        (_UndirectedGraphView, {}, {}),
    ],
    ids=["extra_option", "fixed_direction"],
)
"""Parametrize a test with a subclass, its arguments and its expected attributes."""


def _check_subclass(graph, new_graph, attributes) -> None:
    """Check that a new graph is an undirected instance of the class of the graph."""
    new_graph.edge("A", "B")
    assert type(new_graph) is type(graph)
    for name, value in attributes.items():
        assert getattr(new_graph, name) == value
    assert "A -> B [color=black dir=none fontcolor=black" in new_graph.source


@_SUBCLASS_PARAMETERS
def test_subclass_copy(cls, kwargs, attributes) -> None:
    """Check that a copy of a subclass keeps its class, attributes and edges."""
    graph = cls(**kwargs)
    _check_subclass(graph, graph.copy(), attributes)


@_SUBCLASS_PARAMETERS
def test_subclass_subgraph(cls, kwargs, attributes) -> None:
    """Check that a subgraph of a subclass keeps its class, attributes and edges."""
    graph = cls(**kwargs)
    with graph.subgraph(name="cluster_0") as subgraph:
        pass
    _check_subclass(graph, subgraph, attributes)


def test_visualize_filename(tmp_wd) -> None:
    """Check that visualize does not change the filename of the graph or its copies."""
    graph = GraphView(filename="foo.gv")
    graph.visualize(file_path="bar.png")
    assert graph.filename == "foo.gv"
    assert graph.copy().filename == "foo.gv"
