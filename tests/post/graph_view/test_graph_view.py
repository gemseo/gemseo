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

import pytest
from gemseo.post._graph_view import GraphView


def test_graph_view(tmp_wd):
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
    graph.visualize(show=False, file_path=file_path, clean_up=False)
    file_path = file_path.with_suffix(".dot")
    assert (
        file_path.read_text().strip()
        == (Path(__file__).parent / file_path).read_text().strip()
    )
    assert file_path.exists()


@pytest.mark.parametrize("use_directed_edges", [False, True])
def test_use_directed_edges(tmp_wd, use_directed_edges):
    """Check the argument use_directed_edges passed at instantiation."""
    graph = GraphView(use_directed_edges)
    graph.edge("A", "B")
    file_path = Path(f"use_directed_edges_{str(use_directed_edges).lower()}.png")
    graph.visualize(show=False, file_path=file_path, clean_up=False)
    file_path = file_path.with_suffix(".dot")
    assert (
        file_path.read_text().strip()
        == (Path(__file__).parent / file_path).read_text().strip()
    )


@pytest.mark.parametrize("clean_up", [False, True])
def test_clean_up(tmp_wd, clean_up):
    """Check the argument clean_up."""
    GraphView().visualize(show=False, clean_up=clean_up)
    assert Path("graph_view.dot").exists() is not clean_up
