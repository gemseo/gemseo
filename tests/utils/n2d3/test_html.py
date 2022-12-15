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
"""Test the function create_n2_html."""
from __future__ import annotations

from filecmp import cmp
from pathlib import Path

import pytest
from gemseo.core.coupling_structure import DependencyGraph
from gemseo.core.discipline import MDODiscipline
from gemseo.utils.n2d3.n2_html import N2HTML
from numpy import ones


@pytest.fixture(scope="module")
def graph() -> DependencyGraph:
    """The graph related to two strongly coupled disciplines and a weakly one."""
    description_list = [
        ("D1", ["y21"], ["y12"]),
        ("D2", ["y12"], ["y21"]),
        ("D3", ["y12", "y21", "z"], ["obj", "z"]),
    ]
    disciplines = []
    data = ones(1)
    for desc in description_list:
        name = desc[0]
        input_d = {k: data for k in desc[1]}
        output_d = {k: data for k in desc[2]}
        disc = MDODiscipline(name)
        disc.input_grammar.update_from_data(input_d)
        disc.output_grammar.update_from_data(output_d)
        disciplines.append(disc)
    return DependencyGraph(disciplines)


def test_from_graph(graph, tmp_wd):
    """Check that the content of the HTML file is correct when created from a graph.

    Args:
        graph (DependencyGraph): The dependency graph used to create the HTML file.
        tmp_wd (Path): A temporary working directory.
    """
    file_path = "n2.html"
    N2HTML(file_path).from_graph(graph, self_coupled_disciplines=["D3"])
    assert cmp(file_path, str(Path(__file__).parent / "expected_from_graph.html"))


def test_from_json(tmp_wd):
    """Check that the content of the HTML file is correct when create from a JSON file.

    Args:
        tmp_wd (Path): A temporary working directory.
    """
    file_path = "n2.html"
    N2HTML(file_path).from_json(Path(__file__).parent / "n2.json")
    assert cmp(file_path, str(Path(__file__).parent / "expected_from_json.html"))
