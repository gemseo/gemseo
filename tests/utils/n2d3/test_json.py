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
"""Test the class N2JSON."""
from __future__ import annotations

from json import loads

import pytest
from gemseo.core.coupling_structure import DependencyGraph
from gemseo.core.discipline import MDODiscipline
from gemseo.utils.n2d3.n2_json import N2JSON
from numpy import ones


@pytest.fixture(scope="module")
def n2_json() -> N2JSON:
    """The N2JSON related to two strongly coupled disciplines and a weakly one."""
    description_list = [
        ("D1", ["y21"], ["y12"]),
        ("D2", ["y12"], ["y21"]),
        ("D3", ["y12", "y21"], ["obj"]),
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
    return N2JSON(DependencyGraph(disciplines))


@pytest.fixture(scope="module")
def expected_links(n2_json):
    """The expected links computed by the N2JSON.

    Args:
        n2_json (N2JSON): The N2JSON
            related to two strongly coupled disciplines and a weakly one.
    """
    return [
        {
            "source": 3,
            "target": 4,
            "value": 1,
            "description": n2_json._create_coupling_html(
                "D1", "D2", ["y12"], {"y12": "n/a", "y21": "n/a"}
            ),
        },
        {
            "source": 3,
            "target": 2,
            "value": 1,
            "description": n2_json._create_coupling_html(
                "D1", "D3", ["y12"], {"y12": "n/a", "y21": "n/a"}
            ),
        },
        {
            "source": 4,
            "target": 3,
            "value": 1,
            "description": n2_json._create_coupling_html(
                "D2", "D1", ["y21"], {"y12": "n/a", "y21": "n/a"}
            ),
        },
        {
            "source": 4,
            "target": 2,
            "value": 1,
            "description": n2_json._create_coupling_html(
                "D2", "D3", ["y21"], {"y12": "n/a", "y21": "n/a"}
            ),
        },
        {"source": 0, "target": 0, "value": 1, "description": ""},
        {"source": 1, "target": 1, "value": 1, "description": ""},
        {"source": 2, "target": 2, "value": 1, "description": ""},
        {"source": 3, "target": 3, "value": 1, "description": ""},
        {"source": 4, "target": 4, "value": 1, "description": ""},
    ]


@pytest.fixture(scope="module")
def expected_nodes(n2_json):
    """The expected nodes computed by the N2JSON.

    Args:
        n2_json (N2JSON): The N2JSON
            related to two strongly coupled disciplines and a weakly one.
    """
    disciplines = ["D1", "D2", "D3"]
    n_groups = 2
    children = [[4], [2, 3]]
    nodes = []
    desc = n2_json._create_group_html(0, disciplines, n_groups, children)
    nodes.append(
        {
            "name": n2_json._DEFAULT_GROUP_TEMPLATE.format(0),
            "is_group": True,
            "group": 0,
            "description": desc,
        }
    )
    desc = n2_json._create_group_html(1, disciplines, n_groups, children)
    nodes.append(
        {
            "name": n2_json._DEFAULT_GROUP_TEMPLATE.format(1),
            "is_group": True,
            "group": 1,
            "description": desc,
        }
    )

    disciplines = list(n2_json._graph.disciplines)
    for discipline in [disciplines[index] for index in [2, 0, 1]]:
        desc = n2_json._create_discipline_html(
            discipline,
            {"y12": "n/a", "y21": "n/a"},
        )
        nodes.append(
            {
                "name": discipline.name,
                "is_group": False,
                "group": 0 if discipline.name == "D3" else 1,
                "description": desc,
            }
        )
    return nodes


def test_generate_variables_html(n2_json):
    """Check the private static method generate_variables_html.

    Args:
        n2_json (N2JSON): The N2JSON
            related to two strongly coupled disciplines and a weakly one.
    """
    html = n2_json._create_variables_html(["a", "b"], {"a": 1, "b": 2})
    expected_html = (
        "<div align='center'>"
        "    <table>"
        "        <tr>"
        "            <td>a</td><td>(1)</td>"
        "        </tr>"
        "        <tr>"
        "            <td>b</td><td>(2)</td>"
        "        </tr>"
        "    </table>"
        "</div>"
    )
    assert html == expected_html


def test_generate_coupling_html(n2_json):
    """Check the private static method generate_coupling_html.

    Args:
        n2_json (N2JSON): The N2JSON
            related to two strongly coupled disciplines and a weakly one.
    """
    html = n2_json._create_coupling_html("A", "B", ["a", "b"], {"a": 1, "b": 2})
    expected_html = (
        "The coupling variables from <b>A</b> to <b>B</b>:"
        "<div align='center'>"
        "    <table>"
        "        <tr>"
        "            <td>a</td><td>(1)</td>"
        "        </tr>"
        "        <tr>"
        "            <td>b</td><td>(2)</td>"
        "        </tr>"
        "    </table>"
        "</div>"
    )
    assert html == expected_html


def test_generate_discipline_html(n2_json):
    """Check the private static method generate_discipline_html.

    Args:
        n2_json (N2JSON): The N2JSON
            related to two strongly coupled disciplines and a weakly one.
    """
    html = n2_json._create_discipline_html(
        next(n2_json._graph.disciplines), {"y12": 1, "y21": 2}
    )
    expected_html = (
        "The inputs of <b>D1</b>:"
        "<div align='center'>"
        "    <table>"
        "        <tr>"
        "            <td>y21</td><td>(2)</td>"
        "        </tr>"
        "    </table>"
        "</div>"
        "The outputs of <b>D1</b>:"
        "<div align='center'>"
        "    <table>"
        "        <tr>"
        "            <td>y12</td><td>(1)</td>"
        "        </tr>"
        "    </table>"
        "</div>"
    )

    assert html == expected_html


@pytest.mark.parametrize("group", [0, 1])
def test_generate_group_html(n2_json, group):
    """Check the private static method generate_group_html.

    Args:
        n2_json (N2JSON): The N2JSON
            related to two strongly coupled disciplines and a weakly one.
        group: The index of the group.
    """
    html = n2_json._create_group_html(
        group,
        ["D1", "D2", "D3"],
        2,
        [[4], [2, 3]],
    )
    if group == 1:
        expected_html = (
            "The disciplines of <b>Group 1</b>:"
            "<div align='center'>"
            "    <table>"
            "        <tr>"
            "            <td>D1</td>"
            "        </tr>"
            "        <tr>"
            "            <td>D2</td>"
            "        </tr>"
            "    </table>"
            "</div>"
        )
    else:
        expected_html = (
            "The disciplines of <b>Group 0</b>:"
            "<div align='center'>"
            "    <table>"
            "        <tr>"
            "            <td>D3</td>"
            "        </tr>"
            "    </table>"
            "</div>"
        )

    assert html == expected_html


def test_generate_groups_menu_html(n2_json):
    """Check the private static method generate_groups_menu_html.

    Args:
        n2_json (N2JSON): The N2JSON
            related to two strongly coupled disciplines and a weakly one.
    """
    html = n2_json._create_groups_menu_html(
        ["D1", "D2", "D3"], [[2], [0, 1]], ["Group 0", "Group 1"]
    )

    expected_html = (
        "    <ul class='collapsible'>"
        "        <li>"
        "           <div class='switch'>"
        "              <label>"
        "              <input type='checkbox' "
        "onclick='expand_collapse_all(json.groups.length,svg);' "
        "id='check_all'/>"
        "              <span class='lever'></span>"
        "              </label>All groups"
        "           </div>"
        "       </li>"
        "        <li>"
        "            <div class='collapsible-header'>"
        "               <span id='group_name_0' "
        "contenteditable='true' class='group' "
        "onblur='change_group_name(this,0);'>Group 0"
        "               </span>"
        "            </div>"
        "            <div class='collapsible-body'>"
        "               D1"
        "            </div>"
        "        </li>"
        "        <li>"
        "            <div class='collapsible-header'>"
        "               <div class='switch'>"
        "                   <label>"
        "                   <input type='checkbox' "
        "id='check_1' onclick='expand_collapse_group(1,svg)'/>"
        "                   <span class='lever'></span>"
        "                   </label>"
        "               </div>"
        "               <span id='group_name_1' contenteditable='true' "
        "class='group' onblur='change_group_name(this,1);'>Group 1"
        "               </span>"
        "            </div>"
        "            <div class='collapsible-body'>"
        "               D2,"
        "               D3"
        "            </div>"
        "        </li>"
        "    </ul>"
    )

    assert html == expected_html


def test_get_disciplines_names(n2_json):
    """Check the private static method get_disciplines_names.

    Args:
        n2_json (N2JSON): The N2JSON
            related to two strongly coupled disciplines and a weakly one.
    """
    assert n2_json._get_disciplines_names() == ["D3", "D1", "D2"]


def test_compute_variables_sizes(n2_json):
    """Check the private static method compute_variables_sizes.

    Args:
        n2_json (N2JSON): The N2JSON
            related to two strongly coupled disciplines and a weakly one.
    """
    assert n2_json._compute_variables_sizes() == {"y12": "n/a", "y21": "n/a"}


def test_compute_groups(n2_json):
    """Check the private static method compute_groups.

    Args:
        n2_json (N2JSON): The N2JSON
            related to two strongly coupled disciplines and a weakly one.
    """
    groups, n_groups, children = n2_json._compute_groups(["D1", "D2", "D3"])
    assert n_groups == 2
    assert children == [[2], [3, 4]]
    assert groups == {"D1": 1, "D2": 1, "D3": 0}


def test_create_nodes(n2_json, expected_nodes):
    """Check the private static method create_nodes.

    Args:
        n2_json (N2JSON): The N2JSON
            related to two strongly coupled disciplines and a weakly one.
        expected_nodes: The expected nodes.
    """
    disciplines = ["D3", "D1", "D2"]
    n_groups = 2
    children = [[2], [3, 4]]
    nodes, groups = n2_json._create_nodes(
        {"D1": 1, "D2": 1, "D3": 0},
        {"y12": "n/a", "y21": "n/a"},
        disciplines,
        n_groups,
        children,
    )

    assert groups == [
        n2_json._DEFAULT_WEAKLY_COUPLED_DISCIPLINES,
        n2_json._DEFAULT_GROUP_TEMPLATE.format(1),
    ]
    assert nodes == [expected_nodes[index] for index in [0, 1, 4, 2, 3]]


@pytest.mark.parametrize("name", [1, "foo"])
def test_default_group_template(name):
    """Test the application of the group template for a group index.

    Args:
        name: The name of the group.
    """
    assert N2JSON._DEFAULT_GROUP_TEMPLATE.format(name) == f"Group {name}"


def test_create_links(n2_json, expected_links):
    """Check the private static method create_links.

    Args:
        n2_json (N2JSON): The N2JSON
            related to two strongly coupled disciplines and a weakly one.
        expected_links (List[Dict[str,Union[int,str]]]): The expected links.
    """
    links = n2_json._create_links(
        n2_json._graph.get_disciplines_couplings(),
        5,
        {"y12": "n/a", "y21": "n/a"},
        ["D3", "D1", "D2"],
        2,
    )
    assert links == expected_links


def test_loads(n2_json, expected_links, expected_nodes):
    """Check that the JSON attribute is loaded correctly.

    Args:
        n2_json (N2JSON): The N2JSON
            related to two strongly coupled disciplines and a weakly one.
        expected_links (List[Dict[str,Union[int,str]]]): The expected links.
        expected_nodes: The expected nodes.
    """
    json = loads(str(n2_json))
    assert set(json.keys()) == {
        "nodes",
        "children",
        "links",
        "disciplines",
        "groups",
        "self_coupled_disciplines",
    }
    assert json["groups"] == [
        n2_json._DEFAULT_WEAKLY_COUPLED_DISCIPLINES,
        n2_json._DEFAULT_GROUP_TEMPLATE.format(1),
    ]
    assert json["children"] == [[2], [3, 4]]
    assert json["links"] == expected_links
    assert json["nodes"] == expected_nodes
    assert json["self_coupled_disciplines"] == []
