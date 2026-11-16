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
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from graphviz import Digraph

from gemseo.core.dependency_graph import DependencyGraph
from gemseo.core.discipline import Discipline
from gemseo.discipline.analytic import AnalyticDiscipline
from gemseo.problem.mdo.sellar.sellar_1 import Sellar1
from gemseo.problem.mdo.sellar.sellar_2 import Sellar2
from gemseo.problem.mdo.sellar.sellar_system import SellarSystem
from gemseo.problem.mdo.sobieski.discipline import SobieskiAerodynamics
from gemseo.problem.mdo.sobieski.discipline import SobieskiMission
from gemseo.problem.mdo.sobieski.discipline import SobieskiPropulsion
from gemseo.problem.mdo.sobieski.discipline import SobieskiStructure
from gemseo.util.testing.disciplines_creator import create_disciplines_from_desc
from gemseo.util.testing.helper import assert_exception

data_path = Path(__file__).absolute().parent / "data" / "dependency-graph"

parametrize_render_method = pytest.mark.parametrize(
    "method_name", ["render_full_graph", "render_condensed_graph"]
)

disc_descriptions = {
    "3-weak": {
        "A": (["x"], ["a"]),
        "B": (["x", "a"], ["b"]),
        "C": (["x", "a"], ["c"]),
    },
    "4-weak": {
        "A": (["x"], ["a"]),
        "B": (["x", "a"], ["b"]),
        "C": (["x", "a"], ["c"]),
        "D": (["b", "c"], ["d"]),
    },
    "5": {
        "A": (["b"], ["a", "c"]),
        "B": (["a"], ["b"]),
        "C": (["c", "e"], ["d"]),
        "D": (["d"], ["e", "f"]),
        "E": (["f"], []),
    },
    "16": {
        "A": (["a"], ["b"]),
        "B": (["c"], ["a", "n"]),
        "C": (["b", "d"], ["c", "e"]),
        "D": (["f"], ["d", "g"]),
        "E": (["e"], ["f", "h", "o"]),
        "F": (["g", "j"], ["i"]),
        "G": (["i", "h"], ["k", "l"]),
        "H": (["k", "m"], ["j"]),
        "I": (["l"], ["m", "w"]),
        "J": (["n", "o"], ["p", "q"]),
        "K": (["y"], ["x"]),
        "L": (["w", "x"], ["y", "z"]),
        "M": (["p", "s"], ["r"]),
        "N": (["r"], ["t", "u"]),
        "O": (["q", "t"], ["s", "v"]),
        "P": (["u", "v", "z"], ["obj"]),
    },
    # to avoid memory and cpu overhead, only instantiate the objects when they
    # are needed, not here
    "sellar": (Sellar1, Sellar2, SellarSystem),
    "sobieski": (
        SobieskiAerodynamics,
        SobieskiStructure,
        SobieskiPropulsion,
        SobieskiMission,
    ),
}


@pytest.fixture(params=disc_descriptions.items(), ids=disc_descriptions.keys())
def name_and_graph(request):
    """Return the name and graph from the full description."""
    name, disc_desc = request.param
    disciplines = create_disciplines_from_desc(disc_desc)
    graph = DependencyGraph(disciplines)
    return name, graph


def assert_dot_file(file_name) -> None:
    """Assert the contents of the DOT source file given a pdf file.

    The DOT source file has the suffix `.gv`
    and its reference the suffix `.dot`.
    """
    file_path = Path(file_name)
    # strip because some reference files are stripped by our pre-commit hooks
    assert (
        file_path.with_suffix(".gv").read_text().strip()
        == (data_path / file_path.with_suffix(".dot").name).read_text().strip()
    )


def test_render_full_graph(tmp_wd, name_and_graph) -> None:
    """Test writing the full graph to disk.

    This also checks the expected contents of a graph.
    """
    name, graph = name_and_graph
    file_name = f"{name}.full_graph.pdf"
    graph.render_full_graph(file_name, clean_up=False)
    assert_dot_file(file_name)


@parametrize_render_method
@pytest.mark.parametrize("file_path", ["", Path()], ids=["str", "path"])
def test_render_without_saving(
    tmp_wd, sellar_dependency_graph, method_name, file_path
) -> None:
    """Test rendering the graph without saving."""
    digraph = getattr(sellar_dependency_graph, method_name)(file_path=file_path)
    assert isinstance(digraph, Digraph)
    assert not os.listdir(tmp_wd)


@pytest.mark.parametrize(
    ("show_edge_labels", "tooltip_key"),
    [(False, "edgetooltip="), (True, "labeltooltip=")],
    ids=["hide_edge_labels", "show_edge_labels"],
)
def test_render_full_graph_show_edge_labels(
    sellar_dependency_graph, show_edge_labels, tooltip_key
) -> None:
    """Check that show_edge_labels controls the labels and tooltips of the edges."""
    source = sellar_dependency_graph.render_full_graph(
        file_path="", show_edge_labels=show_edge_labels
    ).source
    # The nodes are always labelled, the edges only when show_edge_labels.
    assert bool(re.search(r"-> \w+ \[label=", source)) is show_edge_labels
    # The tooltip is on the label if any, otherwise on the edge.
    assert tooltip_key in source


@pytest.mark.parametrize(
    "show_edge_labels", [False, True], ids=["hide_edge_labels", "show_edge_labels"]
)
def test_render_condensed_graph_show_edge_labels(
    sellar_dependency_graph, show_edge_labels
) -> None:
    """Check that show_edge_labels controls the labels of the edges, without tooltip."""
    source = sellar_dependency_graph.render_condensed_graph(
        file_path="", show_edge_labels=show_edge_labels
    ).source
    # The nodes are always labelled, the edges only when show_edge_labels.
    assert bool(re.search(r"-> \w+ \[label=", source)) is show_edge_labels
    assert "labeltooltip=" not in source
    assert "edgetooltip=" not in source


def test_render_condensed_graph(tmp_wd, name_and_graph) -> None:
    """Test writing the condensed graph to disk.

    This also checks the expected contents of a graph.
    """
    name, graph = name_and_graph
    file_name = f"{name}.condensed_graph.pdf"
    graph.render_condensed_graph(file_name, clean_up=False)
    assert_dot_file(file_name)


def test_couplings(tmp_wd, name_and_graph) -> None:
    """Test the couplings against references stored in json files."""
    name, graph = name_and_graph
    couplings = graph.get_disciplines_couplings()
    file_path = Path(f"{name}.couplings.json")

    # dump a json of the couplings where the Discipline objects have been converted to
    # a string to allow dumping and comparison

    json.dump(
        couplings,
        file_path.open("w", encoding="utf-8"),
        cls=DisciplineEncoder,
        indent=4,
    )

    # read back the just created json and the reference
    couplings = json.load(file_path.open())
    ref_couplings = json.load((data_path / file_path).open())

    assert couplings == ref_couplings


class DisciplineEncoder(json.JSONEncoder):
    """JSON encoder that handles discipline objects.

    Discipline objects are stringyfied.
    """

    def default(self, o):
        if isinstance(o, Discipline):
            return str(o)
        return super().default(o)


@pytest.fixture(scope="module")
def graph_with_self_coupling() -> DependencyGraph:
    """Dependency graph with a self-coupled discipline."""
    return DependencyGraph([
        AnalyticDiscipline({"y0": "x0+y10+y2"}, name="D0"),
        AnalyticDiscipline({"y10": "x0+x1+y2", "y11": "x0-x1+2*y11"}, name="D1"),
        AnalyticDiscipline({"y2": "x0+x2+y10"}, name="D2"),
    ])


@pytest.mark.parametrize(
    ("file_path", "method"),
    [
        ("full_coupling_graph.pdf", "render_full_graph"),
        ("condensed_coupling_graph.pdf", "render_condensed_graph"),
    ],
)
def test_coupling_structure_plot(
    tmp_wd, graph_with_self_coupling, file_path, method
) -> None:
    """Check the rendering of the coupling graph with a self-coupled discipline."""
    getattr(graph_with_self_coupling, method)(file_path, clean_up=False)
    assert_dot_file(Path(file_path))


def test_no_graphviz(caplog, graph_with_self_coupling) -> None:
    """Check the message logged when graphviz is missing."""
    with patch("gemseo.core.dependency_graph.graphviz_is_missing", True):
        assert graph_with_self_coupling.render_full_graph("graph.pdf") is None
        _, log_level, log_message = caplog.record_tuples[0]
        assert log_level == logging.WARNING
        assert log_message == ("Cannot render graph: graphviz is not installed.")


def test_homonymous_disciplines(tmp_wd) -> None:
    """Check the rendering of the coupling graph with homonymous disciplines.

    Each discipline has its own node
    and the homonymous leaves have distinct hidden head nodes.
    """
    expected = """digraph {
	node [fillcolor=white style=filled]
	d0 [label=AnalyticDiscipline]
	d1 [label=foo]
	d2 [label=AnalyticDiscipline]
	d3 [label=AnalyticDiscipline]
	d0 -> d1 [label=b labeltooltip="Global name, Name in discipline 'AnalyticDiscipline', Name in discipline 'foo'

b, b, b"]
	d1 -> d2 [label=c labeltooltip="Global name, Name in discipline 'foo', Name in discipline 'AnalyticDiscipline'

c, c, c"]
	d1 -> d3 [label=c labeltooltip="Global name, Name in discipline 'foo', Name in discipline 'AnalyticDiscipline'

c, c, c"]
	d2 -> h2 [label=d labeltooltip="Global name, Name in discipline 'AnalyticDiscipline'

d, d"]
	h2 [style=invis]
	d3 -> h3 [label=e labeltooltip="Global name, Name in discipline 'AnalyticDiscipline'

e, e"]
	h3 [style=invis]
}"""  # noqa: E501
    graph = DependencyGraph([
        AnalyticDiscipline({"b": "a"}),
        AnalyticDiscipline({"c": "b"}, name="foo"),
        AnalyticDiscipline({"d": "c"}),
        AnalyticDiscipline({"e": "c"}),
    ])
    file_path = Path("homonymous_disciplines.pdf")
    graph.render_full_graph(file_path, clean_up=False)
    assert file_path.with_suffix(".gv").read_text().strip() == expected


@parametrize_render_method
@pytest.mark.parametrize(
    "file_path",
    ["coupling_graph", "coupling_graph.v2", "."],
    ids=["no_suffix", "unknown_suffix", "dot"],
)
def test_render_with_wrong_suffix(
    tmp_wd, graph_with_self_coupling, method_name, file_path, snapshot
):
    """Check the error raised before drawing when the suffix is not a format."""
    with (
        patch("gemseo.post.graph.draw_coupling_graph") as draw_coupling_graph,
        assert_exception(ValueError, snapshot),
    ):
        getattr(graph_with_self_coupling, method_name)(file_path)

    draw_coupling_graph.assert_not_called()
    assert not os.listdir(tmp_wd)


@parametrize_render_method
def test_render_parent_directory(tmp_wd, graph_with_self_coupling, method_name):
    """Check the error raised when the file path is the parent directory."""
    # The example of the message uses the separator of the platform.
    example = Path("..") / "coupling_graph.pdf"
    msg = (
        "The file path '..' has no extension; "
        f"add one to set the format, e.g. '{example}'."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        getattr(graph_with_self_coupling, method_name)("..")


def test_create_condensed_graph() -> None:
    """Check the nodes and the attributes of the condensed graph."""
    # Reverse the disciplines to check that the order of the members
    # is that of the disciplines passed at instantiation, not alphabetical.
    disciplines = create_disciplines_from_desc(disc_descriptions["5"])[::-1]
    e, d, c, b, a = disciplines
    condensed_graph = DependencyGraph(disciplines).create_condensed_graph()
    assert set(condensed_graph.nodes) == {0, 1, 2}
    members = [condensed_graph.nodes[node]["members"] for node in condensed_graph]
    assert sorted(members, key=lambda m: m[0].name) == [[b, a], [d, c], [e]]
    mapping = condensed_graph.graph["mapping"]
    assert mapping == {
        discipline: node
        for node in condensed_graph
        for discipline in condensed_graph.nodes[node]["members"]
    }
    # The edges of the condensed graph bear the names of the coupling variables
    # between the disciplines of their components.
    assert {
        (tail, head): names
        for tail, head, names in condensed_graph.edges(data=DependencyGraph.io)
    } == {(mapping[a], mapping[c]): {"c"}, (mapping[c], mapping[e]): {"f"}}
