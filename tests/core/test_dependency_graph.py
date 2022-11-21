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
from pathlib import Path
from unittest.mock import patch

import pytest
from gemseo.core.dependency_graph import DependencyGraph
from gemseo.core.discipline import MDODiscipline
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.problems.sellar.sellar import Sellar1
from gemseo.problems.sellar.sellar import Sellar2
from gemseo.problems.sellar.sellar import SellarSystem
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure
from numpy import ones

DATA_PATH = Path(__file__).absolute().parent / "data" / "dependency-graph"

DISC_DESCRIPTIONS = {
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


def create_disciplines_from_desc(disc_desc):
    """Return the disciplines from their descriptions.

    Args:
        disc_desc: The disc_desc of a discipline, either a list of classes or a dict.
    """
    if isinstance(disc_desc, tuple):
        # these are disciplines classes
        return [cls() for cls in disc_desc]

    disciplines = []
    data = ones(1)

    disc_desc_items = disc_desc.items()

    for name, io_names in disc_desc_items:
        disc = MDODiscipline(name)
        input_d = {k: data for k in io_names[0]}
        disc.input_grammar.update_from_data(input_d)
        output_d = {k: data for k in io_names[1]}
        disc.output_grammar.update_from_data(output_d)
        disciplines += [disc]

    return disciplines


@pytest.fixture(params=DISC_DESCRIPTIONS.items(), ids=DISC_DESCRIPTIONS.keys())
def name_and_graph(request):
    """Return the name and graph from the full description."""
    name, disc_desc = request.param
    disciplines = create_disciplines_from_desc(disc_desc)
    graph = DependencyGraph(disciplines)
    return name, graph


def assert_dot_file(file_name):
    """Assert the contents of the dot file given a pdf file."""
    assert_file(Path(file_name).with_suffix(".dot"))


def assert_file(file_path: Path):
    """Assert the contents of the file against its reference."""
    # strip because some reference files are stripped by our pre-commit hooks
    assert (
        file_path.read_text().strip()
        == (DATA_PATH / file_path.name).read_text().strip()
    )


def test_write_full_graph(tmp_wd, name_and_graph):
    """Test writing the full graph to disk.

    This also checks the expected contents of a graph.
    """
    name, graph = name_and_graph
    file_name = f"{name}.full_graph.pdf"
    graph.write_full_graph(file_name)
    assert_dot_file(file_name)


def test_write_condensed_graph(tmp_wd, name_and_graph):
    """Test writing the condensed graph to disk.

    This also checks the expected contents of a graph.
    """
    name, graph = name_and_graph
    file_name = f"{name}.condensed_graph.pdf"
    graph.write_condensed_graph(file_name)
    assert_dot_file(file_name)


def test_couplings(tmp_wd, name_and_graph):
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
    ref_couplings = json.load((DATA_PATH / file_path).open())

    assert couplings == ref_couplings


class DisciplineEncoder(json.JSONEncoder):
    """JSON encoder that handles discipline objects.

    MDODiscipline objects are stringyfied.
    """

    def default(self, o):
        if isinstance(o, MDODiscipline):
            return str(o)
        return super().default(o)


@pytest.fixture(scope="module")
def graph_with_self_coupling() -> DependencyGraph:
    """Dependency graph with a self-coupled discipline."""
    return DependencyGraph(
        [
            AnalyticDiscipline({"y0": "x0+y10+y2"}, name="D0"),
            AnalyticDiscipline({"y10": "x0+x1+y2", "y11": "x0-x1+2*y11"}, name="D1"),
            AnalyticDiscipline({"y2": "x0+x2+y10"}, name="D2"),
        ]
    )


@pytest.mark.parametrize(
    "file_path,method",
    [
        ("full_coupling_graph.pdf", "write_full_graph"),
        ("condensed_coupling_graph.pdf", "write_condensed_graph"),
    ],
)
def test_coupling_structure_plot(tmp_wd, graph_with_self_coupling, file_path, method):
    """Check the rendering of the coupling graph with a self-coupled discipline."""
    getattr(graph_with_self_coupling, method)(file_path)
    assert_dot_file(Path(file_path))


def test_no_graphviz(caplog, graph_with_self_coupling):
    """Check the message logged when graphviz is missing."""
    with patch("gemseo.core.dependency_graph.GraphView", None):
        assert graph_with_self_coupling.write_full_graph("graph.pdf") is None
        _, log_level, log_message = caplog.record_tuples[0]
        assert log_level == logging.WARNING
        assert log_message == (
            "Cannot write graph: "
            "GraphView cannot be imported because graphviz is not installed."
        )
