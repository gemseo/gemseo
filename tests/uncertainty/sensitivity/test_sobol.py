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
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from os import remove

import pytest
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.api import create_discipline
from gemseo.uncertainty.sensitivity.sobol.analysis import SobolAnalysis
from numpy import pi
from numpy.testing import assert_equal


@pytest.fixture
def sobol() -> SobolAnalysis:
    """A Sobol' analysis."""
    discipline = create_discipline(
        "AnalyticDiscipline",
        expressions={"y": "sin(x1)+7*sin(x2)**2+0.1*x3**4*sin(x1)"},
        name="Ishigami",
    )
    space = ParameterSpace()
    for name in ["x1", "x2", "x3"]:
        space.add_random_variable(
            name, "OTUniformDistribution", minimum=-pi, maximum=pi
        )
    return SobolAnalysis([discipline], space, 100)


def test_sobol_algos():
    expected = sorted(["Saltelli", "Jansen", "MauntzKucherenko", "Martinez"])
    assert SobolAnalysis.AVAILABLE_ALGOS == expected


def test_sobol(sobol, tmp_path):
    varnames = ["x1", "x2", "x3"]
    assert sobol.main_method == sobol._FIRST_METHOD
    sobol.compute_indices()
    indices = sobol.indices
    first_order = indices["first"]
    total_order = indices["total"]
    main_indices = sobol.main_indices
    assert main_indices == first_order
    sobol.main_method = "total"
    assert sobol.main_method == sobol._TOTAL_METHOD
    assert sobol.main_indices == total_order
    assert {"y"} == set(first_order.keys())
    assert len(first_order["y"]) == 1
    assert {"y"} == set(total_order.keys())
    assert len(total_order["y"]) == 1
    for name in varnames:
        assert len(first_order["y"][0][name]) == 1
        assert len(total_order["y"][0][name]) == 1

    with pytest.raises(NotImplementedError):
        sobol.main_method = "foo"

    with pytest.raises(TypeError):
        sobol.compute_indices(algo="foo")

    intervals = sobol.get_intervals()
    assert {"y"} == set(intervals.keys())
    assert len(intervals["y"]) == 1
    for name in varnames:
        assert intervals["y"][0][name].shape == (2,)

    intervals = sobol.get_intervals(False)
    for name in varnames:
        assert intervals["y"][0][name].shape == (2,)

    sobol.plot("y", save=True, show=False, directory_path=tmp_path)
    assert (tmp_path / "sobol_analysis.png").exists()
    remove(str(tmp_path / "sobol_analysis.png"))
    sobol.plot("y", save=True, show=False, sort=False, directory_path=tmp_path)
    assert (tmp_path / "sobol_analysis.png").exists()
    remove(str(tmp_path / "sobol_analysis.png"))
    sobol.plot(
        "y",
        save=True,
        show=False,
        sort=False,
        sort_by_total=False,
        directory_path=tmp_path,
    )
    assert (tmp_path / "sobol_analysis.png").exists()
    remove(str(tmp_path / "sobol_analysis.png"))


def test_sobol_outputs(tmp_path):
    expressions = {
        "y1": "sin(x1)+7*sin(x2)**2+0.1*x3**4*sin(x1)",
        "y2": "sin(x2)+7*sin(x1)**2+0.1*x3**4*sin(x2)",
    }
    varnames = ["x1", "x2", "x3"]
    discipline = create_discipline(
        "AnalyticDiscipline", expressions=expressions, name="Ishigami2"
    )
    space = ParameterSpace()
    for variable in varnames:
        space.add_random_variable(
            variable, "OTUniformDistribution", minimum=-pi, maximum=pi
        )

    sobol = SobolAnalysis([discipline], space, 100)
    sobol.compute_indices()
    assert {"y1", "y2"} == set(sobol.main_indices.keys())

    sobol = SobolAnalysis([discipline], space, 100)
    sobol.compute_indices("y1")
    assert {"y1"} == set(sobol.main_indices.keys())


def test_save_load(sobol, tmp_wd):
    """Check saving and loading a SobolAnalysis."""
    sobol.save("foo.pkl")
    new_sobol = SobolAnalysis.load("foo.pkl")
    assert_equal(new_sobol.dataset.data, sobol.dataset.data)
    assert new_sobol.default_output == sobol.default_output
