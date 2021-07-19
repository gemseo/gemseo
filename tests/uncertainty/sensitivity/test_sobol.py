# -*- coding: utf-8 -*-
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
from __future__ import division, unicode_literals

from os import remove

import pytest
from numpy import pi

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.api import create_discipline
from gemseo.uncertainty.sensitivity.sobol.analysis import SobolAnalysis


def test_sobol_algos():
    expected = sorted(["Saltelli", "Jansen", "MauntzKucherenko", "Martinez"])
    assert SobolAnalysis.AVAILABLE_ALGOS == expected


def test_sobol(tmp_path):
    expressions = {"y": "sin(x1)+7*sin(x2)**2+0.1*x3**4*sin(x1)"}
    varnames = ["x1", "x2", "x3"]
    discipline = create_discipline(
        "AnalyticDiscipline", expressions_dict=expressions, name="Ishigami"
    )
    space = ParameterSpace()
    for name in varnames:
        space.add_random_variable(
            name, "OTUniformDistribution", minimum=-pi, maximum=pi
        )

    sobol = SobolAnalysis(discipline, space, 100)
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
    assert set(["y"]) == set(first_order.keys())
    assert len(first_order["y"]) == 1
    assert set(["y"]) == set(total_order.keys())
    assert len(total_order["y"]) == 1
    for name in varnames:
        assert len(first_order["y"][0][name]) == 1
        assert len(total_order["y"][0][name]) == 1

    with pytest.raises(NotImplementedError):
        sobol.main_method = "foo"

    with pytest.raises(TypeError):
        sobol.compute_indices(algo="foo")

    intervals = sobol.get_intervals()
    assert set(["y"]) == set(intervals.keys())
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
        "AnalyticDiscipline", expressions_dict=expressions, name="Ishigami2"
    )
    space = ParameterSpace()
    for variable in varnames:
        space.add_random_variable(
            variable, "OTUniformDistribution", minimum=-pi, maximum=pi
        )

    sobol = SobolAnalysis(discipline, space, 100)
    sobol.compute_indices()
    assert set(["y1", "y2"]) == set(sobol.main_indices.keys())

    sobol = SobolAnalysis(discipline, space, 100)
    sobol.compute_indices("y1")
    assert set(["y1"]) == set(sobol.main_indices.keys())
