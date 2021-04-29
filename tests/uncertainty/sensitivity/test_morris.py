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
from __future__ import absolute_import, division, unicode_literals

from os import chdir
from os.path import exists

import pytest
from numpy import pi

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.api import create_discipline
from gemseo.uncertainty.sensitivity.morris.analysis import MorrisAnalysis


def test_morris(tmp_path):
    expressions = {"y": "sin(x1)+7*sin(x2)**2+0.1*x3**4*sin(x1)"}
    varnames = ["x1", "x2", "x3"]
    discipline = create_discipline(
        "AnalyticDiscipline", expressions_dict=expressions, name="Ishigami"
    )
    space = ParameterSpace()
    for variable in varnames:
        space.add_random_variable(
            variable, "OTUniformDistribution", minimum=-pi, maximum=pi
        )

    morris = MorrisAnalysis(discipline, space, n_samples=None, n_replicates=5)
    morris.compute_indices()
    indices = morris.indices
    assert "mu" in indices
    assert "mu_star" in indices
    assert "sigma" in indices
    assert set(["y"]) == set(morris.main_indices.keys())
    assert set(["y"]) == set(indices["mu"].keys())
    assert set(["y"]) == set(indices["mu_star"].keys())
    assert set(["y"]) == set(indices["sigma"].keys())
    assert len(morris.main_indices["y"]) == 1
    assert len(indices["mu"]) == 1
    assert len(indices["mu_star"]) == 1
    assert len(indices["sigma"]) == 1
    for variable in varnames:
        assert variable in morris.main_indices["y"][0]
        assert variable in indices["mu"]["y"][0]
        assert variable in indices["mu_star"]["y"][0]
        assert variable in indices["sigma"]["y"][0]
        assert indices["sigma"]["y"][0][variable] >= 0
        assert indices["mu_star"]["y"][0][variable] >= indices["mu"]["y"][0][variable]

    assert morris.main_indices == indices["mu_star"]

    chdir(str(tmp_path))
    morris.plot("y", save=True, show=False)
    assert exists("morris_analysis.pdf")
    assert isinstance(morris.sort_parameters("y"), list)
    assert set(morris.sort_parameters("y")) == set(varnames)


def test_morris_outputs(tmp_path):
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

    morris = MorrisAnalysis(discipline, space, n_samples=None, n_replicates=5)
    morris.compute_indices()
    assert set(["y1", "y2"]) == set(morris.main_indices.keys())

    morris = MorrisAnalysis(discipline, space, n_samples=None, n_replicates=5)
    morris.compute_indices("y1")
    assert set(["y1"]) == set(morris.main_indices.keys())


def test_morris_with_bad_input_dimension():
    expressions = {"y": "x1+x2"}
    discipline = create_discipline("AnalyticDiscipline", expressions_dict=expressions)
    space = ParameterSpace()
    space.add_random_variable(
        "x1", "OTUniformDistribution", minimum=-pi, maximum=pi, size=2
    )
    space.add_random_variable("x2", "OTUniformDistribution", minimum=-pi, maximum=pi)
    with pytest.raises(ValueError):
        MorrisAnalysis(discipline, space, n_samples=None, n_replicates=100)


def test_morris_with_bad_step():
    expressions = {"y": "x1+x2"}
    discipline = create_discipline("AnalyticDiscipline", expressions_dict=expressions)
    space = ParameterSpace()
    space.add_random_variable("x1", "OTUniformDistribution", minimum=-pi, maximum=pi)
    space.add_random_variable("x2", "OTUniformDistribution", minimum=-pi, maximum=pi)
    with pytest.raises(ValueError):
        MorrisAnalysis(discipline, space, n_samples=None, n_replicates=100, step=0.0)
    with pytest.raises(ValueError):
        MorrisAnalysis(discipline, space, n_samples=None, n_replicates=100, step=0.5)


def test_morris_with_nsamples():
    expressions = {"y": "x1+x2"}
    discipline = create_discipline("AnalyticDiscipline", expressions_dict=expressions)
    space = ParameterSpace()
    space.add_random_variable("x1", "OTUniformDistribution", minimum=-pi, maximum=pi)
    space.add_random_variable("x2", "OTUniformDistribution", minimum=-pi, maximum=pi)
    morris = MorrisAnalysis(discipline, space, n_samples=7)
    assert morris.n_replicates == 2
