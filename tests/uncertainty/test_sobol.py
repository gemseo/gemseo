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

import pytest
from future import standard_library
from numpy import pi
from openturns import SaltelliSensitivityAlgorithm

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.api import create_discipline
from gemseo.uncertainty.sobol import SobolIndices

standard_library.install_aliases()


def test_sobol():
    expressions = {"y": "sin(x1)+7*sin(x2)**2+0.1*x3**4*sin(x1)"}
    discipline = create_discipline(
        "AnalyticDiscipline", expressions_dict=expressions, name="Ishigami"
    )
    expressions = {"z": "x1"}
    disciplines = [
        discipline,
        create_discipline(
            "AnalyticDiscipline", expressions_dict=expressions, name="x1"
        ),
    ]
    space = ParameterSpace()
    for variable in ["x1", "x2", "x3"]:
        space.add_random_variable(
            variable, "OTUniformDistribution", lower=-pi, upper=pi
        )

    sobol = SobolIndices(discipline, space, 100)
    result, first_order, total_order = sobol.get_indices()
    assert isinstance(result, SaltelliSensitivityAlgorithm)
    assert len(first_order["x1"]) == 1
    assert len(first_order["x2"]) == 1
    assert len(first_order["x3"]) == 1
    assert len(total_order["x1"]) == 1
    assert len(total_order["x2"]) == 1
    assert len(total_order["x3"]) == 1
    with pytest.raises(TypeError):
        sobol.get_indices("foo")
    sobol = SobolIndices(disciplines, space, 100)
