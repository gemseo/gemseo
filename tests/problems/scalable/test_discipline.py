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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import absolute_import, division, unicode_literals

from os.path import exists, join

import pytest
from future import standard_library

from gemseo.algos.design_space import DesignSpace
from gemseo.core.analytic_discipline import AnalyticDiscipline
from gemseo.core.doe_scenario import DOEScenario
from gemseo.problems.scalable.diagonal import ScalableDiagonalModel
from gemseo.problems.scalable.discipline import ScalableDiscipline

standard_library.install_aliases()


@pytest.fixture
def dataset():
    expressions_dict = {
        "y1": "sin(2*pi*x1)+cos(2*pi*x2)+x3",
        "y2": "sin(2*pi*x1)*cos(2*pi*x2)-x3",
    }
    disc = AnalyticDiscipline(name="sinus", expressions_dict=expressions_dict)
    disc.set_cache_policy(disc.MEMORY_FULL_CACHE)
    design_space = DesignSpace()
    design_space.add_variable("x1", l_b=0, u_b=1)
    design_space.add_variable("x2", l_b=0, u_b=1)
    design_space.add_variable("x3", l_b=0, u_b=1)
    doe = DOEScenario([disc], "DisciplinaryOpt", "y1", design_space)
    doe.execute({"algo": "fullfact", "n_samples": 10})
    return disc.cache


def test_constructor(dataset, tmp_path):
    ScalableDiscipline("ScalableDiagonalModel", dataset)
