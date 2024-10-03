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

from numpy import pi

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.uncertainty.sensitivity.factory import SensitivityAnalysisFactory


def test_not_available() -> None:
    factory = SensitivityAnalysisFactory()
    assert factory.is_available("MorrisAnalysis")
    assert not factory.is_available("FooAnalysis")


def test_create() -> None:
    discipline = AnalyticDiscipline(
        {"y": "sin(x1)+7*sin(x2)**2+0.1*x3**4*sin(x1)"}, name="Ishigami"
    )

    space = ParameterSpace()
    for variable in ["x1", "x2", "x3"]:
        space.add_random_variable(
            variable, "OTUniformDistribution", minimum=-pi, maximum=pi
        )
    factory = SensitivityAnalysisFactory()
    analysis = factory.create("MorrisAnalysis")
    samples = analysis.compute_samples(
        (discipline,), space, n_replicates=5, n_samples=0
    )
    other_analysis = factory.create("MorrisAnalysis", samples=samples)
    assert id(analysis.dataset) == id(other_analysis.dataset) == id(samples)
