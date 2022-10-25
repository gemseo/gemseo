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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from pathlib import Path

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.formulations.formulations_factory import MDOFormulationsFactory
from gemseo.formulations.mdf import MDF
from gemseo.problems.sellar.sellar import Sellar1
from gemseo.problems.sellar.sellar import Sellar2
from gemseo.problems.sellar.sellar import SellarSystem

from tests.formulations.not_mdo_formulations.formulation import NotMDOFormulationFactory


@pytest.fixture
def factory(reset_factory) -> MDOFormulationsFactory:
    """The factory of MDOFormulation."""
    return MDOFormulationsFactory()


def test_formulations(factory):
    """Check the property formulations."""
    assert "MDF" in factory.formulations


def test_is_available(monkeypatch, factory):
    """Check the method is_available."""
    monkeypatch.setenv("GEMSEO_PATH", Path(__file__).parent / "not_mdo_formulations")
    assert factory.is_available("MDF")
    assert not factory.is_available("ANotMDOFormulation")


def test_create_with_wrong_formulation_name(factory):
    """Check that a MDOFormulation cannot be instantiated with a wrong name."""
    with pytest.raises(
        ImportError,
        match=(
            "Class foo is not available; \n"
            "available ones are: BiLevel, DisciplinaryOpt, IDF, MDF."
        ),
    ):
        factory.create("foo", None, None, None)


def test_create(factory):
    """Check the creation of a MDOFormulation."""
    design_space = DesignSpace()
    design_space.add_variable("x_shared", 3)
    formulation = factory.create(
        "MDF", [Sellar1(), Sellar2(), SellarSystem()], "obj", design_space
    )
    assert isinstance(formulation, MDF)
    assert "x_shared" in formulation.design_space
    assert [d.name for d in formulation.disciplines] == [
        "Sellar1",
        "Sellar2",
        "SellarSystem",
    ]


def test_not_mdo_formulation():
    """Check the use of a factory of _BaseFormulation that is not a MDOFormulation."""
    factory = NotMDOFormulationFactory()
    assert factory.factory.is_available("ANotMDOFormulation")
    assert not factory.factory.is_available("MDF")
