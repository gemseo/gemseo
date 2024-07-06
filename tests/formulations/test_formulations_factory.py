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
from gemseo.formulations.factory import MDOFormulationFactory
from gemseo.formulations.mdf import MDF
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.sellar_system import SellarSystem
from gemseo.utils.testing.helpers import concretize_classes
from tests.formulations.not_mdo_formulations.formulation import NotMDOFormulationFactory
from tests.formulations.not_mdo_formulations.formulation_a import ANotMDOFormulation


@pytest.fixture
def factory(reset_factory) -> MDOFormulationFactory:
    """The factory of BaseMDOFormulation."""
    return MDOFormulationFactory()


def test_is_available(monkeypatch, factory) -> None:
    """Check the method is_available."""
    monkeypatch.setenv("GEMSEO_PATH", Path(__file__).parent / "not_mdo_formulations")
    assert factory.is_available("MDF")
    assert not factory.is_available("ANotMDOFormulation")


def test_create_with_wrong_formulation_name(factory) -> None:
    """Check that a BaseMDOFormulation cannot be instantiated with a wrong name."""
    with pytest.raises(
        ImportError,
        match=(
            "The class foo is not available; "
            "the available ones are: BiLevel, DisciplinaryOpt, IDF, MDF."
        ),
    ):
        factory.create("foo", None, None, None)


def test_create(factory) -> None:
    """Check the creation of a BaseMDOFormulation."""
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


def test_not_mdo_formulation() -> None:
    """Check the use of a factory of _BaseFormulation that is not a
    BaseMDOFormulation."""
    with concretize_classes(ANotMDOFormulation):
        factory = NotMDOFormulationFactory()
        assert factory.is_available("ANotMDOFormulation")
        assert not factory.is_available("MDF")
