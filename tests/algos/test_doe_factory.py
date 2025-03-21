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
#                       initial documentation
#        :author: Remi Lafage
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#         Francois Gallard : refactoring for v1, May 2016
from __future__ import annotations

import pytest

from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.algos.doe.pydoe.pydoe import PyDOELibrary


@pytest.fixture
def factory():
    """The DOE factory."""
    return DOELibraryFactory()


def test_is_available(factory) -> None:
    """Check that the method is_available works."""
    assert factory.is_available("PYDOE_FULLFACT")
    assert not factory.is_available("unknown_algo_name")


def test_algorithms(factory) -> None:
    """Check that the property algorithms works."""
    assert "PYDOE_FULLFACT" in factory.algorithms


def test_algo_names_to_libraries(factory) -> None:
    """Check that the property algo_names_to_libraries works."""
    assert factory.algo_names_to_libraries["PYDOE_FULLFACT"] == "PyDOELibrary"


def test_libraries(factory) -> None:
    """Check that the property libraries works."""
    assert {"CustomDOE", "DiagonalDOE", "PyDOELibrary"} <= set(factory.libraries)


def test_create_from_algo_name(factory) -> None:
    """Check that the method create works algorithm name."""
    lib = factory.create("PYDOE_FULLFACT")
    assert isinstance(lib, PyDOELibrary)
    assert lib._algo_name == "PYDOE_FULLFACT"


def test_create_from_unknown_name(factory) -> None:
    """Check that the method create raises an ValueError from an unknown name."""
    with pytest.raises(
        ValueError,
        match=(
            r"No algorithm named unknown_name is available; available algorithms are .+"
        ),
    ):
        factory.create("unknown_name")
