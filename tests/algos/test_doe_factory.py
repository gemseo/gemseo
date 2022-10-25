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
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.algos.doe.lib_pydoe import PyDOE


@pytest.fixture
def factory():
    """The DOE factory."""
    return DOEFactory()


def test_is_available(factory):
    """Check that the method is_available works."""
    assert factory.is_available("fullfact")
    assert not factory.is_available("unknown_algo_name")


def test_algorithms(factory):
    """Check that the property algorithms works."""
    assert "fullfact" in factory.algorithms


def test_algo_names_to_libraries(factory):
    """Check that the property algo_names_to_libraries works."""
    assert factory.algo_names_to_libraries["fullfact"] == "PyDOE"


def test_libraries(factory):
    """Check that the property libraries works."""
    assert {"CustomDOE", "DiagonalDOE", "PyDOE"} <= set(factory.libraries)


def test_create_from_algo_name(factory):
    """Check that the method create works from an algorithm name."""
    lib = factory.create("fullfact")
    assert isinstance(lib, PyDOE)
    assert lib.algo_name == "fullfact"


def test_create_from_library_name(factory):
    """Check that the method create works from a DOE library name."""
    lib = factory.create("PyDOE")
    assert isinstance(lib, PyDOE)
    assert lib.algo_name is None


def test_create_from_unknown_name(factory):
    """Check that the method create raises an ImportError from an unknown name."""
    with pytest.raises(
        ImportError,
        match=(
            "No algorithm or library of algorithms named 'unknown_name' "
            "is available; available algorithms are .+"
        ),
    ):
        factory.create("unknown_name")
