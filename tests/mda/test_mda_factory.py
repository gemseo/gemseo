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
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from gemseo.mda.factory import mda_factory
from gemseo.mda.jacobi import MDAJacobi
from gemseo.mda.jacobi_settings import MDAJacobi_Settings


def test_create(sellar_with_2d_array, sellar_disciplines) -> None:
    """Test the factory create."""
    mda = mda_factory.create(
        "MDAJacobi", sellar_disciplines, settings=MDAJacobi_Settings(max_mda_iter=2)
    )
    assert isinstance(mda, MDAJacobi)


def test_is_available() -> None:
    avail = mda_factory.class_names
    assert len(avail) > 2

    for mda in avail:
        assert mda_factory.is_available(mda)
