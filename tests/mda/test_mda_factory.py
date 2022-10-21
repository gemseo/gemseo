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

from gemseo.mda.jacobi import MDAJacobi
from gemseo.mda.mda_factory import MDAFactory


def test_jacobi_sobieski(sellar_disciplines):
    """Test the execution of Jacobi on Sobieski."""
    mda = MDAFactory().create("MDAJacobi", sellar_disciplines, max_mda_iter=2)
    mda.execute()

    mda2 = MDAJacobi(sellar_disciplines, max_mda_iter=2)
    mda2.execute()

    for k, v in mda2.local_data.items():
        assert (v == mda.local_data[k]).all()


def test_available_mdas():
    factory = MDAFactory()
    avail = factory.mdas
    assert len(avail) > 2

    for mda in avail:
        assert factory.is_available(mda)
