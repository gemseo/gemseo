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
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

from future import standard_library

from gemseo import SOFTWARE_NAME
from gemseo.api import configure_logger
from gemseo.mda.jacobi import MDAJacobi
from gemseo.mda.mda_factory import MDAFactory
from gemseo.problems.sellar.sellar import Sellar1, Sellar2, SellarSystem

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)


class TestMDAFactory(unittest.TestCase):
    """Tests of MDA factory"""

    def test_jacobi_sobieski(self):
        """Test the execution of Jacobi on Sobieski"""
        disciplines = [Sellar1(), Sellar2(), SellarSystem()]
        mda = MDAFactory().create("MDAJacobi", disciplines, max_mda_iter=2)
        mda.execute()

        disciplines2 = [Sellar1(), Sellar2(), SellarSystem()]
        mda2 = MDAJacobi(disciplines2, max_mda_iter=2)
        mda2.execute()

        for k, v in mda2.local_data.items():
            assert (v == mda.local_data[k]).all()

    def test_available_mdas(self):
        factory = MDAFactory()
        avail = factory.mdas
        assert len(avail) > 2

        for mda in avail:
            assert factory.is_available(mda)
