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

import unittest
from os.path import dirname
from os.path import join

from gemseo.post.post_factory import PostFactory

DIRNAME = dirname(__file__)
POWER2 = join(DIRNAME, "power2_opt_pb.h5")


class TestPostFactory(unittest.TestCase):
    """"""

    def test_is_available(self):
        """"""
        factory = PostFactory()
        assert factory.is_available("OptHistoryView")
        assert not factory.is_available("TOTO")
        self.assertRaises(ImportError, factory.create, None, "toto")

    def test_post(self):
        available = PostFactory().posts
        assert "GradientSensitivity" in available
        assert "Correlations" in available

    def test_execute_from_hdf(self):
        PostFactory().execute(POWER2, "OptHistoryView", save=False)
