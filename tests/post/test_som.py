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
#    INITIAL AUTHORS - API and implementation and/or documentation
#       :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, unicode_literals

import unittest
from os import remove
from os.path import exists, join

from future import standard_library

from gemseo import SOFTWARE_NAME
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.api import configure_logger
from gemseo.post.post_factory import PostFactory
from gemseo.problems.analytical.power_2 import Power2
from gemseo.third_party.junitxmlreq import link_to
from gemseo.utils.py23_compat import TemporaryDirectory

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)


class Test_SOM(unittest.TestCase):
    """ """

    @link_to("Req-VIZ-1", "Req-VIZ-1.1", "Req-VIZ-1.2", "Req-VIZ-2", "Req-VIZ-1.4")
    def test_som(self):
        with TemporaryDirectory() as outdir:
            problem = Power2()
            OptimizersFactory().execute(problem, "SLSQP")
            factory = PostFactory()
            for val in problem.database.values():
                val.pop("pow2")
            post = factory.execute(problem, "SOM", n_x=4, n_y=3, show=False, save=True)
            assert len(post.output_files) == 1
            for outf in post.output_files:
                assert exists(outf)
                remove(outf)

    def test_som_annotate(self):
        problem = Power2()
        OptimizersFactory().execute(problem, "SLSQP")
        factory = PostFactory()
        for val in problem.database.values():
            val.pop("pow2")
        post = factory.execute(
            problem, "SOM", n_x=4, n_y=3, show=False, save=True, annotate=True
        )
        assert len(post.output_files) == 1
        for outf in post.output_files:
            assert exists(outf)
            remove(outf)
