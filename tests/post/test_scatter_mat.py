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

from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from os import remove
from os.path import dirname, exists, join

from future import standard_library

from gemseo import SOFTWARE_NAME
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.api import configure_logger
from gemseo.post.post_factory import PostFactory
from gemseo.problems.analytical.power_2 import Power2
from gemseo.third_party.junitxmlreq import link_to

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)
POWER2 = join(dirname(__file__), "power2_opt_pb.h5")


class Test_ScatterPlotMatrix(unittest.TestCase):
    """ """

    @link_to(
        "Req-VIZ-1",
        "Req-VIZ-1.1",
        "Req-VIZ-1.2",
        "Req-VIZ-2",
        "Req-VIZ-1.8",
        "Req-VIZ-4",
    )
    def test_scatter(self):
        """ """
        factory = PostFactory()
        if factory.is_available("ScatterPlotMatrix"):
            problem = Power2()
            OptimizersFactory().execute(problem, "SLSQP")
            post = factory.execute(
                problem,
                "ScatterPlotMatrix",
                save=True,
                file_path="scatter1",
                variables_list=problem.get_all_functions_names(),
            )
            assert len(post.output_files) == 1
            for outf in post.output_files:
                assert exists(outf)
                remove(outf)

    @link_to(
        "Req-VIZ-1",
        "Req-VIZ-1.1",
        "Req-VIZ-1.2",
        "Req-VIZ-2",
        "Req-VIZ-5",
        "Req-VIZ-1.8",
        "Req-VIZ-4",
    )
    def test_scatter_load(self):
        """ """
        factory = PostFactory()
        if factory.is_available("ScatterPlotMatrix"):
            problem = OptimizationProblem.import_hdf(POWER2)
            post = factory.execute(
                problem,
                "ScatterPlotMatrix",
                save=True,
                file_path="scatter2",
                variables_list=problem.get_all_functions_names(),
            )
            assert len(post.output_files) == 1
            for outf in post.output_files:
                assert exists(outf)
                remove(outf)

            self.assertRaises(
                Exception,
                factory.execute,
                problem,
                "ScatterPlotMatrix",
                save=True,
                variables_list=["I dont exist"],
            )

            post = factory.execute(
                problem, "ScatterPlotMatrix", save=True, variables_list=[]
            )
            for outf in post.output_files:
                assert exists(outf)
                remove(outf)
