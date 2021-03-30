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
# INITIAL AUTHORS - initial API and implementation and/or
#                   initial documentation
#        :author:  Francois
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

from future import standard_library
from numpy import array

from gemseo import SOFTWARE_NAME
from gemseo.api import configure_logger
from gemseo.core.analytic_discipline import AnalyticDiscipline
from gemseo.core.mdo_scenario import MDOScenario

standard_library.install_aliases()
configure_logger(SOFTWARE_NAME)


class Test_AnalyticDiscipline(unittest.TestCase):
    """ """

    def test_basic(self):
        expr_dict = {"y_1": "2*x**2", "y_2": "3*x**2+5+z**3"}
        disc = AnalyticDiscipline("analytic", expr_dict)
        input_data = {"x": array([2.0]), "z": array([3.0])}
        disc.check_jacobian(
            input_data, derr_approx=disc.FINITE_DIFFERENCES, step=1e-5, threshold=1e-3
        )

    def test_fail(self):
        expr_dict = {"y": MDOScenario}
        self.assertRaises(TypeError, AnalyticDiscipline, "analytic", expr_dict)
        self.assertRaises(ValueError, AnalyticDiscipline, "analytic", None)

        expr_dict = {"y": "log(x)"}
        disc = AnalyticDiscipline("analytic", expr_dict)
        input_data = {"x": array([0.0])}
        self.assertRaises(TypeError, disc.execute, input_data)
