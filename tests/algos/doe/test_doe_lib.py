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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, print_function, unicode_literals

from unittest import TestCase

from future import standard_library

from gemseo import SOFTWARE_NAME
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.api import configure_logger
from gemseo.problems.analytical.power_2 import Power2
from gemseo.utils.testing_utils import IS_NT

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)


class Test_DOE_lib(TestCase):
    def test_fail_sample(self):
        problem = Power2(exception_error=True)
        factory = DOEFactory()
        if factory.is_available("PyDOE"):
            lib = factory.create("PyDOE")
            lib.execute(problem, "lhs", n_samples=4)

    def test_evaluate_samples(self):
        problem = Power2()
        factory = DOEFactory()
        if factory.is_available("PyDOE"):
            doe = factory.create("PyDOE")
            doe.execute(problem, "fullfact", n_samples=2, wait_time_between_samples=1)
            if not IS_NT:
                doe.execute(
                    problem,
                    "fullfact",
                    n_samples=2,
                    n_processes=2,
                    wait_time_between_samples=1,
                )
