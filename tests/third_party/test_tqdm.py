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
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

from future import standard_library

from gemseo import SOFTWARE_NAME
from gemseo.api import configure_logger
from gemseo.third_party.tqdm import get_printable_rate

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)


class TestDataConversion(unittest.TestCase):
    def test_units_rate(self):
        rate, unit = get_printable_rate(2, 24 * 3600)
        assert rate == 2.0
        assert unit == "iters/day"

        rate, unit = get_printable_rate(2, 3600)
        assert unit == "iters/hour"

        rate, unit = get_printable_rate(2, 60)
        assert unit == "iters/min"

        rate, unit = get_printable_rate(2, 1)
        assert unit == "iters/sec"

        rate, unit = get_printable_rate(1, 1.5 * 24 * 3600)
        assert unit == "iters/day"

        rate, unit = get_printable_rate(2, 24 * 3600)
        assert unit == "iters/day"

        rate, unit = get_printable_rate(10.0, 1)
        assert unit == "iters/sec"

        rate, unit = get_printable_rate(0.1, 1)
        assert unit == "iters/min"

        rate, unit = get_printable_rate(0.1, 60)
        assert unit == "iters/hour"

        rate, unit = get_printable_rate(0.1, 60 * 60)
        assert unit == "iters/day"
