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
from __future__ import annotations

from gemseo.machine_learning.linear_model_fitting.null_space import NullSpace
from gemseo.machine_learning.linear_model_fitting.null_space_settings import (
    NullSpace_Settings,
)
from gemseo.utils.testing.helpers import assert_exception


def test_null_space_error(input_data, output_data, snapshot):
    """Check the error raised when using NulSpace algorithm without extra data."""
    algo = NullSpace(NullSpace_Settings())
    with assert_exception(ValueError, snapshot):
        algo.fit(input_data, output_data)


def test_null_space(input_data, output_data):
    """Check the NulSpace algorithm."""
    algo = NullSpace(NullSpace_Settings())
    algo.fit(input_data, output_data, (input_data, output_data))
