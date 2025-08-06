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

import re

import pytest
from numpy import array
from numpy import nan

from gemseo.algos.problem_function import ProblemFunction
from gemseo.algos.stop_criteria import DesvarIsNan
from gemseo.algos.stop_criteria import FunctionIsNan


def test_check_function_output_includes_nan():
    """Check the error raised by check_function_output_includes_nan()."""
    with pytest.raises(
        DesvarIsNan, match=re.escape("Found a NaN in the input array [nan].")
    ):
        ProblemFunction.check_function_output_includes_nan(array([nan]))

    with pytest.raises(
        FunctionIsNan,
        match=re.escape(
            "Found a NaN in the output data of the function f "
            "evaluated at the input array [1.]."
        ),
    ):
        ProblemFunction.check_function_output_includes_nan(
            array([nan]), function_name="f", xu_vect=array([1.0])
        )


@pytest.mark.parametrize("value", [array(["some_string"]), array("some_string")])
def test_check_function_output_includes_nan_with_strings(value):
    """Check that strings are ignored in the test for NaN values."""
    ProblemFunction.check_function_output_includes_nan(value)
