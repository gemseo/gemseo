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
"""Tests for Observables."""

from __future__ import annotations

import logging

from gemseo.core.functions.array_function import ArrayFunction
from gemseo.core.functions.collections.functions import Functions
from gemseo.core.functions.collections.observables import Observables


def test_functions():
    """Check that Observables is a subclass of Functions."""
    assert issubclass(Observables, Functions)


def test_f_types():
    """Check the authorized function types."""
    assert Observables._F_TYPES == (ArrayFunction.FunctionType.OBS,)


def test_format_cast(problem):
    """Check that the method format casts a function to observable."""
    function = Observables().format(ArrayFunction(lambda x: x, name="o"))
    assert function.f_type == function.FunctionType.OBS


def test_format_warn(problem, caplog):
    """Check that the method format warns and return None if already observed."""
    observables = Observables()
    observable = ArrayFunction(
        lambda x: x, name="o", f_type=ArrayFunction.FunctionType.OBS
    )
    assert observables.format(observable) is not None
    observables.append(observable)
    assert observables.format(observable) is None
    assert caplog.record_tuples[0] == (
        "gemseo.core.functions.collections.observables",
        logging.WARNING,
        'The optimization problem already observes "o".',
    )
