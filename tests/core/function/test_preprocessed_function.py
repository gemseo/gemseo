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

import warnings

import pytest
from numpy import array
from numpy import full
from numpy import nan
from numpy.testing import assert_allclose

from gemseo.core.function.preprocessed_function import PreprocessedFunction
from gemseo.optimization.termination_criteria import DesvarIsNan
from gemseo.optimization.termination_criteria import FunctionIsNan
from gemseo.problem.optimization.power_2 import Power2
from gemseo.util.testing.helper import assert_exception


def test_check_function_output_includes_nan(snapshot):
    """Check the error raised by check_function_output_includes_nan()."""
    with assert_exception(DesvarIsNan, snapshot):
        PreprocessedFunction.check_function_output_includes_nan(array([nan]))

    with assert_exception(FunctionIsNan, snapshot):
        PreprocessedFunction.check_function_output_includes_nan(
            array([nan]), function_name="f", xu_vect=array([1.0])
        )


@pytest.mark.parametrize("value", [array(["some_string"]), array("some_string")])
def test_check_function_output_includes_nan_with_strings(value):
    """Check that strings are ignored in the test for NaN values."""
    PreprocessedFunction.check_function_output_includes_nan(value)


def test_unpickle_pre_refactor_denormalize_attributes():
    """Check that a PreprocessedFunction pickled before the denormalization renaming
    loads.

    Pre-refactor pickles store the ``_unnormalize_vect`` / ``_unnormalize_grad``
    attributes; ``__setstate__`` must remap them to ``_denormalize_vect`` /
    ``_denormalize_grad`` and rebind them to the renamed design-space methods,
    otherwise the first normalized evaluation raises ``AttributeError``.
    """
    problem = Power2()
    problem.preprocess_functions()
    function = problem.objective
    x_normalized = full(problem.design_space.dimension, 0.3)
    expected = function.evaluate(x_normalized)

    # Simulate a pre-refactor pickle: the modern state is downgraded to the old
    # attribute names bound to the deprecated design-space aliases.
    state = function.__getstate__()
    design_space = state.pop("_denormalize_vect").__self__
    del state["_denormalize_grad"]
    state["_unnormalize_vect"] = design_space.unnormalize_vect
    state["_unnormalize_grad"] = design_space.unnormalize_grad

    restored = PreprocessedFunction.__new__(PreprocessedFunction)
    restored.__setstate__(state)

    assert "_unnormalize_vect" not in restored.__dict__
    assert "_unnormalize_grad" not in restored.__dict__
    # The methods are rebound to the non-deprecated design-space methods,
    # so evaluation does not emit a DeprecationWarning.
    assert restored._denormalize_vect == design_space.denormalize_vect
    assert restored._denormalize_grad == design_space.denormalize_grad

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert_allclose(restored.evaluate(x_normalized), expected)
