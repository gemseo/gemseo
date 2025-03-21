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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import logging
import operator
import re
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy import array
from numpy import array_equal
from numpy import float64
from numpy import inf
from numpy import int64
from numpy import ndarray
from numpy import ones
from numpy import zeros
from numpy.linalg import norm
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal
from pydantic import ValidationError
from scipy.sparse import csr_array

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.algos.optimization_result import OptimizationResult
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.problems.mdo.sobieski.core.problem import SobieskiProblem
from gemseo.utils.repr_html import REPR_HTML_WRAPPER

CURRENT_DIR = Path(__file__).parent
TEST_INFILE = CURRENT_DIR / "design_space.csv"
FAIL_HDF = CURRENT_DIR / "fail.hdf5"

DesignVariableType = DesignSpace.DesignVariableType

DTYPE = DesignSpace._DesignSpace__DEFAULT_COMMON_DTYPE
FLOAT = DesignSpace.DesignVariableType.FLOAT
INTEGER = DesignSpace.DesignVariableType.INTEGER


@pytest.fixture
def design_space():
    """The main design space to be used by the test function.

    Feel free to add new variables.
    """
    ds = DesignSpace()
    ds.add_variable("x1", lower_bound=0.0, upper_bound=2.0)
    ds.add_variable("x2", lower_bound=-2.0, upper_bound=2.0)
    ds.add_variable("x3", type_=INTEGER, lower_bound=0, upper_bound=2)
    ds.add_variable("x4", type_="float", lower_bound=-1.0, upper_bound=0.0, value=-0.5)
    ds.add_variable(
        "x5", size=3, type_="float", lower_bound=-1.0, upper_bound=0.0, value=-0.5
    )
    ds.add_variable("x6", upper_bound=2.0)
    ds.add_variable("x7", lower_bound=0.0)
    ds.add_variable("x8", type_=INTEGER, lower_bound=1, upper_bound=1)
    ds.add_variable("x9", size=3, lower_bound=-1.0, upper_bound=2.0)
    ds.add_variable("x10", size=3)
    ds.add_variable("x11", size=2)
    ds.add_variable("x12")
    ds.add_variable("x13", value=array([0.5]))
    ds.add_variable("x14", type_=INTEGER, value=array([2]))
    ds.add_variable("x15")
    ds.add_variable("x16", size=2, type_=FLOAT, value=array([1.0, 2.0]))
    ds.add_variable("x17", size=2, type_=INTEGER, value=array([1, 2]))
    ds.add_variable("x18", lower_bound=-1.0, upper_bound=2.0)
    ds.add_variable("x19", lower_bound=1.0, upper_bound=3.0)
    ds.add_variable("x20", type_=b"float")
    ds.add_variable("x21", value=0.5)
    ds.add_variable("x22", size=2)
    ds.add_variable(
        "x23", lower_bound=0.0, upper_bound=1.0, value=array([1]), type_="float"
    )
    return ds


def test_add_variable_when_already_exists(design_space) -> None:
    """Check that adding an existing variable raises an error."""
    design_space.add_variable("varname")
    with pytest.raises(
        ValueError, match=re.escape("The variable 'varname' already exists.")
    ):
        design_space.add_variable(name="varname")


nonpositivity_message = "Input should be greater than 0"


@pytest.mark.parametrize(
    ("size", "message"),
    [
        (-1, nonpositivity_message),
        (0, nonpositivity_message),
        (0.4, "Input should be a valid integer, got a number with a fractional part"),
    ],
)
def test_add_variable_with_wrong_size(design_space, size, message) -> None:
    """Check that adding a variable with a wrong size raises an error."""
    with pytest.raises(ValidationError, match=message):
        design_space.add_variable(name="varname", size=size)


def test_add_variable_with_unkown_type(design_space) -> None:
    """Check that adding a variable with unknown type raises an error."""
    with pytest.raises(ValidationError, match="Input should be 'float' or 'integer'"):
        design_space.add_variable(name="varname", type_="a")


def test_add_variable_with_unnumerizable_value(design_space) -> None:
    """Check that adding a variable with unnumerizable value raises an error."""
    expected = re.escape(
        "The following value of variable 'varname' is neither None nor complex "
        "and cannot be cast to float: <built-in function len> (index 0).",
    )
    with pytest.raises(ValueError, match=expected):
        design_space.add_variable(name="varname", value=len)


@pytest.mark.parametrize(
    ("arg", "message"),
    [
        (
            "lower_bound",
            "The following lower bound component is not a number: nan (index 0).",
        ),
        (
            "upper_bound",
            "The following upper bound component is not a number: nan (index 0).",
        ),
        (
            "value",
            "The following value of variable 'varname' is neither None nor a number: "
            "nan (index 0).",
        ),
    ],
)
def test_add_variable_with_nan_value(design_space, arg, message) -> None:
    """Check that adding a variable with nan value raises an error."""
    with pytest.raises(ValueError, match=re.escape(message)):
        design_space.add_variable(name="varname", **{arg: float("nan")})


@pytest.mark.parametrize(
    ("arg", "side"), [("lower_bound", "lower"), ("upper_bound", "upper")]
)
def test_add_variable_with_inconsistent_bound_size(design_space, arg, side) -> None:
    """Check that using bounds with inconsistent size raises an error."""
    with pytest.raises(ValidationError, match=f"The {side} bound should be of size 3."):
        design_space.add_variable(name="varname", size=3, **{arg: [0.0, 0.0]})


def test_add_variable_with_upper_bounds_lower_than_lower_ones(design_space) -> None:
    """Check that using upper bounds lower than lower ones raises an error."""
    with pytest.raises(
        ValidationError,
        match=re.escape(
            "The upper bounds must be greater than or equal to the lower bounds."
        ),
    ):
        design_space.add_variable(
            name="varname",
            size=3,
            lower_bound=[0, 1.0, 0],
            upper_bound=[1, 0.0, 1],
        )


@pytest.mark.parametrize("arg", ["lower_bound", "upper_bound", "value"])
def test_add_variable_with_2d_object(design_space, arg) -> None:
    """Check that using a 2d iterable object raises an error."""
    variable_name = " of variable 'varname'" if arg == "value" else ""
    message = (
        (
            f"The value [[1.]]{variable_name} has a dimension greater than 1 "
            "while a scalar or a 1D iterable object (array, list, tuple, ...) "
            "was expected."
        )
        if arg == "value"
        else "validation errors for Variable"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        design_space.add_variable("varname", **{arg: [[1.0]]})


@pytest.mark.parametrize(
    ("l_b", "u_b", "value"),
    [
        (1.0, 2.0, 3.0),
        (1.0, 2.0, 0.0),
    ],
)
def test_add_variable_with_value_out_of_bounds(design_space, l_b, u_b, value) -> None:
    """Check that setting a value out of bounds raises an error.

    Check also that after this error is raised, the design space does not contain the
    variable.
    """
    expected = (
        f"The current value of variable 'varname' ({value}) is not "
        f"between the lower bound {l_b} and the upper bound {u_b}."
    )
    with pytest.raises(ValueError, match=re.escape(expected)):
        design_space.add_variable(
            name="varname",
            size=1,
            type_=FLOAT,
            lower_bound=l_b,
            upper_bound=u_b,
            value=value,
        )

    assert "varname" not in design_space


def test_creation_4() -> None:
    design_space = DesignSpace()
    design_space.add_variable("varname")
    with pytest.raises(
        KeyError,
        match=re.escape("There is no current value for the design variables: varname"),
    ):
        design_space.get_current_value(normalize=True)


def test_add_variable_value(design_space) -> None:
    design_space.add_variable(
        "varname",
        size=3,
        type_=FLOAT,
        lower_bound=0.0,
        upper_bound=1.0,
        value=[None, None, None],
    )


@pytest.mark.parametrize(
    "current_x",
    [
        {"x1": array([1.0]), "x2": array([0.0])},
        OptimizationResult(x_opt=array([1.0, 0.0])),
    ],
)
def test_set_current_value(design_space, current_x) -> None:
    design_space.filter(["x1", "x2"])
    design_space.set_current_value(current_x)
    x_n = design_space.get_current_value(normalize=True)
    assert (x_n == 0.5).all()


def test_set_current_value_with_malformed_mapping_arg(design_space) -> None:
    """Check that setting the current value from a malformed mapping raises an error."""
    design_space.filter("x1")
    with pytest.raises(
        Exception,
        match=re.escape(
            "The variable x1 of size 1 cannot be set with an array of size 2."
        ),
    ):
        design_space.set_current_value({"x1": array([1.0, 1.0])})


def test_set_current_value_with_malformed_opt_arg(design_space) -> None:
    """Check that setting the current value from a malformed optimization result raises
    an error."""
    with pytest.raises(
        Exception,
        match=f"Invalid x_opt, dimension mismatch: {design_space.dimension} != 1",
    ):
        design_space.set_current_value(OptimizationResult(x_opt=array([1.0])))


def test_set_current_value_with_malformed_current_x(design_space) -> None:
    """Check that setting the current value from a float raises an error."""
    with pytest.raises(
        TypeError,
        match=re.escape(
            "The current design value should be either an array, "
            "a dictionary of arrays or an optimization result; "
            "got <class 'float'> instead."
        ),
    ):
        design_space.set_current_value(1.0)


def test_read_from_csv() -> None:
    """Check that a variable name is correct when reading a CSV file."""
    ds = DesignSpace.from_csv(CURRENT_DIR / "design_space_4.csv")
    assert ds.variable_names == ["x_shared"]


def test_integer_variable_set_current_x(design_space) -> None:
    """Check that an integer value is correctly set."""
    design_space.filter("x3")
    x_i = array([0], dtype=int64)
    design_space.set_current_value(x_i)
    x_i_conv = design_space.convert_dict_to_array(
        design_space.convert_array_to_dict(x_i)
    )
    assert x_i_conv.dtype == x_i.dtype
    assert x_i_conv == x_i


def test_integer_variable_round_vect(design_space) -> None:
    """Check that a float value is correctly rounded to the closest integer."""
    design_space.filter("x3")
    assert design_space.round_vect(array([1.2])) == 1
    assert design_space.round_vect(array([1.9])) == 2

    assert (design_space.round_vect(array([[1.9], [1.2]])) == array([[2], [1]])).all()

    vector = array([1.2])
    rounded_vector = design_space.round_vect(vector)
    assert rounded_vector == array([1.0])
    assert vector != rounded_vector

    rounded_vector = design_space.round_vect(vector, copy=False)
    assert rounded_vector == array([1.0])
    assert vector == rounded_vector


@pytest.mark.parametrize("copy", [True, False])
def test_filter_by_variable_names(design_space, copy) -> None:
    """Check that the design space can be filtered by variables dimensions."""
    design_space_with_x5 = design_space.filter("x5", copy=copy)
    if not copy:
        design_space_with_x5 = design_space

    assert "x4" not in design_space_with_x5
    assert "x5" in design_space_with_x5

    if copy:
        assert design_space_with_x5 is not design_space


def test_filter_with_an_unknown_variable(design_space) -> None:
    """Check that filtering a design space with an unknown name raises an error."""
    with pytest.raises(
        ValueError, match=re.escape("Variable 'unknown_x' is not known.")
    ):
        design_space.filter("unknown_x")


@pytest.mark.parametrize("current_value", [[0.2, 0.5], None])
def test_filter_dimensions(current_value) -> None:
    """Check that the design space can be filtered by variables dimensions."""
    space = DesignSpace()
    space.add_variable("z", 1, "float", -0.6, -0.4, -0.5)
    space.add_variable("x", 2, "float", [0.1, 0.4], [0.3, 0.6], current_value)
    space.add_variable("y", 1, "integer", 7, 9, 8)
    space.filter_dimensions("x", [0])
    assert space.dimension == 3
    assert space.variable_sizes == {"z": 1, "x": 1, "y": 1}
    assert space.variable_types == {"z": "float", "x": "float", "y": "integer"}
    assert_array_equal(space.get_lower_bounds(), [-0.6, 0.1, 7])
    assert_array_equal(space.get_upper_bounds(), [-0.4, 0.3, 9])
    if current_value is not None:
        assert_array_equal(space.get_current_value(), [-0.5, 0.2, 8])

    assert space.names_to_indices == {"z": range(1), "x": range(1, 2), "y": range(2, 3)}


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        ([0, 3], "Dimension 3 of variable 'x5' does not exist."),
        ([0, 3, 4], "Dimensions 3 and 4 of variable 'x5' do not exist."),
    ],
)
def test_filter_dimensions_nonexistent(design_space, indices, message) -> None:
    """Check that the design space cannot filter nonexistent dimensions."""
    with pytest.raises(ValueError, match=message):
        design_space.filter_dimensions("x5", indices)


def check_variable(
    space: DesignSpace,
    name: str,
    reference_space: DesignSpace,
    check_value: bool,
) -> None:
    """Check a variable of a design space.

    Args:
        space: The design space.
        name: The name of the variable.
        reference_space: The design space of reference.
        check_value: Whether to check the value of the variable.

    Returns:
        Whether the variable is valid.
    """
    assert name in space
    assert space.get_size(name) == reference_space.get_size(name)
    assert space.get_type(name) == reference_space.get_type(name)
    assert_equal(space.get_lower_bound(name), reference_space.get_lower_bound(name))
    assert_equal(space.get_upper_bound(name), reference_space.get_upper_bound(name))
    if check_value:
        assert_equal(
            space.get_current_value([name]),
            reference_space.get_current_value([name]),
        )


def test_extend() -> None:
    """Test the extension of a design space with another."""
    design_space = DesignSpace()
    design_space.add_variable(
        "x1", type_="float", lower_bound=-1.0, upper_bound=0.0, value=-0.5
    )
    other = DesignSpace()
    other.add_variable(
        "x2", size=3, type_="float", lower_bound=-1.0, upper_bound=0.0, value=-0.5
    )
    other.add_variable("x3")
    design_space.extend(other)
    check_variable(design_space, "x2", other, True)
    check_variable(design_space, "x3", other, False)


def test_active_bounds() -> None:
    """Check whether active bounds are correctly identified."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=2.0)
    design_space.add_variable("y", lower_bound=-2.0, upper_bound=2.0)
    design_space.add_variable("z")
    lb_1, ub_1 = design_space.get_active_bounds({
        "x": array([0.0]),
        "y": array([2.0]),
        "z": array([2.0]),
    })

    lb_2, ub_2 = design_space.get_active_bounds(array([1e-12, 2.0 - 1e-12, 1e-12]))

    assert lb_1 == lb_2
    assert lb_1["x"] == [True]
    assert lb_1["y"] == [False]
    assert not lb_1["z"][0]
    assert ub_1 == ub_2
    assert ub_1["y"] == [True]
    assert ub_1["x"] == [False]
    assert not ub_1["z"][0]

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Expected dict or array for x_vec argument; got <class 'str'>."
        ),
    ):
        design_space.get_active_bounds("test")

    with pytest.raises(
        KeyError,
        match=re.escape(
            "There is no current value for the design variables: x, y and z."
        ),
    ):
        design_space.get_active_bounds()


def test_get_indexed_variable_names() -> None:
    """Check the variables names obtained with get_indexed_variable_names()."""
    design_space = DesignSpace()
    design_space.add_variable("x")
    design_space.add_variable("z", size=2)
    assert design_space.get_indexed_variable_names() == ["x", "z[0]", "z[1]"]
    assert design_space.get_indexed_variable_names(["x", "z"]) == ["x", "z[0]", "z[1]"]
    assert design_space.get_indexed_variable_names(["z", "x"]) == ["z[0]", "z[1]", "x"]
    assert design_space.get_indexed_variable_names("x") == ["x"]
    assert design_space.get_indexed_variable_names(["z"]) == ["z[0]", "z[1]"]


@pytest.mark.parametrize(
    ("name", "lower_bound", "upper_bound"),
    [("x6", -inf, 2.0), ("x7", 0.0, inf)],
)
def test_bounds(design_space, name, lower_bound, upper_bound) -> None:
    """Check that bounds are correctly retrieved."""
    assert design_space.get_lower_bound(name) == lower_bound
    assert design_space.get_upper_bound(name) == upper_bound
    assert design_space.get_lower_bounds([name]) == lower_bound
    assert design_space.get_upper_bounds([name]) == upper_bound


@pytest.mark.parametrize(
    ("variable_names", "as_dict", "lower_bounds"),
    [
        (
            None,
            False,
            [
                0.0,
                -2.0,
                0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -inf,
                0.0,
                1,
                -1.0,
                -1.0,
                -1.0,
                -inf,
                -inf,
                -inf,
                -inf,
                -inf,
                -inf,
                -inf,
                -inf,
                -inf,
                -inf,
                -inf,
                -inf,
                -inf,
                -1.0,
                1.0,
                -inf,
                -inf,
                -inf,
                -inf,
                0.0,
            ],
        ),
        (
            None,
            True,
            {
                "x1": [0.0],
                "x2": [-2.0],
                "x3": [0],
                "x4": [-1.0],
                "x5": [-1.0, -1.0, -1.0],
                "x6": [-inf],
                "x7": [0.0],
                "x8": [1],
                "x9": [-1.0, -1.0, -1.0],
                "x10": [-inf, -inf, -inf],
                "x11": [-inf, -inf],
                "x12": [-inf],
                "x13": [-inf],
                "x14": [-inf],
                "x15": [-inf],
                "x16": [-inf, -inf],
                "x17": [-inf, -inf],
                "x18": [-1.0],
                "x19": [1.0],
                "x20": [-inf],
                "x21": [-inf],
                "x22": [-inf, -inf],
                "x23": [0.0],
            },
        ),
        (["x22", "x23"], False, [-inf, -inf, 0.0]),
        (["x22", "x23"], True, {"x22": [-inf, -inf], "x23": [0.0]}),
        (["x23"], False, [0.0]),
        (["x23"], True, {"x23": [0.0]}),
    ],
)
def test_get_lower_bounds(design_space, variable_names, as_dict, lower_bounds) -> None:
    """Check the getting of the lower bounds."""
    assert_equal(design_space.get_lower_bounds(variable_names, as_dict), lower_bounds)


@pytest.mark.parametrize(
    ("variable_names", "as_dict", "upper_bounds"),
    [
        (
            None,
            False,
            [
                2.0,
                2.0,
                2,
                0.0,
                0.0,
                0.0,
                0.0,
                2.0,
                inf,
                1,
                2.0,
                2.0,
                2.0,
                inf,
                inf,
                inf,
                inf,
                inf,
                inf,
                inf,
                inf,
                inf,
                inf,
                inf,
                inf,
                inf,
                2.0,
                3.0,
                inf,
                inf,
                inf,
                inf,
                1.0,
            ],
        ),
        (
            None,
            True,
            {
                "x1": [2.0],
                "x2": [2.0],
                "x3": [2],
                "x4": [0.0],
                "x5": [0.0, 0.0, 0.0],
                "x6": [2.0],
                "x7": [inf],
                "x8": [1],
                "x9": [2.0, 2.0, 2.0],
                "x10": [inf, inf, inf],
                "x11": [inf, inf],
                "x12": [inf],
                "x13": [inf],
                "x14": [inf],
                "x15": [inf],
                "x16": [inf, inf],
                "x17": [inf, inf],
                "x18": [2.0],
                "x19": [3.0],
                "x20": [inf],
                "x21": [inf],
                "x22": [inf, inf],
                "x23": [1.0],
            },
        ),
        (["x22", "x23"], False, [inf, inf, 1.0]),
        (["x22", "x23"], True, {"x22": [inf, inf], "x23": [1.0]}),
        (["x23"], False, [1.0]),
        (["x23"], True, {"x23": [1.0]}),
    ],
)
def test_get_upper_bounds(design_space, variable_names, as_dict, upper_bounds) -> None:
    """Check the getting of the upper bounds."""
    assert_equal(design_space.get_upper_bounds(variable_names, as_dict), upper_bounds)


def test_bounds_set_lower_bound_with_nan(design_space) -> None:
    """Check that setting lower bound with nan raises an error."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The following lower bound component is not a number: nan (index 0)."
        ),
    ):
        design_space.set_lower_bound("x6", float("nan"))


def test_bounds_set_lower_bound_with_inconsistent_size(design_space) -> None:
    """Check that setting lower bound with inconsistent sized value raises an error."""
    with pytest.raises(
        ValueError, match=re.escape("The lower bound should be of size 1.")
    ):
        design_space.set_lower_bound("x6", ones(2))


def test_bounds_set_upper_bound_with_nan(design_space) -> None:
    """Check that setting upper bound with nan raises an error."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The following upper bound component is not a number: nan (index 0)."
        ),
    ):
        design_space.set_upper_bound("x6", float("nan"))


def test_bounds_set_upper_bound_with_inconsistent_size(design_space) -> None:
    """Check that setting upper bound with inconsistent sized value raises an error."""
    with pytest.raises(
        ValueError, match=re.escape("The upper bound should be of size 1.")
    ):
        design_space.set_upper_bound("x6", ones(2))


def test_normalization() -> None:
    """Check the normalization of design variables."""
    design_space = DesignSpace()
    design_space.add_variable(
        "x_1",
        size=2,
        lower_bound=array([-inf, 0.0]),
        upper_bound=array([0.0, inf]),
    )
    design_space.add_variable("x_2", lower_bound=0.0, upper_bound=10.0)
    design_space.add_variable("x_3", type_=INTEGER, lower_bound=0.0, upper_bound=10.0)
    # Test the normalization policies:
    assert not design_space.normalize["x_1"][0]
    assert not design_space.normalize["x_1"][1]
    assert design_space.normalize["x_2"]
    assert not design_space.normalize["x_3"]
    # Test the normalization:
    design_space.set_current_value(array([-10.0, 10.0, 5.0, 5]))
    current_x_norm = design_space.get_current_value(normalize=True)
    ref_current_x_norm = array([-10.0, 10.0, 0.5, 5])
    assert norm(current_x_norm - ref_current_x_norm) == pytest.approx(0.0)

    unnorm_curent_x = design_space.unnormalize_vect(current_x_norm)
    current_x = design_space.get_current_value()
    assert norm(unnorm_curent_x - current_x) == pytest.approx(0.0)

    x_2d = ones((5, 4))
    x_u = design_space.unnormalize_vect(x_2d)
    assert (x_u == array([1.0, 1.0, 10.0, 1] * 5).reshape((5, 4))).all()

    x_n = design_space.normalize_vect(x_2d)
    assert (x_n == array([1.0, 1.0, 0.1, 1] * 5).reshape((5, 4))).all()


def test_normalize_vect_with_integer(design_space) -> None:
    """Check that an integer vector is correctly normalized."""
    design_space.filter("x8")
    assert design_space.normalize_vect(ones(1))[0] == 1


@pytest.mark.parametrize(
    ("vect", "get_item"),
    [
        (ones(1) * 0, operator.itemgetter(0)),
        (array([[0.0], [0.0]]), lambda x: x[0][0]),
        (array([[0.0], [0.0]]), lambda x: x[1][0]),
    ],
)
def test_unnormalize_vect_with_integer(design_space, vect, get_item) -> None:
    """Check that an integer vector is correctly unnormalized."""
    design_space.filter("x8")
    assert get_item(design_space.unnormalize_vect(vect)) == 0


def test_norm_policy() -> None:
    """Check the normalization policy."""
    design_space = DesignSpace()
    design_space.add_variable(
        "x_1",
        size=2,
        lower_bound=array([-inf, 0.0]),
        upper_bound=array([0.0, inf]),
    )

    with pytest.raises(ValueError, match=re.escape("Variable 'foo' is not known.")):
        design_space._add_norm_policy("foo")


def test_current_x() -> None:
    names = ["x_1", "x_2"]
    sizes = {"x_1": 1, "x_2": 2}
    l_b = {"x_1": 0.5, "x_2": (-inf, 2.0)}
    u_b = {"x_1": inf, "x_2": (4.0, 5.0)}
    var_types = {
        "x_1": FLOAT,
        "x_2": INTEGER,
    }
    x_0 = np.array([0.5, 4.0, 4.0])
    # create the design space
    design_space = DesignSpace()

    # fill the design space
    for name in names:
        design_space.add_variable(
            name,
            size=sizes[name],
            type_=var_types[name],
            lower_bound=l_b[name],
            upper_bound=u_b[name],
        )

    design_space.set_current_value(x_0)
    design_space.check()

    expected = re.escape("Expected current_x variables: x_1 and x_2; got x_1.")
    with pytest.raises(ValueError, match=expected):
        design_space.set_current_value({"x_1": array([0.0])})

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The component x_1[0] of the given array (-999.5) is lower "
            "than the lower bound (0.5) by 1.0e+03.",
        ),
    ):
        design_space.set_current_value(x_0 - 1000.0)

    """
    Design Space: 3 scalar variables
    Variable   Type     Lower  Current  Upper
    x_1        float    0.5    0.5      inf
    x_2[0]     integer  -inf   4        4
    x_2[1]     integer  2      4        5
    """

    assert design_space.get_type("x_1") == np.array([FLOAT])
    with pytest.raises(ValueError, match=re.escape("Variable 'x_3' is not known.")):
        assert design_space.get_type("x_3")
    with pytest.raises(ValueError, match=re.escape("Variable 'x_3' is not known.")):
        assert design_space.get_size("x_3")

    design_space.set_current_variable("x_1", np.array([5.0]))
    assert design_space.get_current_value(as_dict=True)["x_1"][0] == 5.0

    with pytest.raises(ValueError, match=re.escape("Variable 'x_3' is not known.")):
        design_space.set_current_variable("x_3", 1.0)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The upper bounds must be greater than or equal to the lower bounds."
        ),
    ):
        design_space.add_variable("error", lower_bound=1.0, upper_bound=0.0)

    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=2.0)
    design_space.set_current_value({"x": None})
    assert not design_space.has_current_value


def get_sobieski_design_space():
    """Return the design space for the Sobieski problem."""
    names = [
        "x_shared",
        "x_1",
        "x_2",
        "x_3",
        "y_14",
        "y_32",
        "y_31",
        "y_24",
        "y_34",
        "y_23",
        "y_21",
        "y_12",
    ]
    problem = SobieskiProblem()
    def_inputs = problem.get_default_inputs_equilibrium(names)

    ref_ds = DesignSpace()
    for name in names:
        value = def_inputs[name]
        l_b, u_b = problem.get_bounds_by_name([name])
        size = value.size
        ref_ds.add_variable(
            name,
            size=size,
            type_="float",
            lower_bound=l_b,
            upper_bound=u_b,
            value=value,
        )

    return ref_ds


def test_read_write(tmp_wd) -> None:
    """Check that from_csv and to_csv work correctly."""
    ref_ds = get_sobieski_design_space()
    f_path = Path("sobieski_design_space.csv")
    ref_ds.to_csv(f_path)
    read_ds = DesignSpace.from_csv(f_path)
    read_ds.get_lower_bounds()
    check_ds(ref_ds, read_ds, f_path)

    ds = DesignSpace.from_csv(TEST_INFILE)
    assert not ds.has_current_value
    for i in range(1, 9):
        testfile = CURRENT_DIR / f"design_space_fail_{i}.csv"
        with pytest.raises(ValueError):
            DesignSpace.from_csv(testfile)

    for i in range(1, 4):
        testfile = CURRENT_DIR / f"design_space_{i}.csv"
        header = None
        if i == 2:
            header = ["name", "value", "lower_bound", "type", "upper_bound"]
        DesignSpace.from_csv(testfile, header=header)

    ds = DesignSpace.from_csv(TEST_INFILE)
    ds.set_lower_bound("x_shared", -inf)
    ds.set_upper_bound("x_shared", inf)

    out_f = Path("table.csv")
    ds.to_csv(out_f, sortby="upper_bound")
    assert out_f.exists()


@pytest.mark.parametrize("index", [1, 2, 3, 4, 5, 6, 7, 8])
def test_read_write_failure(tmp_wd, index) -> None:
    """Check that from_csv and to_csv work correctly."""
    testfile = CURRENT_DIR / f"design_space_fail_{index}.csv"
    with pytest.raises(ValueError):
        DesignSpace.from_csv(testfile)


def test_dict_to_array() -> None:
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=2.0)
    design_space.add_variable("y", lower_bound=-2.0, upper_bound=2.0)

    with pytest.raises(KeyError, match="'y'"):
        design_space.convert_dict_to_array({"x": array([1.0])})


@pytest.mark.parametrize(("name", "dtype"), [("x1", float64), ("x3", int64)])
def test_dict_to_array_dtype(design_space, name, dtype) -> None:
    """Check the data type of the array returned by
    ``DesignSpace.convert_dict_to_array``."""
    assert (
        design_space.convert_dict_to_array(
            {"x1": array([1.0]), "x3": array([1])}, [name]
        ).dtype
        == dtype
    )


def check_ds(ref_ds, read_ds, f_path) -> None:
    """:param ref_ds: param read_ds:
    :param f_path:
    :param read_ds:
    """
    assert f_path.exists()
    assert read_ds.variable_names == ref_ds.variable_names

    err = read_ds.get_lower_bounds() - ref_ds.get_lower_bounds()
    assert norm(err) == pytest.approx(0.0)

    err = read_ds.get_upper_bounds() - ref_ds.get_upper_bounds()
    assert norm(err) == pytest.approx(0.0)

    err = read_ds.get_current_value() - ref_ds.get_current_value()
    assert norm(err) == pytest.approx(0.0)

    type_read = [t for name in read_ds for t in read_ds.get_type(name)]

    type_ref = [t for name in read_ds for t in ref_ds.get_type(name)]

    assert type_read == type_ref

    for name in ref_ds:
        assert name in read_ds

    ref_str = str(ref_ds)
    assert ref_str == str(read_ds)
    assert len(ref_str) > 1000
    assert len(ref_str.split("\n")) > 20


def test_hdf5_export(tmp_wd) -> None:
    """Tests the export of a Design space in the HDF5 format."""
    ref_ds = get_sobieski_design_space()
    f_path = Path("_sobieski_design_space.h5")
    ref_ds.to_hdf(f_path)


def test_hdf5_with_node(tmp_wd):
    """Tests the hdf import/export of a Design space in a specific node."""
    ref_ds = get_sobieski_design_space()
    f_path = Path("ssbj_ds_node.h5")
    node_ds = "node_ds"
    ref_ds.to_hdf(f_path, hdf_node_path=node_ds)

    with pytest.raises(KeyError):
        DesignSpace().from_hdf(f_path)

    with pytest.raises(KeyError):
        DesignSpace().from_hdf(f_path, hdf_node_path="wrong_node")

    imp_ds = DesignSpace().from_hdf(f_path, hdf_node_path=node_ds)

    check_ds(ref_ds, imp_ds, f_path)


@pytest.mark.parametrize("suffix", [".csv", ".h5", ".hdf", ".hdf5", ".txt"])
def test_to_from_file(tmp_wd, suffix) -> None:
    """Check that the methods to_file() and from_file() work correctly."""
    file_path = Path("foo").with_suffix(suffix)
    design_space = get_sobieski_design_space()
    design_space.to_file(file_path)
    assert h5py.is_hdf5(file_path) == file_path.suffix.startswith((".h5", ".hdf"))

    read_design_space = DesignSpace.from_file(file_path)
    check_ds(design_space, read_design_space, file_path)


def test_import_error_with_missing_file() -> None:
    """Check that a missing HDF file cannot be imported."""
    hdf_file_name = "i_dont_exist.h5"
    assert not Path(hdf_file_name).exists()
    with pytest.raises((OSError, FileNotFoundError)):
        DesignSpace.from_hdf(hdf_file_name)


def test_fail_import() -> None:
    """Check that a malformed HDF file cannot be imported."""
    with pytest.raises(KeyError):
        DesignSpace.from_hdf(FAIL_HDF)


@pytest.fixture(scope="module")
def table_template() -> str:
    """Table template with capitalization of field names."""
    return """
+------+-------------+-------+-------------+-------+
| Name | Lower bound | Value | Upper bound | Type  |
+------+-------------+-------+-------------+-------+
| x    |     -inf    |  None |     inf     | float |
| y{index_0} |     -inf    |  None |     inf     | float |
| y{index_1} |     -inf    |  None |     inf     | float |
+------+-------------+-------+-------------+-------+
""".strip()


@pytest.fixture(scope="module")
def table_template_2() -> str:
    """Table template without capitalization of field names."""
    return """
+------+-------------+-------+-------------+-------+
| name | lower_bound | value | upper_bound | type  |
+------+-------------+-------+-------------+-------+
| x    |     -inf    |  None |     inf     | float |
| y{index_0} |     -inf    |  None |     inf     | float |
| y{index_1} |     -inf    |  None |     inf     | float |
+------+-------------+-------+-------------+-------+
""".strip()


@pytest.fixture
def design_space_2() -> DesignSpace:
    """Return a design space with scalar and vectorial variables."""
    design_space = DesignSpace()
    design_space.add_variable("x")
    design_space.add_variable("y", size=2)
    return design_space


@pytest.mark.parametrize(
    ("with_index", "indexes"),
    [(True, ("[0]", "[1]")), (False, ("   ", "   "))],
)
def test_get_pretty_table(
    table_template_2,
    design_space_2,
    with_index,
    indexes,
) -> None:
    """Check that a design space is correctly rendered."""
    assert (
        table_template_2.format(index_0=indexes[0], index_1=indexes[1])
        == design_space_2.get_pretty_table(with_index=with_index).get_string()
    )


def test_get_pretty_table_with_selected_fields(design_space_2) -> None:
    """Check the rendering of a design as a pretty table with selected fields."""
    assert (
        """
+------+
| name |
+------+
| x    |
| y    |
| y    |
+------+
""".strip()
        == design_space_2.get_pretty_table(fields=["name"]).get_string()
    )


@pytest.mark.parametrize("name", ["", "foo"])
def test_str(table_template, design_space_2, name) -> None:
    """Check that a design space is correctly rendered."""
    if name:
        design_space_2.name = name
        prefix = " "
    else:
        prefix = ""

    table_template = f"Design space:{prefix}{name}\n{table_template}"
    assert table_template.format(index_0="[0]", index_1="[1]") == str(design_space_2)


@pytest.mark.parametrize(
    ("normalized", "expected"),
    [(False, [-1, 0.5, 2]), (True, [0, 0.5, 1])],
)
def test_project_into_bounds(design_space, normalized, expected) -> None:
    """Tests the projection onto the design space bounds."""
    design_space.filter("x9")
    x_p = design_space.project_into_bounds([-2, 0.5, 3], normalized=normalized)
    assert norm(x_p - expected) == pytest.approx(0.0)


def test_contains(design_space) -> None:
    """Check the DesignSpace.__contains__."""
    assert "x1" in design_space
    assert "unknown_name" not in design_space


def test_len(design_space) -> None:
    """Check the length of a design space."""
    assert len(design_space) == len(design_space.variable_names)


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["x10"], [0, 1, 2]),
        (["x11"], [3, 4]),
        (["x12"], [5]),
        (["x10", "x11"], [0, 1, 2, 3, 4]),
        (["x10", "x12"], [0, 1, 2, 5]),
        (["x11", "x12"], [3, 4, 5]),
    ],
)
def test_get_variables_indexes(design_space, names, expected) -> None:
    """Test the variables indexes getter."""
    design_space.filter(["x10", "x11", "x12"])
    assert (design_space.get_variables_indexes(names) == array(expected)).all()


@pytest.mark.parametrize(
    ("use_design_space_order", "expected"),
    [(True, array([0, 1, 2, 3, 4, 5])), (False, array([3, 4, 0, 1, 2, 5]))],
)
def test_get_variables_indexes_in_user_order(
    design_space,
    use_design_space_order,
    expected,
) -> None:
    """Test the variables indexes getter in user order."""
    design_space.filter(["x10", "x11", "x12"])
    assert_equal(
        design_space.get_variables_indexes(
            ["x11", "x10", "x12"],
            use_design_space_order,
        ),
        expected,
    )


def test_gradient_normalization(design_space) -> None:
    """Check that the normalization of the gradient performs well."""
    design_space.filter(["x18", "x19"])
    x_vect = array([0.5, 1.5])
    assert array_equal(
        design_space.unnormalize_vect(x_vect, minus_lb=False, no_check=False),
        design_space.normalize_grad(x_vect),
    )


def test_gradient_unnormalization(design_space) -> None:
    """Check that the unnormalization of the gradient performs well."""
    design_space.filter(["x18", "x19"])
    x_vect = array([0.5, 1.5])
    assert array_equal(
        design_space.normalize_vect(x_vect, minus_lb=False),
        design_space.unnormalize_grad(x_vect),
    )


def test_sparse_normalization() -> None:
    """Tests (de)normalization of sparse Jacobians."""
    design_space = DesignSpace()
    design_space.add_variable("x")
    design_space.add_variable("y", lower_bound=1.0, upper_bound=3.0)

    jac = array([[1.0, 1.0], [1.0, 2.0]])
    sparse_jac = csr_array(jac)

    assert (
        design_space.normalize_grad(jac)
        == design_space.normalize_grad(sparse_jac).toarray()
    ).all()
    assert (
        design_space.unnormalize_grad(jac)
        == design_space.unnormalize_grad(sparse_jac).toarray()
    ).all()


def test_vartype_passed_as_bytes(design_space) -> None:
    """Check that a variable type passed as bytes is properly decoded."""
    assert design_space.variable_types["x20"] == FLOAT


@pytest.mark.parametrize(
    ("name", "kind"),
    [("x13", "f"), ("x14", "i"), ("x16", "f"), ("x17", "i")],
)
def test_current_x_various_types(design_space, name, kind) -> None:
    """Check that set_current_value handles various types of data."""
    design_space.filter(["x13", "x14", "x15", "x16", "x17"])
    design_space.set_current_value({
        "x13": array([0.5]),
        "x14": array([2.0]),
        "x15": None,
        "x16": array([1.0, 2.0]),
        "x17": array([1, 2]),
    })
    assert design_space._current_value[name].dtype.kind == kind


def test_current_x_with_missing_variable(design_space) -> None:
    design_space.filter(["x13", "x14", "x15", "x16", "x17"])
    design_space.set_current_value({
        "x13": array([0.5]),
        "x14": array([2.0]),
        "x15": None,
        "x16": array([1.0, 2.0]),
        "x17": array([1, 2]),
    })
    assert design_space._current_value["x15"] is None


def test_design_space_name() -> None:
    """Check the naming of a design space."""
    assert DesignSpace().name == ""
    assert DesignSpace(name="my_name").name == "my_name"


@pytest.fixture(scope="module")
def design_space_for_normalize_vect() -> DesignSpace:
    """A design space to check normalize_vect."""
    design_space = DesignSpace()
    design_space.add_variable("x_1", 2, FLOAT, array([-inf, 0.0]), array([0.0, inf]))
    design_space.add_variable("x_2", 1, FLOAT, 0.0, 10.0)
    design_space.add_variable("x_3", 1, INTEGER, 0.0, 10.0)
    return design_space


@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.parametrize(
    ("input_vec", "ref"),
    [
        (np.array([-10, -20, 5, 5]), np.array([-10, -20, 0.5, 5])),
        (np.array([-10.0, -20, 5.0, 5]), np.array([-10, -20, 0.5, 5])),
    ],
)
def test_normalize_vect(
    design_space_for_normalize_vect,
    input_vec,
    ref,
    use_out,
) -> None:
    """Test that the normalization is correctly computed whether the input values are
    floats or integers."""
    out = array([0.0, 0.0, 0.0, 0.0]) if use_out else None
    result = design_space_for_normalize_vect.normalize_vect(input_vec, out=out)
    assert result == pytest.approx(ref)
    assert (id(result) == id(out)) is use_out


@pytest.mark.parametrize("out", [zeros(4), None])
@pytest.mark.parametrize(
    ("input_vec", "ref"),
    [
        (np.array([-10, -20, 0, 1]), np.array([-10, -20, 0, 1])),
        (np.array([-10.0, -20, 0.5, 1]), np.array([-10, -20, 5, 1])),
    ],
)
def test_unnormalize_vect(input_vec, ref, out) -> None:
    """Test that the unnormalization is correctly computed whether the input values are
    floats or integers."""
    design_space = DesignSpace()
    design_space.add_variable(
        "x_1",
        2,
        FLOAT,
        array([-inf, 0.0]),
        array([0.0, inf]),
    )
    design_space.add_variable("x_2", 1, FLOAT, 0.0, 10.0)
    design_space.add_variable("x_3", 1, INTEGER, 0.0, 10.0)

    assert design_space.unnormalize_vect(
        # Pass a copy of the array because the fact that DesignSpace.unnormalize_vect
        # overwrites the input array conflicts with pytest.mark.parametrize.
        array(input_vec),
        out=out,
    ) == pytest.approx(ref)


def test_unnormalize_vect_logging(caplog) -> None:
    """Check the warning logged when unnormalizing a vector."""
    design_space = DesignSpace()
    design_space.add_variable("x")  # unbounded variable
    design_space.add_variable(
        "y", 2, lower_bound=-3.0, upper_bound=4.0
    )  # bounded variable
    design_space.unnormalize_vect(array([2.0, -5.0, 6.0]))
    msg = "All components of the normalized vector should be between 0 and 1; "
    msg += f"lower bounds violated: {array([-5.0])}; "
    msg += f"upper bounds violated: {array([6.0])}."
    assert ("gemseo.algos.design_space", logging.WARNING, msg) in caplog.record_tuples


def test_iter() -> None:
    """Check that a DesignSpace can be iterated."""
    design_space = DesignSpace()
    design_space.add_variable("x1")
    design_space.add_variable("x2", size=2)
    assert list(design_space) == ["x1", "x2"]


def test_ineq() -> None:
    """Check that DesignSpace cannot be equal to any object other than a DesignSpace."""
    design_space = DesignSpace()
    assert design_space != 1


def test_transform() -> None:
    """Check that transformation and inverse transformation works correctly."""
    parameter_space = DesignSpace()
    parameter_space.add_variable("x", lower_bound=0.0, upper_bound=2.0)
    vector = array([1.0])
    transformed_vector = parameter_space.transform_vect(vector)
    assert transformed_vector == array([0.5])
    untransformed_vector = parameter_space.untransform_vect(transformed_vector)
    assert vector == untransformed_vector


@pytest.mark.parametrize("value", [array([1.0, 2.0]), None])
def test_rename_variable(value) -> None:
    """Check the renaming of a variable."""
    design_space = DesignSpace()
    design_space.add_variable("x", 2, "integer", 0.0, 2.0, value)
    names_to_indices = design_space._DesignSpace__names_to_indices
    indices = names_to_indices["x"]
    design_space.rename_variable("x", "y")
    assert "x" not in names_to_indices
    assert names_to_indices["y"] == indices
    assert "x" not in design_space
    assert "y" in design_space
    variable = design_space._variables["y"]
    assert variable.size == 2
    assert variable.type == "integer"
    assert_equal(variable.lower_bound, [0, 0])
    assert_equal(variable.upper_bound, [2, 2])
    assert "x" not in design_space._current_value
    if value is None:
        assert "y" not in design_space._current_value
    else:
        assert_array_equal(design_space._current_value["y"], value)


def test_rename_unknown_variable() -> None:
    """Check that a value error is raised when renaming of an unknown variable."""
    design_space = DesignSpace()
    with pytest.raises(
        ValueError, match=re.escape("The variable x is not in the design space.")
    ):
        design_space.rename_variable("x", "y")


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        (
            {
                "int": {
                    "size": 2,
                    "var_type": INTEGER,
                    "value": array([1, 2]),
                },
                "float": {
                    "size": 1,
                    "var_type": FLOAT,
                    "value": array([1.0]),
                },
            },
            True,
        ),
        (
            {
                "float_1": {
                    "size": 2,
                    "var_type": FLOAT,
                    "value": array([1, 2]),
                },
                "float_2": {
                    "size": 1,
                    "var_type": FLOAT,
                    "value": array([1.0]),
                },
            },
            False,
        ),
        (
            {
                "int_1": {
                    "size": 2,
                    "var_type": INTEGER,
                    "value": array([1, 2]),
                },
                "int_2": {
                    "size": 1,
                    "var_type": INTEGER,
                    "value": array([1]),
                },
            },
            True,
        ),
    ],
)
def test_has_integer_variables(variables, expected) -> None:
    """Test that the correct bool is returned by the _has_integer_variables method."""
    design_space = DesignSpace()
    for key, val in variables.items():
        design_space.add_variable(
            key,
            size=val["size"],
            type_=val["var_type"],
            value=val["value"],
        )

    assert design_space.has_integer_variables == expected


@pytest.fixture(scope="module")
def design_space_with_complex_value() -> DesignSpace:
    """A design space with a float variable whose value is complex."""
    design_space = DesignSpace()
    design_space.add_variable("x")
    design_space.set_current_value({"x": array([1.0 + 0j])})
    return design_space


@pytest.mark.parametrize("cast", [False, True])
def test_get_current_x_no_complex(design_space_with_complex_value, cast) -> None:
    """Check that the complex value of a float variable is converted to float."""
    current_x = design_space_with_complex_value.get_current_value(complex_to_real=cast)
    assert (current_x.dtype.kind == "c") is not cast


def test_cast_to_var_type(design_space: DesignSpace) -> None:
    """Test that a given value is cast to var_type in add_variable.

    Args:
        design_space: fixture that returns the design space for testing.
    """
    design_space.filter(["x23"])
    assert design_space.get_current_value() == array([1.0], dtype=np.float64)


@pytest.mark.parametrize("normalize", [True, False])
def test_normalization_casting(design_space: DesignSpace, normalize: bool) -> None:
    """Test that integer variable keep their type after unnormalization."""
    design_space.filter(["x14"])
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x, "f")
    out = problem.evaluate_functions(design_vector_is_normalized=normalize)
    assert out[0]["f"] == array([2])
    assert out[0]["f"].dtype == int64


@pytest.fixture(scope="module")
def design_space_to_check_membership() -> DesignSpace:
    """A design space to test the method check_membership."""
    design_space = DesignSpace()
    design_space.add_variable("x", type_=INTEGER, lower_bound=-2, upper_bound=-1)
    design_space.add_variable("y", size=2, lower_bound=1.0, upper_bound=2.0)
    return design_space


@pytest.mark.parametrize(
    ("x_vect", "variable_names", "error", "error_msg"),
    [
        (
            [0, 0],
            None,
            TypeError,
            (
                "The input vector should be an array or a dictionary; "
                "got a <class 'list'> instead."
            ),
        ),
        (array([-1, 1, 1]), None, None, None),
        (zeros(4), None, ValueError, "The array should be of size 3; got 4."),
        (array([-1, 1, 1]), ["x", "y"], None, None),
        (array([1, 1, -1]), ["y", "x"], None, None),
        (
            array([-3, 1, 1]),
            None,
            ValueError,
            (
                "The components [0] of the given array ([-3]) are lower "
                "than the lower bound ([-2.]) by [1.]."
            ),
        ),
        (
            array([1, 1, 1]),
            None,
            ValueError,
            (
                "The components [0] of the given array ([1]) are greater "
                "than the upper bound ([-1.]) by [2.]."
            ),
        ),
        (
            array([-1, 1, 3]),
            None,
            ValueError,
            (
                "The components [2] of the given array ([3]) are greater "
                "than the upper bound ([2.]) by [1.]."
            ),
        ),
        ({"x": array([-1]), "y": array([1, 1])}, None, None, None),
        ({"x": array([-1]), "y": array([1, 1])}, ["x", "y"], None, None),
        ({"x": array([-1]), "y": array([1, 1])}, ["y", "x"], None, None),
        ({"x": array([-1]), "y": array([1, 1])}, ["x"], None, None),
        ({"x": array([-1]), "y": array([1, 1])}, ["y"], None, None),
        ({"x": array([-1]), "y": None}, None, None, None),
        (
            {"x": array([-1.5]), "y": None},
            None,
            ValueError,
            "The variable x is of type integer; got x[0] = -1.5.",
        ),
        (
            {"x": array([-1]), "y": array([1, 1, 1])},
            None,
            ValueError,
            "The variable y of size 2 cannot be set with an array of size 3.",
        ),
        (
            {"x": array([-1]), "y": array([1, 0])},
            None,
            ValueError,
            (
                "The component y[1] of the given array (0) is lower "
                "than the lower bound (1.0) by 1.0e+00."
            ),
        ),
        (
            {"x": array([-1]), "y": array([1, 3])},
            None,
            ValueError,
            (
                "The component y[1] of the given array (3) is greater "
                "than the upper bound (2.0) by 1.0e+00."
            ),
        ),
    ],
)
def test_check_membership(
    design_space_to_check_membership,
    x_vect,
    variable_names,
    error,
    error_msg,
) -> None:
    """Check the method check_membership."""
    ds = design_space_to_check_membership

    if error is None:
        ds.check_membership(x_vect, variable_names)
        if variable_names is None and isinstance(x_vect, ndarray):
            assert_equal(ds._DesignSpace__lower_bounds_array, array([-2, 1, 1]))
            assert_equal(ds._DesignSpace__upper_bounds_array, array([-1, 2, 2]))
    else:
        with pytest.raises(error, match=re.escape(error_msg)):
            ds.check_membership(x_vect, variable_names)


@pytest.mark.parametrize(
    ("l_b", "expected_lb"),
    [(-5, array([-5, -5])), (array([-5, -inf]), array([-5, -inf]))],
)
@pytest.mark.parametrize(
    ("u_b", "expected_ub"),
    [(5, array([5, 5])), (array([5, inf]), array([5, inf]))],
)
def test_infinity_bounds_for_int(l_b, u_b, expected_lb, expected_ub) -> None:
    """Check that integer variables can handle -/+ infinity bounds.

    Args:
        l_b: The lower bounds.
        u_b: The upper bounds.
        expected_ub: The expected upper bounds.
        expected_lb: The expected lower bounds.
    """
    ds = DesignSpace()
    ds.add_variable("x", 2, lower_bound=l_b, upper_bound=u_b, type_=INTEGER)
    assert array_equal(ds._lower_bounds["x"], expected_lb)
    assert array_equal(ds._upper_bounds["x"], expected_ub)


@pytest.fixture(scope="module")
def fbb_design_space() -> DesignSpace:
    """Foo-bar-baz (fbb) design space for test_get_current_value()."""
    design_space = DesignSpace()
    design_space.add_variable("foo", lower_bound=1.0, value=1.0)
    design_space.add_variable(
        "bar", size=2, lower_bound=1.0, upper_bound=3.0, value=2.0
    )
    design_space.add_variable("baz", lower_bound=1.0, upper_bound=3.0, value=3.0)
    design_space.set_current_variable("baz", array([3.0 + 0.5j]))
    return design_space


@pytest.mark.parametrize("names", [None, ["baz", "foo"]])
@pytest.mark.parametrize("cast", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("as_dict", [False, True])
def test_get_current_value(fbb_design_space, names, cast, normalize, as_dict) -> None:
    """Check get_current_value() with all the combinations of its arguments."""
    result = fbb_design_space.get_current_value(
        variable_names=names,
        complex_to_real=cast,
        normalize=normalize,
        as_dict=as_dict,
    )
    if normalize:
        expected = {
            "foo": array([1.0]),
            "bar": array([0.5, 0.5]),
            "baz": array([1.0 + 0.25j]),
        }
    else:
        expected = {
            "foo": array([1.0]),
            "bar": array([2.0, 2.0]),
            "baz": array([3.0 + 0.5j]),
        }

    names = names or ["foo", "bar", "baz"]
    expected = {k: v for k, v in expected.items() if k in names}

    if cast:
        expected = {k: v.real for k, v in expected.items()}

    if not as_dict:
        expected = fbb_design_space.convert_dict_to_array(
            expected, variable_names=names
        )

    assert_equal(result, expected)


@pytest.mark.parametrize("as_dict", [True, False])
def test_get_current_value_empty_names(as_dict) -> None:
    design_space = DesignSpace()
    assert not design_space.get_current_value(variable_names=[], as_dict=as_dict)


def test_get_current_value_bad_names() -> None:
    design_space = DesignSpace()
    match = "There are no such variables named: bar and foo."
    with pytest.raises(ValueError, match=match):
        design_space.get_current_value(variable_names=["foo", "bar"])


@pytest.fixture(scope="module")
def mixed_design_space_without_value() -> DesignSpace:
    """A mixed design space without current value."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-2.0, upper_bound=3.0)
    design_space.add_variable("y", type_="integer", lower_bound=-2, upper_bound=3)
    return design_space


@pytest.mark.parametrize("variable_names", [None, ["x", "y"]])
@pytest.mark.parametrize("complex_to_real", [False, True])
def test_get_current_value_missing_as_dict_not_normalized(
    mixed_design_space_without_value, variable_names, complex_to_real
) -> None:
    """Check the access to a missing current value as a dictionary."""
    assert (
        mixed_design_space_without_value.get_current_value(
            variable_names, complex_to_real, as_dict=True, normalize=False
        )
        == {}
    )


@pytest.mark.parametrize("variable_names", [None, ["x", "y"]])
@pytest.mark.parametrize("complex_to_real", [False, True])
def test_get_current_value_missing_as_array_not_normalized(
    mixed_design_space_without_value, variable_names, complex_to_real
) -> None:
    """Check the access to a missing current value as a NumPy array."""
    with pytest.raises(
        KeyError,
        match=re.escape("There is no current value for the design variables: x and y."),
    ):
        mixed_design_space_without_value.get_current_value(
            variable_names, complex_to_real, as_dict=False, normalize=False
        )


@pytest.mark.parametrize("name", ["x", "y"])
@pytest.mark.parametrize("complex_to_real", [False, True])
@pytest.mark.parametrize("as_dict", [False, True])
def test_get_current_value_missing_partial_not_normalized(
    mixed_design_space_without_value, name, complex_to_real, as_dict
) -> None:
    """Check the access to a missing current value."""
    with pytest.raises(
        KeyError,
        match=re.escape(f"There is no current value for the design variables: {name}."),
    ):
        mixed_design_space_without_value.get_current_value(
            [name], complex_to_real, as_dict, normalize=False
        )


@pytest.mark.parametrize("variable_names", [None, ["x", "y"], ["x"], ["y"]])
@pytest.mark.parametrize("complex_to_real", [False, True])
@pytest.mark.parametrize("as_dict", [False, True])
def test_get_current_value_missing_normalized(
    mixed_design_space_without_value, variable_names, complex_to_real, as_dict
) -> None:
    """Check the access to a missing current value."""
    with pytest.raises(
        KeyError,
        match=re.escape(
            "The current value of a design space cannot be normalized "
            "when some variables have no current value. "
            "There is no current value for the design variables: x and y."
        ),
    ):
        mixed_design_space_without_value.get_current_value(
            variable_names, complex_to_real, as_dict, normalize=True
        )


@pytest.fixture(scope="module")
def mixed_design_space_with_partial_value() -> DesignSpace:
    """A mixed design space with a current value."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-2.0, upper_bound=3.0, value=1.0)
    design_space.add_variable("y", type_="integer", lower_bound=-2, upper_bound=3)
    return design_space


@pytest.mark.parametrize("variable_names", [None, ["x", "y"]])
@pytest.mark.parametrize("complex_to_real", [False, True])
def test_get_current_value_partially_missing_as_dict_not_normalized(
    mixed_design_space_with_partial_value, variable_names, complex_to_real
) -> None:
    """Check the access to a partially missing current value as a dictionary."""
    current_value = mixed_design_space_with_partial_value.get_current_value(
        variable_names, complex_to_real, as_dict=True, normalize=False
    )
    assert current_value.keys() == {"x"}
    assert_equal(current_value["x"], [1.0])


@pytest.mark.parametrize("variable_names", [None, ["x", "y"], ["y"]])
@pytest.mark.parametrize("complex_to_real", [False, True])
def test_get_current_value_partially_missing_as_array_not_normalized(
    mixed_design_space_with_partial_value, variable_names, complex_to_real
) -> None:
    """Check the access to a missing current value as a NumPy array."""
    with pytest.raises(
        KeyError,
        match=re.escape("There is no current value for the design variables: y."),
    ):
        mixed_design_space_with_partial_value.get_current_value(
            variable_names, complex_to_real, as_dict=False, normalize=False
        )


@pytest.mark.parametrize("variable_names", [None, ["x", "y"], ["x"], ["y"]])
@pytest.mark.parametrize("complex_to_real", [False, True])
@pytest.mark.parametrize("as_dict", [False, True])
def test_get_current_value_partially_missing_as_array_normalized(
    mixed_design_space_with_partial_value, variable_names, complex_to_real, as_dict
) -> None:
    """Check the access to a missing current value as a NumPy array."""
    with pytest.raises(
        KeyError,
        match=re.escape(
            "The current value of a design space cannot be normalized "
            "when some variables have no current value. "
            "There is no current value for the design variables: y."
        ),
    ):
        mixed_design_space_with_partial_value.get_current_value(
            variable_names, complex_to_real, as_dict, normalize=True
        )


@pytest.fixture
def mixed_design_space_with_value() -> DesignSpace:
    """A design space with mixed variables."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-2.0, upper_bound=3.0, value=1.0)
    design_space.add_variable(
        "y", type_="integer", lower_bound=-2, upper_bound=3, value=1
    )
    return design_space


@pytest.mark.parametrize(("name", "dtype"), [("x", float64), ("y", int64)])
@pytest.mark.parametrize("complex_to_real", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_get_current_value_as_array_dtype(
    mixed_design_space_with_value, name, complex_to_real, dtype, normalize
) -> None:
    """Check the data type of the current value."""
    assert (
        mixed_design_space_with_value.get_current_value(
            [name], complex_to_real, False, normalize
        ).dtype
        == dtype
    )


@pytest.mark.parametrize(("name", "dtype"), [("x", float64), ("y", int64)])
@pytest.mark.parametrize("complex_to_real", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_get_current_value_as_dict_dtype(
    mixed_design_space_with_value, name, complex_to_real, dtype, normalize
) -> None:
    """Check the data type of the current value."""
    assert (
        mixed_design_space_with_value.get_current_value(
            [name], complex_to_real, True, normalize=normalize
        )[name].dtype
        == dtype
    )


@pytest.mark.parametrize(
    ("l_b", "u_b", "value"),
    [
        (-inf, inf, array([0, 0])),
        (array([1, 2]), inf, array([1, 2])),
        (array([1, 2]), array([2, 4]), array([1, 3])),
        (-inf, array([1, 2]), array([1, 2])),
    ],
)
def test_initialize_missing_current_values(l_b, u_b, value) -> None:
    """Check the initialization of the missing current values."""
    design_space = DesignSpace()
    design_space.add_variable(
        "x", size=2, type_=INTEGER, lower_bound=l_b, upper_bound=u_b
    )
    design_space.initialize_missing_current_values()
    assert_equal(design_space.get_current_value("x"), value)


def test_get_current_value_order() -> None:
    """Check that the order of variables is correctly handled in get_current_value."""
    design_space = DesignSpace()
    design_space.add_variable("x", value=0.0)
    design_space.add_variable("y", value=1.0)
    assert_equal(design_space.get_current_value(), array([0.0, 1.0]))
    assert_equal(
        design_space.get_current_value(variable_names=["x", "y"]),
        array([0.0, 1.0]),
    )
    assert_equal(
        design_space.get_current_value(variable_names=["y", "x"]),
        array([1.0, 0.0]),
    )


def test_export_import_with_none_value(tmp_wd) -> None:
    """Check that a design space exported without default value can be imported."""
    design_space = DesignSpace()
    design_space.add_variable("x")
    design_space.to_csv("foo.csv")
    txt_design_space = DesignSpace.from_csv("foo.csv")
    assert txt_design_space == design_space


def test_repr_html(design_space_2) -> None:
    """Check the HTML representation of a design space."""
    assert design_space_2._repr_html_() == REPR_HTML_WRAPPER.format(
        """Design space:<br/><table>
    <thead>
        <tr>
            <th>Name</th>
            <th>Lower bound</th>
            <th>Value</th>
            <th>Upper bound</th>
            <th>Type</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>x</td>
            <td>-inf</td>
            <td>None</td>
            <td>inf</td>
            <td>float</td>
        </tr>
        <tr>
            <td>y[0]</td>
            <td>-inf</td>
            <td>None</td>
            <td>inf</td>
            <td>float</td>
        </tr>
        <tr>
            <td>y[1]</td>
            <td>-inf</td>
            <td>None</td>
            <td>inf</td>
            <td>float</td>
        </tr>
    </tbody>
</table>""",
    )


def test_normalization_runtimewarning() -> None:
    """Check that normalization does no longer print a RuntimeWarning."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0, upper_bound=2)
    design_space.add_variable("y", lower_bound=1, upper_bound=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        design_space.normalize_vect(array([1.0, 1.0]))


def test_add_variable_from():
    """Check that add_variable_from can add variables from different design spaces."""
    ds1 = DesignSpace()
    ds1.add_variable(
        "x", 2, type_=DesignVariableType.INTEGER, lower_bound=1, upper_bound=3, value=2
    )
    ds2 = DesignSpace()
    ds2.add_variable(
        "y", 3, type_=DesignVariableType.INTEGER, lower_bound=3, upper_bound=5, value=4
    )
    ds2.add_variable("z")

    ds = DesignSpace()
    ds.add_variables_from(ds1, "x")
    ds.add_variables_from(ds2, "z", "y")

    assert ds.variable_names == ["x", "z", "y"]

    assert ds.get_size("x") == 2
    assert ds.get_size("y") == 3
    assert ds.get_size("z") == 1

    assert ds.get_type("x") == DesignVariableType.INTEGER
    assert ds.get_type("y") == DesignVariableType.INTEGER
    assert ds.get_type("z") == DesignVariableType.FLOAT

    assert_equal(ds.get_lower_bound("x"), array([1, 1]))
    assert_equal(ds.get_lower_bound("y"), array([3, 3, 3]))
    assert_equal(ds.get_lower_bound("z"), array([-inf]))

    assert_equal(ds.get_upper_bound("x"), array([3, 3]))
    assert_equal(ds.get_upper_bound("y"), array([5, 5, 5]))
    assert_equal(ds.get_upper_bound("z"), array([inf]))

    assert_equal(ds.get_current_value(["x"]), array([2, 2]))
    assert_equal(ds.get_current_value(["y"]), array([4, 4, 4]))
    assert "z" not in ds._current_value


def test_to_scalar_variables():
    """Check the splitting of design variables into scalar variables."""
    space = DesignSpace()
    space.add_variable("foo", 1, "float", 1, 2)
    space.add_variable("y", 2, "integer", [3, 5], [4, 6], [3, 6])
    new_space = space.to_scalar_variables()
    assert new_space.variable_names == ["foo", "y[0]", "y[1]"]
    assert new_space.get_type("foo") == DesignSpace.DesignVariableType.FLOAT
    assert new_space.get_size("foo") == 1
    assert new_space.get_size("y[0]") == 1
    assert new_space.get_type("y[0]") == DesignSpace.DesignVariableType.INTEGER
    assert new_space.get_size("y[1]") == 1
    assert new_space.get_type("y[1]") == DesignSpace.DesignVariableType.INTEGER
    assert_array_equal(new_space.get_lower_bounds(), [1, 3, 5])
    assert_array_equal(new_space.get_upper_bounds(), [2, 4, 6])
    with pytest.raises(
        KeyError,
        match=re.escape("There is no current value for the design variables: foo."),
    ):
        new_space.get_current_value(["foo"])

    assert_array_equal(new_space.get_current_value(["y[0]", "y[1]"]), [3, 6])


def test_normalize_integer_variables() -> None:
    """Check the normalization of the integer variables."""
    space = DesignSpace()
    space.add_variable("x", type_="integer", lower_bound=-1, upper_bound=3)
    space.enable_integer_variables_normalization = False
    assert space.normalize_vect(array([1])) == pytest.approx(1)
    assert space.unnormalize_vect(array([1])) == pytest.approx(1)
    space.enable_integer_variables_normalization = True
    assert space.normalize_vect(array([1])) == pytest.approx(0.5)
    assert space.unnormalize_vect(array([1])) == pytest.approx(3)


@pytest.mark.parametrize(
    ("type_", "rounded"), [("float", array([0.9])), ("integer", array([1]))]
)
def test_round_vect(type_, rounded) -> None:
    """Check the rounding of a design vector."""
    space = DesignSpace()
    space.add_variable("x", type_=type_)
    assert space.round_vect(array([0.9])) == rounded


@pytest.mark.parametrize(
    "variables",
    [
        (),
        (("y", 1, None),),
        (("x", 2, None),),
        (("x", 1, None),),
        (("x", 1, 2),),
    ],
)
def test_eq(variables) -> None:
    """Check equality of design spaces."""
    space = DesignSpace()
    space.add_variable("x", value=1)
    other = DesignSpace()
    for name, size, value in variables:
        other.add_variable(name, size=size, value=value)

    assert space != other != space


def test_normalize_vect_with_inout_argument() -> None:
    """Check the normalization with the output array passed as input."""
    space = DesignSpace()
    space.add_variable("x", type_="integer", lower_bound=-1, upper_bound=2)
    inout = array([0])
    space.normalize_vect(array([0]), out=inout)
    assert_array_equal(inout, array([0]))
